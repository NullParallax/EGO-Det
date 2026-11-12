from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..configs.config import EgoDetV2Config
from .box_ops import absolute_xyxy_to_normalized_cxcywh, pairwise_box_iou


def _gather_bank(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, count = indices.shape
    trailing = values.shape[2:]
    gather_index = indices.view(batch, count, *([1] * len(trailing))).expand(
        batch, count, *trailing
    )
    return torch.gather(values, 1, gather_index)


class SlotSeeder(nn.Module):
    def __init__(self, config: EgoDetV2Config) -> None:
        super().__init__()
        self.config = config
        channels = int(config.d_model)
        self.seed_mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
        )

    @torch.no_grad()
    def select_indices_reference(
        self,
        bank: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validated MOD9A selector kept verbatim as the equivalence oracle."""

        boxes = bank["boxes"].float()
        valid = bank["valid"]
        scores = bank["score"].squeeze(-1)
        batch, _ = valid.shape
        slot_count = int(self.config.num_slots)
        indices = torch.zeros((batch, slot_count), dtype=torch.long, device=boxes.device)
        seed_valid = torch.zeros((batch, slot_count), dtype=torch.bool, device=boxes.device)
        thresholds = (float(self.config.seed_diversity_iou), 0.85, 1.01)

        for batch_index in range(batch):
            candidates = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(scores[batch_index, candidates], descending=True)
            candidates = candidates[order]
            chosen: List[int] = []
            chosen_set = set()
            for threshold in thresholds:
                for candidate_tensor in candidates:
                    candidate = int(candidate_tensor)
                    if candidate in chosen_set:
                        continue
                    if not chosen:
                        accept = True
                    elif threshold > 1.0:
                        accept = True
                    else:
                        chosen_tensor = torch.tensor(chosen, device=boxes.device, dtype=torch.long)
                        ious, _ = pairwise_box_iou(
                            boxes[batch_index, candidate : candidate + 1],
                            boxes[batch_index, chosen_tensor],
                        )
                        accept = bool((ious[0] < threshold).all())
                    if accept:
                        chosen.append(candidate)
                        chosen_set.add(candidate)
                        if len(chosen) == slot_count:
                            break
                if len(chosen) == slot_count:
                    break

            actual = len(chosen)
            if actual:
                if actual < slot_count:
                    repeats = [chosen[index % actual] for index in range(slot_count - actual)]
                    chosen.extend(repeats)
                indices[batch_index] = torch.tensor(chosen[:slot_count], device=boxes.device)
                seed_valid[batch_index, : min(actual, slot_count)] = True
        return indices, seed_valid

    @torch.no_grad()
    def select_indices_optimized(
        self,
        bank: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the exact reference sequence with a batched greedy scan.

        Valid-subset construction and score sorting intentionally remain
        per-image so ``torch.argsort`` sees exactly the same tensor as the
        reference oracle.  The greedy state is then advanced for every image
        in parallel.  Each iteration computes only the selected box against
        the K sorted boxes; no K-by-K IoU matrix is materialized.
        """

        boxes = bank["boxes"].float()
        valid = bank["valid"]
        scores = bank["score"].squeeze(-1)
        batch, bank_size = valid.shape
        slot_count = int(self.config.num_slots)
        indices = torch.zeros(
            (batch, slot_count),
            dtype=torch.long,
            device=boxes.device,
        )
        seed_valid = torch.zeros(
            (batch, slot_count),
            dtype=torch.bool,
            device=boxes.device,
        )
        if bank_size == 0:
            return indices, seed_valid

        thresholds = boxes.new_tensor(
            (float(self.config.seed_diversity_iou), 0.85, 1.01),
            dtype=torch.float32,
        )

        # Preserve the reference ordering exactly: nonzero produces ascending
        # global indices and argsort receives only the valid score subset.
        sorted_candidates = torch.zeros(
            (batch, bank_size),
            dtype=torch.long,
            device=boxes.device,
        )
        sorted_boxes = torch.zeros(
            (batch, bank_size, 4),
            dtype=torch.float32,
            device=boxes.device,
        )
        remaining = torch.zeros(
            (batch, bank_size),
            dtype=torch.bool,
            device=boxes.device,
        )
        max_actual = 0
        for batch_index in range(batch):
            candidates = torch.nonzero(
                valid[batch_index],
                as_tuple=False,
            ).flatten()
            order = torch.argsort(
                scores[batch_index, candidates],
                descending=True,
            )
            candidates = candidates[order]
            candidate_count = candidates.shape[0]
            sorted_candidates[
                batch_index,
                :candidate_count,
            ].copy_(candidates)
            sorted_boxes[
                batch_index,
                :candidate_count,
            ].copy_(boxes[batch_index].index_select(0, candidates))
            remaining[batch_index, :candidate_count] = True
            max_actual = max(
                max_actual,
                min(candidate_count, slot_count),
            )

        actual = valid.sum(dim=1, dtype=torch.long).clamp(
            max=slot_count,
        )
        max_iou_to_chosen = torch.zeros(
            (batch, bank_size),
            dtype=torch.float32,
            device=boxes.device,
        )
        chosen = torch.zeros(
            (batch, slot_count),
            dtype=torch.long,
            device=boxes.device,
        )
        chosen_count = torch.zeros(
            batch,
            dtype=torch.long,
            device=boxes.device,
        )
        threshold_index = torch.zeros(
            batch,
            dtype=torch.long,
            device=boxes.device,
        )
        cursor = torch.zeros(
            batch,
            dtype=torch.long,
            device=boxes.device,
        )

        # Persistent workspaces avoid allocating B-by-K masks and IoU
        # intermediates on every greedy iteration.
        rank = torch.arange(
            bank_size,
            dtype=torch.long,
            device=boxes.device,
        ).view(1, bank_size)
        eligible = torch.empty(
            (batch, bank_size),
            dtype=torch.bool,
            device=boxes.device,
        )
        mask_buffer = torch.empty_like(eligible)
        eligible_u8 = torch.empty(
            (batch, bank_size),
            dtype=torch.uint8,
            device=boxes.device,
        )
        coordinate_buffer = torch.empty(
            (batch, bank_size),
            dtype=torch.float32,
            device=boxes.device,
        )
        intersection_buffer = torch.empty_like(coordinate_buffer)
        union_buffer = torch.empty_like(coordinate_buffer)
        sorted_wh = (
            sorted_boxes[..., 2:] - sorted_boxes[..., :2]
        ).clamp(min=0)
        sorted_area = sorted_wh[..., 0] * sorted_wh[..., 1]
        del sorted_wh

        # One accepted candidate or one threshold transition is performed per
        # iteration.  The worst case is max(actual) selections plus the two
        # transitions before the unconditional threshold.
        for _ in range(max_actual + 2):
            active = chosen_count < actual
            threshold = thresholds[
                threshold_index.clamp(max=2)
            ]

            torch.lt(
                max_iou_to_chosen,
                threshold[:, None],
                out=eligible,
            )
            unconditional = chosen_count.eq(0)
            unconditional.logical_or_(threshold_index.ge(2))
            eligible.logical_or_(unconditional[:, None])
            eligible.logical_and_(remaining)
            torch.ge(
                rank,
                cursor[:, None],
                out=mask_buffer,
            )
            eligible.logical_and_(mask_buffer)
            eligible.logical_and_(active[:, None])

            # max on uint8 returns the first rank carrying value 1.  Rows with
            # no eligible candidate return value 0 and are threshold advances.
            eligible_u8.copy_(eligible)
            has_eligible, selected_rank = eligible_u8.max(dim=1)
            take = has_eligible.to(torch.bool)
            selected_rank_2d = selected_rank[:, None]

            write_index = chosen_count.clamp(
                max=slot_count - 1,
            )[:, None]
            previous_chosen = chosen.gather(1, write_index)
            selected_global = sorted_candidates.gather(
                1,
                selected_rank_2d,
            )
            chosen.scatter_(
                1,
                write_index,
                torch.where(
                    take[:, None],
                    selected_global,
                    previous_chosen,
                ),
            )

            previous_remaining = remaining.gather(
                1,
                selected_rank_2d,
            )
            remaining.scatter_(
                1,
                selected_rank_2d,
                previous_remaining & ~take[:, None],
            )

            selected_box_index = selected_rank.view(
                batch,
                1,
                1,
            ).expand(batch, 1, 4)
            selected_box = sorted_boxes.gather(
                1,
                selected_box_index,
            )
            selected_area = sorted_area.gather(
                1,
                selected_rank_2d,
            )

            # Exact selected-box-to-all-candidates IoU using preallocated
            # B-by-K buffers.  The operation order matches pairwise_box_iou.
            torch.maximum(
                selected_box[:, :, 0],
                sorted_boxes[..., 0],
                out=coordinate_buffer,
            )
            torch.minimum(
                selected_box[:, :, 2],
                sorted_boxes[..., 2],
                out=intersection_buffer,
            )
            intersection_buffer.sub_(coordinate_buffer).clamp_(min=0)
            torch.maximum(
                selected_box[:, :, 1],
                sorted_boxes[..., 1],
                out=coordinate_buffer,
            )
            torch.minimum(
                selected_box[:, :, 3],
                sorted_boxes[..., 3],
                out=union_buffer,
            )
            union_buffer.sub_(coordinate_buffer).clamp_(min=0)
            intersection_buffer.mul_(union_buffer)
            torch.add(
                selected_area,
                sorted_area,
                out=union_buffer,
            )
            union_buffer.sub_(intersection_buffer).clamp_(min=1.0e-7)
            intersection_buffer.div_(union_buffer)
            intersection_buffer.masked_fill_(
                ~take[:, None],
                -1.0,
            )
            torch.maximum(
                max_iou_to_chosen,
                intersection_buffer,
                out=max_iou_to_chosen,
            )

            chosen_count.add_(take.to(torch.long))
            advance_threshold = active & ~take
            cursor = torch.where(
                take,
                selected_rank + 1,
                torch.where(
                    advance_threshold,
                    torch.zeros_like(cursor),
                    cursor,
                ),
            )
            threshold_index.add_(
                advance_threshold.to(torch.long)
            )

        slot_rank = torch.arange(
            slot_count,
            dtype=torch.long,
            device=boxes.device,
        ).view(1, slot_count)
        repeat_positions = torch.remainder(
            slot_rank,
            actual.clamp(min=1)[:, None],
        )
        indices.copy_(chosen.gather(1, repeat_positions))
        seed_valid.copy_(slot_rank < actual[:, None])
        return indices, seed_valid

    @torch.no_grad()
    def select_indices(
        self,
        bank: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.seed_selection_mode == "top_score":
            return self.select_indices_top_score(bank)
        if (
            self.config.efficiency_mode == "optimized"
            and int(self.config.efficiency_ablation_level) >= 2
        ):
            return self.select_indices_optimized(bank)
        return self.select_indices_reference(bank)

    @torch.no_grad()
    def select_indices_top_score(
        self,
        bank: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the highest-scoring valid proposals without IoU diversity."""

        valid = bank["valid"]
        scores = bank["score"].squeeze(-1)
        batch = int(valid.shape[0])
        slot_count = int(self.config.num_slots)
        indices = torch.zeros(
            (batch, slot_count),
            dtype=torch.long,
            device=valid.device,
        )
        seed_valid = torch.zeros(
            (batch, slot_count),
            dtype=torch.bool,
            device=valid.device,
        )
        for batch_index in range(batch):
            candidates = torch.nonzero(
                valid[batch_index],
                as_tuple=False,
            ).flatten()
            if candidates.numel() == 0:
                continue
            order = torch.argsort(
                scores[batch_index, candidates],
                descending=True,
            )
            chosen = candidates[order][:slot_count]
            actual = int(chosen.numel())
            if actual < slot_count:
                repeat_positions = torch.arange(
                    slot_count,
                    device=valid.device,
                ) % max(actual, 1)
                chosen = chosen[repeat_positions]
            indices[batch_index] = chosen[:slot_count]
            seed_valid[batch_index, :actual] = True
        return indices, seed_valid

    def forward(
        self,
        bank: Dict[str, torch.Tensor],
        semantic_support,
        semantic_state: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            seed_indices, seed_valid = self.select_indices(bank)
        seed_boxes_abs = _gather_bank(bank["boxes"].float(), seed_indices)
        seed_desc = _gather_bank(bank["desc"], seed_indices)
        b0 = absolute_xyxy_to_normalized_cxcywh(
            seed_boxes_abs,
            bank["image_sizes"],
        ).detach().float()

        no_bank = ~bank["valid"].any(dim=1)
        if (
            self.config.efficiency_mode == "optimized"
            and int(self.config.efficiency_ablation_level) >= 2
        ):
            default_box = b0.new_tensor([0.5, 0.5, 0.1, 0.1])
            b0 = torch.where(
                no_bank[:, None, None],
                default_box.view(1, 1, 4),
                b0,
            )
            scales = torch.stack(
                (
                    bank["image_sizes"][:, 1],
                    bank["image_sizes"][:, 0],
                    bank["image_sizes"][:, 1],
                    bank["image_sizes"][:, 0],
                ),
                dim=-1,
            ).to(device=b0.device, dtype=b0.dtype)
            from .box_ops import box_cxcywh_to_xyxy

            recomputed_boxes_abs = (
                box_cxcywh_to_xyxy(b0) * scales[:, None]
            )
            # The reference recomputes every batch item when any image has an
            # empty bank. Preserve that exact mixed-batch behavior without
            # converting ``no_bank.any()`` to a Python bool.
            seed_boxes_abs = torch.where(
                no_bank.any(),
                recomputed_boxes_abs,
                seed_boxes_abs,
            )
            seed_desc = seed_desc.masked_fill(
                no_bank[:, None, None],
                0,
            )
        elif no_bank.any():
            default_box = b0.new_tensor([0.5, 0.5, 0.1, 0.1])
            b0[no_bank] = default_box
            scales = torch.stack(
                (
                    bank["image_sizes"][:, 1],
                    bank["image_sizes"][:, 0],
                    bank["image_sizes"][:, 1],
                    bank["image_sizes"][:, 0],
                ),
                dim=-1,
            ).to(device=b0.device, dtype=b0.dtype)
            from .box_ops import box_cxcywh_to_xyxy

            seed_boxes_abs = box_cxcywh_to_xyxy(b0) * scales[:, None]
            seed_desc = seed_desc.clone()
            seed_desc[no_bank] = 0

        if self.config.semantic_regional_retrieval:
            retrieved = semantic_support.retrieve(
                semantic_state,
                seed_boxes_abs,
            )
        else:
            retrieved = torch.zeros_like(seed_desc)
        h0 = self.seed_mlp(torch.cat((seed_desc, retrieved), dim=-1))
        s0 = h0
        c0 = h0
        return b0, s0, c0, seed_indices, seed_valid


@torch.no_grad()
def run_slot_seeder_cuda_performance_contract(
    *,
    device: str | torch.device = "cuda:0",
    random_trials: int = 100,
    warmup_iterations: int = 10,
    measure_iterations: int = 100,
) -> Dict[str, object]:
    """Validate and time the formal B=8, K=1000, N=300 CUDA selector.

    This opt-in contract lives beside the selector so the production change
    remains confined to this module.  Correctness uses the reference selector
    on the same CUDA inputs; only optimized steady-state calls are timed.
    """

    contract_device = torch.device(device)
    if contract_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the SlotSeeder performance contract requires CUDA")
    if int(random_trials) < 100:
        raise ValueError("random_trials must be >= 100")
    if int(warmup_iterations) < 1 or int(measure_iterations) < 1:
        raise ValueError("warmup_iterations and measure_iterations must be >= 1")

    batch = 8
    bank_size = 1000
    slot_count = 300
    config = EgoDetV2Config(
        efficiency_mode="optimized",
        batch_size=batch,
        grad_accum_steps=2,
    )
    if (
        int(config.bank_topk_train) != bank_size
        or int(config.num_slots) != slot_count
    ):
        raise AssertionError(
            "formal Seeder contract requires K_train=1000 and N=300"
        )
    seeder = SlotSeeder(config).to(contract_device).eval()
    generator = torch.Generator(device=contract_device)
    generator.manual_seed(1701)
    rank = torch.arange(
        bank_size,
        dtype=torch.long,
        device=contract_device,
    ).view(1, bank_size)

    def make_bank(trial: int) -> Dict[str, torch.Tensor]:
        raw = torch.rand(
            batch,
            bank_size,
            4,
            generator=generator,
            device=contract_device,
        )
        lower = torch.minimum(raw[..., :2], raw[..., 2:])
        upper = torch.maximum(raw[..., :2], raw[..., 2:]) + 0.01
        boxes = torch.cat((lower, upper), dim=-1)
        scores = torch.randn(
            batch,
            bank_size,
            1,
            generator=generator,
            device=contract_device,
        )
        valid = (
            torch.rand(
                batch,
                bank_size,
                generator=generator,
                device=contract_device,
            )
            > 0.15
        )

        # Every contract run covers a fully empty image and a candidate-short
        # image.  Dedicated trials additionally cover all-empty banks, score
        # ties, and completely overlapping boxes.
        valid[0].zero_()
        shortage = (trial * 37) % slot_count
        valid[1].copy_(rank[0] < shortage)
        if trial == 0:
            valid.zero_()
        elif trial == 1:
            counts = torch.tensor(
                (0, 1, 2, 50, 299, 300, 999, 1000),
                dtype=torch.long,
                device=contract_device,
            )
            valid.copy_(rank < counts[:, None])
        if trial % 10 == 2:
            scores.zero_()
        if trial % 10 == 3:
            boxes.copy_(boxes[:, :1].expand_as(boxes))
        return {"boxes": boxes, "score": scores, "valid": valid}

    for trial in range(int(random_trials)):
        bank = make_bank(trial)
        reference_indices, reference_valid = (
            seeder.select_indices_reference(bank)
        )
        optimized_indices, optimized_valid = (
            seeder.select_indices_optimized(bank)
        )
        if not torch.equal(reference_indices, optimized_indices):
            raise AssertionError(
                f"Seeder index mismatch in CUDA random trial {trial}"
            )
        if not torch.equal(reference_valid, optimized_valid):
            raise AssertionError(
                f"seed_valid mismatch in CUDA random trial {trial}"
            )

    benchmark_bank = make_bank(int(random_trials) + 1)
    benchmark_bank["valid"].fill_(True)
    for _ in range(int(warmup_iterations)):
        seeder.select_indices_optimized(benchmark_bank)
    torch.cuda.synchronize(contract_device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(int(measure_iterations)):
        seeder.select_indices_optimized(benchmark_bank)
    end_event.record()
    torch.cuda.synchronize(contract_device)
    wall_seconds = time.perf_counter() - wall_start
    cuda_ms = (
        float(start_event.elapsed_time(end_event))
        / float(measure_iterations)
    )
    wall_ms = (
        wall_seconds * 1000.0 / float(measure_iterations)
    )
    result: Dict[str, object] = {
        "status": "PASS" if wall_ms < 200.0 else "FAIL",
        "batch": batch,
        "bank_size": bank_size,
        "slot_count": slot_count,
        "random_trials": int(random_trials),
        "warmup_iterations": int(warmup_iterations),
        "measure_iterations": int(measure_iterations),
        "cuda_ms_per_call": cuda_ms,
        "wall_ms_per_call": wall_ms,
        "target_under_200ms": wall_ms < 200.0,
        "recommended_under_100ms": wall_ms < 100.0,
        "device": str(contract_device),
        "device_name": torch.cuda.get_device_name(contract_device),
    }
    if wall_ms >= 200.0:
        raise AssertionError(
            "batch8 SlotSeeder steady-state wall time "
            f"{wall_ms:.3f} ms is not below the 200 ms contract"
        )
    return result
