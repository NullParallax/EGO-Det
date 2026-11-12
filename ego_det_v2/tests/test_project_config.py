"""Contract tests for the user-facing YAML configuration."""
from __future__ import annotations
import tempfile
import unittest
from dataclasses import fields, asdict
from copy import deepcopy
from io import StringIO
from contextlib import redirect_stdout
from unittest import mock
import sys
from pathlib import Path

from ego_det_v2.configs.config import EgoDetV2Config
from ego_det_v2.configs.project import (
    DEFAULT_CONFIG, evaluation_model_config, load_settings,
    model_from_settings, path_from_settings, resolve_root,
)

class TestUserConfig(unittest.TestCase):
    def test_project_yaml_can_build_paper_model(self):
        cfg = model_from_settings(load_settings())
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.grad_accum_steps, 2)
        self.assertEqual(cfg.num_slots, 300)
        self.assertEqual(cfg.bank_topk_eval, 1500)
        self.assertEqual(set(load_settings()["model"]), {field.name for field in fields(EgoDetV2Config)})
        self.assertEqual(cfg.train_short_sides, (480,512,544,576,608))

    def test_paths_are_root_relative(self):
        settings = load_settings()
        actual = path_from_settings(settings, "val_annotations", resolve_root())
        self.assertEqual(actual, resolve_root() / "data/coco/raw/annotations/instances_val2017.json")

    def test_no_silent_unknown_model_param(self):
        with self.assertRaisesRegex(ValueError,"Unrecognized EGO-Det model"):
            model_from_settings({"model": {"batch_szie": 8}})

    def test_checkpoint_architecture_not_overridden_by_yaml(self):
        settings = load_settings()
        saved = EgoDetV2Config(batch_size=3)
        resolved = evaluation_model_config(saved,settings)
        self.assertEqual(resolved.batch_size,3)
        self.assertEqual(resolved.num_slots,saved.num_slots)
        settings["evaluation"]["model_overrides"] = {"num_slots":100}
        with self.assertRaises(ValueError):
            evaluation_model_config(saved,settings)

    def test_all_dataclass_fields_configurable(self):
        cfg=model_from_settings({"model": {"num_slots":300,"slot_rounds":3,"num_workers":2}})
        self.assertEqual(cfg.num_workers,2)

    def test_invalid_yaml_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            fn=Path(td)/"bad.yaml"
            fn.write_text("model:\n  unknown: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unrecognized EGO-Det model"):
                model_from_settings(load_settings(fn))


class TestLaunchersReadYaml(unittest.TestCase):
    """No GPU/data required: verify CLI/config plumbing with mocked model engines."""

    def test_train_uses_config_paths_and_cli_overrides(self):
        from ego_det_v2.launch import launch_train_ego_det_v2_e2e as launcher
        args = ["train.py", "--batch-size", "2", "--grad-accum", "8"]
        fake_loader = (object(), object())
        with (
            mock.patch.object(sys, "argv", args),
            mock.patch.object(launcher, "make_loader", return_value=fake_loader) as loader,
            mock.patch.object(launcher, "EgoDetV2") as model,
            mock.patch.object(launcher, "EgoDetV2Criterion"),
            mock.patch.object(launcher, "fit") as fit,
            mock.patch.object(launcher, "choose_device", return_value="cpu"),
            mock.patch.object(launcher, "set_runtime_defaults"),
            redirect_stdout(StringIO()),
        ):
            launcher.main()
        kwargs = fit.call_args.kwargs
        self.assertEqual(kwargs["config"].batch_size, 2)
        self.assertEqual(kwargs["config"].grad_accum_steps, 8)
        self.assertEqual(kwargs["config"].epochs, 12)
        self.assertTrue(str(kwargs["config"].output_dir).startswith(str(resolve_root())))
        self.assertEqual(loader.call_args_list[0].kwargs["annotation_file"],
                         resolve_root()/"data/coco/raw/annotations/instances_train2017.json")
        self.assertEqual(model.call_args.kwargs["paths"].overlock_weights,
                         resolve_root()/"weights/overlock/overlock_t_in1k_224.pth")

    def test_eval_uses_checkpoint_architecture_and_runtime_batch(self):
        from ego_det_v2.launch import launch_eval_ego_det_v2 as launcher
        settings = deepcopy(load_settings())
        settings["evaluation"]["metrics_json"] = None
        settings["evaluation"]["predictions_json"] = None
        stored = EgoDetV2Config(batch_size=7).to_dict()
        with (
            mock.patch.object(sys, "argv", ["eval.py", "--batch-size", "1", "--subset-size", "10"]),
            mock.patch.object(launcher, "load_settings", return_value=settings),
            mock.patch.object(launcher.torch, "load", return_value={"model":{"w":"dummy"}, "config":stored}),
            mock.patch.object(launcher, "EgoDetV2") as model,
            mock.patch.object(launcher, "EgoDetV2Criterion"),
            mock.patch.object(launcher, "make_loader", return_value=(object(), object())) as loader,
            mock.patch.object(launcher, "evaluate", return_value={"AP":0.4527}) as evaluate,
            mock.patch.object(launcher, "choose_device", return_value="cpu"),
            redirect_stdout(StringIO()),
        ):
            launcher.main()
        cfg = model.call_args.args[0]
        self.assertEqual(cfg.batch_size,7)
        self.assertEqual(loader.call_args.kwargs["batch_size"],1)
        self.assertEqual(loader.call_args.kwargs["num_workers"],0)
        self.assertEqual(evaluate.call_args.kwargs["subset_size"],10)
        model.return_value.load_state_dict.assert_called_once_with({"w":"dummy"}, strict=True)

    def test_eval_does_not_silently_ignore_coco_error(self):
        from ego_det_v2.launch import launch_eval_ego_det_v2 as launcher
        settings = deepcopy(load_settings())
        settings["evaluation"]["metrics_json"] = None
        settings["evaluation"]["predictions_json"] = None
        with (
            mock.patch.object(sys, "argv", ["eval.py"]),
            mock.patch.object(launcher, "load_settings", return_value=settings),
            mock.patch.object(launcher.torch, "load", return_value={"model":{}, "config":{}}),
            mock.patch.object(launcher, "EgoDetV2"),
            mock.patch.object(launcher, "EgoDetV2Criterion"),
            mock.patch.object(launcher, "make_loader", return_value=(object(), object())),
            mock.patch.object(launcher, "evaluate", return_value={"coco_eval_error": "simulated failure"}),
            mock.patch.object(launcher, "choose_device", return_value="cpu"),
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                launcher.main()

if __name__=="__main__": unittest.main()
