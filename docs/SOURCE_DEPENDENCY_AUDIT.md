# EGO-Det V2 源码依赖审计（第一阶段）

> 审计对象：私有仓库 `NullParallax/EGO-Det`，以 2026-11-12 已推送的 `main` 初始三次提交为基线。基于用户此前的候选源码 ZIP，已与 GitHub 上模型入口、骨干适配器、评估入口三个 Git Blob SHA 核对一致。本报告仅记录静态审计，不等于已经复现模型的运行结果。

## 1. 结论

- **正式论文模型保留 `ego_det_v2` 主线**；不要直接按“V2是旧版本”删除或用 `ego_joint_object_slot` 的后期分支替换。
- `EgoDetV2` 实例化链依赖两个具有历史命名的顶层包：`stage1` 和 `ego_joint_object_slot`。它们并非全部冗余。
- 核心模型入口闭包 **15** 个 Python 模块；训练入口 **29** 个；评估入口 **26** 个；训练/评估并集 **30** 个（静态导入近似值）。
- 完整候选源码共有 **66** 个（上述三个源代码包中的）Python 模块；`python -m compileall -q ego_det_v2 ego_joint_object_slot stage1 tools` 在本次审计环境完成，但**未执行模型/权重/数据集运行测试**。
- `ego_det_v2/tests/test_paper_visualization.py` 引用了缺失的 `ego_det_v2.paper_visualization` 子包，原始全量源码 ZIP 中也没有相应实现；应将此测试标为待补依赖，而不是宣称整个测试集已通过。

## 2. 正式模型依赖关系

```text
EgoDetV2  (ego_det_v2/models/ego_det_v2.py)
 ├── EgoDetV2Config  (ego_det_v2/configs/config.py)
 ├── EGOJointPaths  (ego_joint_object_slot/configs/paths.py)
 ├── OverLoCKFPNBackbone
 │    ├── ego_joint_object_slot/utils/import_utils.py
 │    └── stage1/multiscale_adapter_v6.py
 │         └── stage1/overlock_official_nofrills.py  [third-party]
 ├── DenseGeometryBranch
 ├── SemanticSupport → SlotSeeder
 ├── HypothesisDenoising  [training only]
 ├── SlotUpdateRound ×3
 ├── EvidenceReadoutRound ×3  [paper configuration]
 ├── SlotPredictionHead ×3
 └── ExistenceHead + compose_scores
```

上述依赖来自 `EgoDetV2.__init__` 与导入图。模型支持 `direct_regression` 对照路径，但论文正式配置 `readout_mode=boundary`，不要将其误当成默认正式路径。

## 3. 训练/评估的静态闭包（完整文件清单）

| 模块 | 主要职责 | 主线使用 | 建议 |
|---|---|---|---|
| `ego_det_v2/configs/config.py` | 正式模型配置与检查 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/data/coco_dataset.py` | COCO读取及训练/验证变换 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/data/collate.py` | Detection batch collate | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/engine/diagnostics.py` | 训练诊断 | 训练 | **保留，不改计算逻辑** |
| `ego_det_v2/engine/ema.py` | 训练中的EMA工具（正式评估为raw） | 训练 | **保留，不改计算逻辑** |
| `ego_det_v2/engine/evaluator.py` | COCO评估与指标 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/engine/geometry_diagnostics.py` | 几何/证据质量诊断 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/engine/train_loop.py` | 训练、恢复和checkpoint逻辑 | 训练 | **保留，不改计算逻辑** |
| `ego_det_v2/launch/_common.py` | 数据加载、路径及device配置 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/launch/launch_eval_ego_det_v2.py` | 正式评估入口 | 评估 | **保留，不改计算逻辑** |
| `ego_det_v2/launch/launch_train_ego_det_v2_e2e.py` | 正式训练入口 | 训练 | **保留，不改计算逻辑** |
| `ego_det_v2/losses/criterion.py` | 总损失和匹配调度 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/losses/dense_losses.py` | Dense分支损失 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/losses/ownership_losses.py` | 证据归属损失 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/losses/slot_losses.py` | Slot相关损失 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/backbone.py` | OverLoCK + FPN与FakeBackbone | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/box_ops.py` | Box基础几何函数 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/dense_geometry_branch.py` | Dense几何候选证据库 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/ego_det_v2.py` | 核心EgoDetV2模型 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/evidence_readout.py` | 基于证据的边界框读取 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/hdn.py` | 训练Denoising | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/heads.py` | 分类、质量与存在性预测 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/matcher.py` | Hungarian匹配 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/semantic_support.py` | 语义支持/区域检索 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/slot_seeding.py` | 候选播种与多样性约束 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_det_v2/models/slot_update.py` | 目标Slot的迭代更新 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_joint_object_slot/configs/paths.py` | 历史路径兼容层 | 训练与评估 | **保留，不改计算逻辑** |
| `ego_joint_object_slot/utils/import_utils.py` | 历史项目路径插入工具 | 训练与评估 | **保留，不改计算逻辑** |
| `stage1/multiscale_adapter_v6.py` | OverLoCK多尺度特征适配 | 训练与评估 | **保留，不改计算逻辑** |
| `stage1/overlock_official_nofrills.py` | 第三方OverLoCK骨干实现，需许可核验 | 训练与评估 | **保留，不改计算逻辑** |

> 注：各 `__init__.py` 是包入口，不一定出现在上述闭包列表中；必须随相应包保留。静态导入树不能证明运行时路径全覆盖。

## 4. 已发现的整理风险

**高风险（在权重严格加载与复现前保持不动）**

1. `EgoDetV2.__init__` 注册了 `backbone`、`dense_geometry`、`semantic_support`、`slot_seeding`、`hdn`、`slot_updates`、`readouts`、`round_heads`、`existence_head` 等模块。改这些**属性名**将改变 `state_dict` 的 key 前缀。
2. `ego_det_v2/models/backbone.py` 仍需 `stage1` 的 OverLoCK 适配，模型构建时依赖预训练权重路径。评估工具 `launch_eval_ego_det_v2.py` 先构造模型、再用 `model.load_state_dict(state)`（默认 strict=True）加载正式权重。
3. `EGOJointPaths` 依赖 `EGO_JOINT_PROJECT_ROOT` 或项目根路径，并包含历史 teacher/proposal 字段。应在下一阶段拆分兼容层，切勿简单删掉整个包。
4. 当前 `EgoDetV2Config` 的默认 Batch Size = 4、Accum = 4、训练尺度中包括 640/672，与最终论文记录中的 Batch Size = 8、Accum = 2 和训练尺度不完全一致。复现必须依赖显式论文参数，而非默认值。

**需要修复，但不是正式推理主链阻塞**

1. `ego_det_v2/tests/test_paper_visualization.py` 导入的 `ego_det_v2.paper_visualization` 不存在。既不应直接删除历史测试，也不应把它纳入“全部通过”证明；建议未来单独恢复实现或迁入历史归档。
2. `stage1/overlock_official_nofrills.py` 自述来自 OverLoCK 官方实现，公开发布前应核验**第三方源码许可**和保留原始版权声明/致谢。
3. README 现阶段仍应保持 `REPRODUCIBILITY STAGING`，直到权重 SHA256、严格加载、完整 COCO 评估通过。

## 5. 建议的重构顺序与 Git 提交边界

| 阶段 | 允许的变动 | 检查要求 |
|---|---|---|
| R0：源码审计（本次） | 仅新增本报告和静态审计工具 | 无 PyTorch 环境即可运行，核心模块静态导入完整 |
| R1：外部 API | 提供 `egodet` 的公开导入别名，保留 `ego_det_v2` 的真实实现和旧命令 | 检查导入与 CLI 帮助信息 |
| R2：路径去耦 | 逐步移除 `ego_joint_object_slot` 路径工具对旧工程名的绑定，保留向后兼容 | 单元测试、参数和配置指纹比对 |
| R3：目录命名 | 将 `stage1` 的必要适配器移动到清晰的 backbone 目录，同时保留 checkpoint 的子模块属性层级 | `strict=True` 权重加载 + 相同输入输出对比 |
| R4：统一 Train/Eval/Demo | 统一对外 CLI 和文档，不替换核心模型 | 重新评估 COCO val2017、目标 AP≈45.269 |
| R5：公开发布 | 核验第三方授权、许可、权重、引用、作者审批 | 可从空环境复现且不包含密钥 |

**不建议做的事：** 在同一个 Commit 中同时改模型的 `nn.Module` 属性名、文件路径和损失函数；也不要把数据目录或任何 Checkpoint 提交到 Git。

## 6. 本次提供的只读工具

```bash
cd ~/projects/EGO-Det
python tools/check_source_layout.py
python -m compileall -q ego_det_v2 ego_joint_object_slot stage1 tools
git status --short
```

`check_source_layout.py` 使用 Python 标准库 AST，不需要环境中安装 PyTorch。它会对缺失的可视化测试包输出 `WARN`，对其他本地缺失导入输出 `ERROR`；目前只检查静态导入而不验证动态导入、外部依赖和运行行为。

## 7. 版本与实验身份证据

- 正式 checkpoint 候选：`outputs/ego_det_v2/train_mod10b_batch8_long/epoch_12.pth`；历史记录大小为 562,663,950 bytes，尚待在本机独立比对哈希。
- 历史 COCO val2017 raw 模型：AP 45.2689、AP50 63.3310、AP75 49.4821；该记录是论文跑出的结果，不代表当前 WSL 仓库已经复现。
- 模型主线与 V1.1.4 Unified 是两条不同的实现，不可混用 checkpoint 或论文指标。

**后续首个涉及模型行为的改动，必须先保存严格加载时的 missing_keys/unexpected_keys（均应为空）与相同输入下的预测结果，作为回归基线。**
