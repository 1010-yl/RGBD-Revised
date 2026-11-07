# RGB-D 返修：原始代码静态审计（2026-11-02）

> 已检查用户提供的 `UMFNet-checkpoint.py`, `SwinTransformers-checkpoint.py`, `UMFNet_train.py`, `UMFNet_test.py`, `data.py`, `split_data.py`, `options.py`, `run_umfnet_train.sh`。此处所有“原代码”均指这些源文件；未检查服务器上实际 `models/UMFNet.py` 的文件哈希及历史权重。`legacy/` 保存的是上传版本，原始代码不被覆盖。

## 1. 高优先级证据链

| 严重性 | 原文件与位置 | 证据 | 后果/处理 |
| --- | --- | --- | --- |
| 高 | `UMFNet-checkpoint.py:91-137` | UAM 独立建立 RGB/T GaussianHead、CrossModalGaussianHead，并映射出 `F_t_tilde`。 | 与 Wang et al. (CVPR 2026) RGB-T 方法核心高度类似；在 RGB-D 中需移除跨模态“对齐”动机，另建深度噪声学习的条件分布。 |
| 高 | `UMFNet-checkpoint.py:165-192` | CGM 使用 confidence generator、channel scaling/bias、spatial gating。 | 不能简单重命名；使用功能不同的深度专属结构处理。 |
| 高 | `UMFNet-checkpoint.py:231-286` | 双 Swin 分支；四个尺度全执行 UAM→CGM。 | 需要改变主干或交互方式、模块位置。 |
| 高 | `SwinTransformers-checkpoint.py:435-460` 与 `UMFNet-checkpoint.py:273-285` | Backbone 保存 patch embedding + 四个 SwinLayer，共五个特征；主模型只取前四个。 | 最后一个 Swin Layer 被执行却没有被用到预测；修订版改成每一级 Swin stage 的 **downsample 前** 特征并接入最终 stage。 |
| 高 | `UMFNet_train.py:281-309` | 每轮在 `test_loader` 上算 MAE，并保存 best 模型。 | 若该目录为最终测试集，发生模型选择泄漏；修订版训练入口只在 val 选 best，test 必须独立评估。 |
| 高 | `data.py:285-293`、`data.py:381-389` | depth 以 `convert('L')` 加载。 | 高位深物理深度被降到 8 bit；无法据此直接讨论物理传感器噪声幅度。修订版读取 8/16 位单通道，并显式创建有效性掩码。 |
| 高 | `split_data.py:14-17` | 发现输出目录存在就 `shutil.rmtree(dst_base)`。 | 有误删实验划分/日志风险；**不要运行原 split_data.py**。修订版安全创建新的 train/val/test 目录，不覆盖旧数据。 |
| 中 | `UMFNet_train.py:302-307` | 首轮 `epoch == 1` 更新 best_mae，但不保存 best checkpoint。 | 如果首轮是最优轮次，best 可能不存在。修订版比较 `val_mae<inf` 后立即保存。 |
| 中 | `UMFNet_train.py:193-210` 和 `UMFNet_test.py:117-123` | depth `.repeat(1,3,1,1)` 适配 RGB 型 Swin。 | 修订版独立 Depth CNN 接收真实单通道深度，不能再 repeat。 |
| 中 | `options.py:17-23`, `run_umfnet_train.sh:31-52` | `image_root='all'` 默认 UVT20K/UNVT5000，启动脚本未传递自定义训练 depth/GT 路径。 | 默认训练来源是 RGB-T，并非已确认的 NJU2000 RGB-D；修订版明确指定 train/val/test 文件夹。 |
| 中 | `UMFNet_test.py:80-83, 125-145` | 指标 S/E/Fw 用逐样本 min-max 归一化结果，MAE 用 sigmoid 概率。 | 两类计算采用不同的后处理口径，论文需披露并在所有模型上保持一致。 |
| 中 | `draw_overlay.py:22-35` | 随机抽选 5 张图展示 Ours，没有其他方法列。 | 不能证明审稿人要求的定性比较；需从真实预测及配对样本生成多方法结果图。 |
| 中 | Table 1 | 审稿人质疑全部 `±` 的来源。 | 未找到模型多种子统计代码；必须核实数字原始来源，不能编造标准差。 |

## 2. 模块与公式归属

Wang 等人的 UMFNet（CVPR 2026）提出 UAM、CGM、跨模态概率对齐和置信度调制。**原论文必须在 Related Work、Method、Reviewer Response 中明确引用**。新研究在改造 RGB-D 场景时，应以可复现实验结果支持新增贡献，而不能把原有高斯重参数化、KL 和全局调制表述成自主发明。

## 3. 本离线包的处置范围

1. 保留原始参考文件于 `legacy/`；完全不覆盖原工程。
2. 新的 `models/SwinTransformers.py` 修复 stage 输出定义，模块形状可做 CPU 级测试。
3. 新的 `models/RGBDRevised.py` 是 **研究原型**：RGB Swin、轻量 Depth CNN、RGB/Depth 边缘先验、条件深度高斯 NLL、浅层结构引导、阶段 2/3 动态融合、阶段 4 RGB 语义保留。
4. 新的 `make_safe_nju_split.py` 不清空已有目录，可保留旧测试划分；新 `data_rgbd.py` 保留深度有效性掩码并在 train 施加模拟噪声。
5. 新 `train_rgbd_revised.py` 只根据独立 val 选择 checkpoint；新 `evaluate_rgbd.py` 专门用于最终 test 和噪声鲁棒性测试。
6. 配套 CPU 测试**不能**替代 GPU 真实环境、真正训练、指标可信性或加速结果。

## 4. 未完成事项（需 GPU 或原数据）

- 检查服务器真实源码哈希；真实 NJU2000 的 depth 文件像素格式及量纲；现有模型权重是否能被 Swin backbone 部分载入。
- 真实 384×384 Swin-B 正向、反向、显存；完整训练结果及 random seed 统计。
- 新 DAPF 的 train soft / eval hard 路由差异，实际计算减少与精度损失；GFLOPs 和 FPS。
- 噪声鲁棒性、校准及失败案例；与公开 RGB-D 方法的公平比较。
- 根据真实实验数值修订论文表格和 rebuttal，不能填造任何指标。
