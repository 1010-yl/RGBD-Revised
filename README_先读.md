# RGB-D 论文返修：离线预备包 v1

## 当前状态

**已完成**：原源码静态审计；新原型网络源码；修复 Swin 最后一级阶段特征未送入预测的问题；安全 train/val/test 划分工具；深度掩码与模拟噪声数据管线；新训练和独立测试入口；论文 Method 草稿及审稿回复骨架；CPU 模拟输入测试。

**尚未完成**：AutoDL 真实 GPU 启动/训练；原 NJU2000 数据格式的现场核验；原始 Swin-B 权重加载；完整 384×384 Swin-B 端到端实测；消融实验及鲁棒性测评；论文指标更新；正式可提交的返修稿。不能宣称模型性能得到验证或论文已经改完。

特别强调：这里是 **原创性待文献检索和实验验证的研究原型**，不是经评审确认的新算法。引用 Wang et al. CVPR 2026 是必要的。本包不会抹除该来源。

## 项目结构

```text
RGBD_revised_offline_v1/
├── models/
│   ├── SwinTransformers.py   # 四个真实 Swin Stage 输出，修复最后一级被忽略
│   └── RGBDRevised.py        # RGB-Swin + Depth-CNN + PG-DUM + DAPF + Decoder
├── legacy/                   # 原始用户提供的主要训练/模型文件备份（不修改）
├── data_rgbd.py              # 同步 RGB/Depth/GT 读入，深度掩码与模拟噪声
├── make_safe_nju_split.py    # 生成不覆盖旧数据的 train/val/test
├── train_rgbd_revised.py     # 根据独立 Val MAE 选 best
├── evaluate_rgbd.py         # 仅针对最终 Test 的指标/噪声评估
├── framework_actual_prototype.svg # 当前原型的网络总图
├── tests/
│   ├── test_cpu_modules.py
│   └── test_swin_stage_cpu.py
├── SOURCE_AUDIT.md
├── PAPER_METHOD_DRAFT.md
├── REBUTTAL_SKELETON.md
└── EXPERIMENTS.csv
```

## 无 GPU 就能运行的检查

上传工程并进入项目根目录后：

```bash
python -m py_compile models/*.py data_rgbd.py make_safe_nju_split.py \
  train_rgbd_revised.py evaluate_rgbd.py
python tests/test_cpu_modules.py
python tests/test_swin_stage_cpu.py
```

`test_swin_stage_cpu.py` 在未安装 timm 时仅临时替换 timm 的少量工具函数，用于 CPU 算子级检查，不代表正式 timm 环境验证。我们已在构建环境测试通过形状、梯度、以及三种路由配置。CPU 随机输入不能用于计算准确率或 FPS。

## 数据集：安全分割（需先确认来源路径）

**不要执行旧 `split_data.py`**，旧脚本会删除现有 `NJU2000_split` 目录。

如果已有 `/root/autodl-tmp/NJU2000_split/test/`，建议保留这批原测试样本不变：

```bash
python make_safe_nju_split.py \
  --source-root /root/autodl-tmp/nju2000 \
  --output-root /root/autodl-tmp/NJU2000_split_revised \
  --existing-test-root /root/autodl-tmp/NJU2000_split/test \
  --val-count 150 --seed 42
```

若没有旧测试划分，则去掉 `--existing-test-root`，默认 500 test / 150 val / 剩余 train（假设源数据有足够样本）。程序要求新输出路径不存在，不删除任何已有目录，并保存 `split_manifest.json`。默认使用软链接，**不得在迁移服务器时忽略软链接指向的原始数据**；若计划删除源目录或跨机器移盘请加 `--copy`（会占额外空间）。

为保证旧论文结果对照公平，必须核实旧 test 文件名清单，而不仅是样本数量；由于旧脚本使用未排序的 `os.listdir`，即使 `random.seed(42)` 也不保证跨环境选出相同的测试样本。

## GPU 可用之后的训练示例（现在暂不执行）

请在准确检查图片深度格式、数据划分、Swin 预训练权重后再执行。建议先只做 `--epochs 1` 的 GPU 冒烟测试，而不是直接开始长训练。

```bash
python train_rgbd_revised.py \
  --split-root /root/autodl-tmp/NJU2000_split_revised \
  --save-dir ./runs/baseline_s42 \
  --variant baseline --seed 42 --epochs 1 \
  --batch-size 4 --swin-pretrained /path/to/swin_base.pth
```

四组主消融分别设置 `--variant baseline / pgdum / dapf / full`。`--variant no_rgb_prior` 为去掉 RGB 边缘先验的消融。为保证可比性，使用相同分割清单、训练超参数、Swin 权重，分别保存日志与 checkpoint。

训练配置默认值取自当前参考脚本附近，但改造的 `uncertainty-weight`, `route-cost-weight` 与深度噪声配置是待调参提案；不可写成已验证最优参数。

### 独立测试与噪声测试（GPU 可用后）

```bash
python evaluate_rgbd.py \
  --checkpoint ./runs/full_s42/best.pth \
  --test-root /root/autodl-tmp/NJU2000_split_revised/test \
  --output ./runs/full_s42/test_noise_metrics.csv
```

默认条件包括干净、深度高斯噪声（0.01/0.03/0.05）、随机缺失（5/10/20/100%）。这只是模拟退化，不构成物理传感器噪声的直接证据。深度完全丢失时检测用 RGB-only 回退；无效掩码取决于数据格式，应先抽查实际深度图。

## 关键技术限制

1. 两个新模块没有准确率实测或公开方法复现结果，不应向审稿人声称提升。
2. DAPF 训练期做软路由、测试期做硬路由，存在分布差异。仅观察到分支跳过代码，不等于已证明 GFLOPs/FPS 降低。
3. 代码的深度分布均值是归一化灰度深度预测，不是物理标定后的米制深度。若数据提供真实测距，应按量纲重新设计加载器与损失。
4. `PAPER_METHOD_DRAFT.md` 与 `REBUTTAL_SKELETON.md` 是草稿，待真实验证后才可改成完成时态。
5. 原始 Swin-B checkpoint、完整数据集、服务器 torch/timm 版本、论文 10 月 8 日自己的 RGB-D 返修稿 **当前没有被验证**。必须保留真实审计边界。
6. 代码包不是对用户原仓库的原位修改；所有变更保存在独立目录，原文件在 `legacy/`。
