# 实验代码使用说明

> **配套文档**：方法设计见 [博弈论医学诊断架构_汇报_0502.md](博弈论医学诊断架构_汇报_0502.md)（v3 重新包装版本）；实验规划见 [实验方案_AutoDL.md](实验方案_AutoDL.md)。

## 目录结构

```
code_gjh/
├── common/                       # 共享模块
│   ├── paths.py                  # 环境变量化的路径配置（DATA_ROOT/OUTPUT_ROOT）
│   ├── data.py                   # MLL/PBC 数据加载（支持 LT 变体的指数衰减下采样）
│   ├── models.py                 # Swin-T / ResNet50 骨干 + 分类头 + 竞标网络
│   ├── losses.py                 # 7 种长尾损失（CE/WCE/Focal/LDAM-DRW/LA/GALA）
│   └── train_utils.py            # 早停 / resume / 日志 / 指标计算
├── experiments/                  # 实验脚本（核心）
│   ├── run_main.py               # 主方法（Bidding Game，P1+P2+P3 三阶段）
│   ├── run_baseline.py           # 端到端基线统一脚本
│   ├── run_ride.py               # RIDE (ICLR 2021)
│   ├── run_sade.py               # SADE (NeurIPS 2024)
│   ├── run_gradcam.py            # Grad-CAM 三专家注意力可视化
│   └── summarize_results.py      # 扫描 results/ 生成 LaTeX/Markdown 对比表
├── setup_autodl.sh               # AutoDL 一键环境配置（GPU检查/装包/下权重）
├── run_all.sh                    # 全套实验批量执行（含 Tier 0/1/2 + 成本消融）
└── .gitignore
```

---

## 一、首次在 AutoDL 上启动

### 1. 租 5090 实例（必须）

选择镜像：**PyTorch 2.6+ 且 CUDA 12.8+**（5090 是 Blackwell 架构，老 CUDA 不支持）。
搜索关键词："PyTorch 2.6" / "5090"。

### 2. 上传数据集

```
/root/autodl-tmp/BMC_dataset/
├── MLL/bone_marrow_cell_dataset/
│   └── <class>/<image>.tif
└── PBC_dataset_normal_DIB/
    └── <class>/<image>.jpg
```

推荐：本地 7-zip 压缩 → 阿里云盘 → AutoPanel 转存。

### 3. 拉代码

```bash
cd /root
git clone <your-repo>
cd code_gjh
```

### 4. 一键配置环境

```bash
chmod +x setup_autodl.sh
./setup_autodl.sh
```

脚本会自动：检查 GPU、安装依赖、下载 Swin-T 权重、验证数据集路径。

### 5. 设置环境变量

```bash
export DATA_ROOT=/root/autodl-tmp
export OUTPUT_ROOT=/root/autodl-tmp/code_gjh
```

数据放数据盘（关机不丢），输出也放数据盘（关机后还能下载）。

---

## 二、运行实验

### 后台运行单个实验（推荐）

```bash
# 主方法（约 5h 在 5090 上）
nohup python -m experiments.run_main --dataset mll --backbone swin_t \
    > logs/main.log 2>&1 &

# 查进度
tail -f logs/main.log
```

### 一键全套

```bash
chmod +x run_all.sh
nohup ./run_all.sh > logs/run_all.log 2>&1 &
tail -f logs/run_all.log
```

包含 Tier 0/1/2 全部实验，约 **60-70 小时**，¥180-220。

---

## 三、中断后恢复

**所有脚本都支持 `--resume`**，从最近 checkpoint 自动继续：

```bash
python -m experiments.run_main --dataset mll --backbone swin_t --resume
```

`run_all.sh` 中所有命令默认带 `--resume`，断电/掉线/重启后再跑一次即可。

每个 epoch 结束都会写入 `{run_name}_resume.pth`，包含：模型/优化器/scheduler 状态、history 记录、最优 checkpoint。

---

## 四、跨阶段复用（消融实验加速）

主方法 P1（骨干 ~3h）和 P2（专家 ~1h）训完后，跑各种消融时**不需要重训**：

```bash
# 跑成本机制消融（仅 P3，每组约 30min）
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --expert_costs 0.1,0.1,0.1 --resume

python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --learnable_costs --cost_init 0.5,0.5,0.5 --resume

# alpha 系数消融
python -m experiments.run_main --dataset mll --backbone swin_t \
    --skip_p1 --skip_p2 --p3_alpha 0.0 --resume
```

---

## 五、实验列表速查

### 主方法

| 命令 | 用途 | 时间 |
|------|------|------|
| `run_main.py --dataset mll` | MLL 主实验 | ~5h |
| `run_main.py --dataset pbc --variant lt --imb_factor 100` | PBC-LT 泛化 | ~1.5h |
| `run_main.py --dataset mll --backbone resnet50` | 骨干消融 | ~4h |

### 端到端基线（统一脚本）

| 命令 | 方法 |
|------|------|
| `run_baseline.py --method ce` | Standard CE |
| `run_baseline.py --method weighted_ce` | Weighted CE |
| `run_baseline.py --method focal` | Focal Loss (γ=2) |
| `run_baseline.py --method ldam_drw` | LDAM-DRW (NeurIPS 2019) |
| `run_baseline.py --method logit_adjustment` | Logit Adjustment (ICLR 2021) |
| `run_baseline.py --method gala` | GALA (2024) |

### 多专家基线

| 命令 | 方法 |
|------|------|
| `run_ride.py` | RIDE (ICLR 2021) |
| `run_sade.py` | SADE (NeurIPS 2024) |

### 消融实验（核心）

```bash
# 成本机制消融（论证非对称性的必要性）
--skip_p1 --skip_p2 --expert_costs 0.1,0.1,0.1     # 对称低成本
--skip_p1 --skip_p2 --expert_costs 0.5,0.5,0.5     # 对称中等
--skip_p1 --skip_p2 --expert_costs 1.0,1.0,1.0     # 对称高成本
--skip_p1 --skip_p2 --expert_costs 1.0,0.5,0.1     # 反向
--skip_p1 --skip_p2 --learnable_costs              # 可学习

# Alpha 敏感性
--skip_p1 --skip_p2 --p3_alpha 0.0    # 退化为 CE 门控
--skip_p1 --skip_p2 --p3_alpha 0.1
--skip_p1 --skip_p2 --p3_alpha 1.0
--skip_p1 --skip_p2 --p3_alpha 2.0

# Logit Adjustment tau 敏感性
run_baseline.py --method logit_adjustment --la_tau 0.5
run_baseline.py --method logit_adjustment --la_tau 2.0
```

### 可视化

| 命令 | 用途 | 输入 |
|------|------|------|
| `run_gradcam.py --run_name mll_swin_t_game` | 三专家注意力对比 | 已训好的 main 模型 |
| `summarize_results.py` | 扫描结果生成 LaTeX 表 | 所有 results/ 子目录 |

---

## 六、命令行参数详解

### `run_main.py` 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `mll` | 数据集：`mll` / `pbc` |
| `--variant` | `original` | PBC 变体：`original` / `lt` |
| `--imb_factor` | `100` | LT 不平衡因子 |
| `--backbone` | `swin_t` | `swin_t` / `resnet50` |
| `--p1_max_epochs` | `50` | P1 最大 epoch |
| `--p1_patience` | `10` | P1 早停 patience |
| `--p2_max_epochs` | `30` | P2 最大 epoch |
| `--p2_patience` | `7` | P2 早停 patience |
| `--p3_max_epochs` | `30` | P3 最大 epoch |
| `--p3_patience` | `7` | P3 早停 patience |
| `--p3_alpha` | `0.5` | 博弈损失权重 |
| `--expert_costs` | `0.1,0.5,1.0` | 固定成本（不与 `--learnable_costs` 一起用） |
| `--learnable_costs` | False | 启用可学习成本 |
| `--cost_init` | `0.1,0.5,1.0` | 可学习成本初值 |
| `--la_tau` | `1.0` | Expert C 的 Logit Adjustment τ |
| `--skip_p1` | False | 复用已有 P1 checkpoint |
| `--skip_p2` | False | 复用已有 P2 experts |
| `--resume` | False | 从最近 checkpoint 继续 |
| `--seed` | `42` | 随机种子 |

### 路径环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `DATA_ROOT` | 项目根目录 | `BMC_dataset` 父目录 |
| `OUTPUT_ROOT` | 项目根目录 | `checkpoints/` `results/` `logs/` 父目录 |

---

## 七、输出文件结构

```
$OUTPUT_ROOT/
├── checkpoints/
│   ├── {run_name}_backbone.pth         # P1 训好的骨干（可被 --skip_p1 复用）
│   ├── {run_name}_expert_{0,1,2}.pth   # P2 训好的 3 个专家
│   ├── {run_name}_bidder_{0,1,2}.pth   # P3 训好的 3 个竞标网络
│   ├── {run_name}_costs.pth            # 仅当 --learnable_costs 时存在
│   ├── {run_name}_p1_resume.pth        # P1 断点续训
│   ├── {run_name}_p2_*_resume.pth      # 各专家断点续训
│   └── {run_name}_p3_resume.pth        # P3 断点续训
├── results/
│   ├── {run_name}/
│   │   ├── p1_history.json             # P1 per-epoch 指标
│   │   ├── p1_curves.png               # P1 训练曲线
│   │   ├── p2_ExpertA_history.json     # 各专家 per-epoch 指标
│   │   ├── p3_history.json             # P3 per-epoch 指标（含成本演化）
│   │   ├── summary.json                # 5 个方法（P1/A/B/C/Game）的测试集指标
│   │   ├── test_report.txt             # 完整 classification report
│   │   └── bid_analysis.npz            # 测试集每样本的出价/标签（用于纳什分析图）
│   ├── SUMMARY.md                      # 所有实验汇总（运行 summarize_results 生成）
│   └── SUMMARY.tex                     # LaTeX 表格
└── logs/
    └── {run_name}.log                  # 完整训练日志
```

---

## 八、本地 ↔ AutoDL 协作流程

```
本地（Claude Code 改代码）         AutoDL 5090（跑实验）
──────────────────────              ─────────────────────
git commit -am "xxx"                git pull
git push                            python -m experiments.run_main --resume
                                            ↓
                                    每 epoch 写 history.json + resume.pth
                                            ↓
                                    跑完后用 scp 拉结果回本地
                                    或 AutoPanel 网页下载 results/

scp -P {port} -r \
  root@{host}:/root/autodl-tmp/code_gjh/results \
  ./
```

### 中途修改代码场景

```bash
# 1. AutoDL 实验跑了一半
# 2. 本地发现 bug，改代码 git push
# 3. AutoDL 上：
git pull
# 4. 重新启动（带 --resume，从断点续训）
python -m experiments.run_main --resume
```

注意：如果代码改动影响模型结构，`--resume` 会失败。这种情况只能从头跑。

---

## 九、常见问题

### 跑到一半 OOM 了

减小 batch size：
```bash
python -m experiments.run_main --dataset mll \
    --p1_batch 16 --p2_batch 32 --p3_batch 32
```

### `--resume` 加载失败

通常是因为代码改动导致模型结构不一致。删掉对应的 `_resume.pth` 重新跑：
```bash
rm checkpoints/{run_name}_*_resume.pth
python -m experiments.run_main ...  # 不加 --resume
```

### 想清空所有结果重跑

```bash
rm -rf checkpoints/ results/ logs/
```

### 数据集找不到

检查环境变量：
```bash
echo $DATA_ROOT
ls $DATA_ROOT/BMC_dataset/MLL/bone_marrow_cell_dataset/ | head
```

如果路径不对，重新 export 即可。

---

## 十、复现完整论文结果

跑完 `run_all.sh` 后：

```bash
# 1. 生成对比表
python -m experiments.summarize_results

# 2. 生成 Grad-CAM 图
python -m experiments.run_gradcam --run_name mll_swin_t_game

# 3. 检查输出
ls -lh results/SUMMARY.md results/SUMMARY.tex
ls -lh results/mll_swin_t_game/gradcam/
```

LaTeX 表格直接复制到论文里。

---

## 十一、扩展指南

### 加新基线方法

在 `common/losses.py` 加新损失，然后修改 `experiments/run_baseline.py` 的 `--method` 选项。

### 加新数据集

在 `common/data.py` 的 `build_dataset` 函数加新分支，再在 `common/paths.py` 添加路径常量。

### 改三个专家配置

在 `experiments/run_main.py` 的 `phase2()` 函数里修改 Expert A/B/C 的损失函数（约 5 行）。
