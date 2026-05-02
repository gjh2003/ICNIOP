# ICONIP 论文实验方案

> **目标会议**：ICONIP 2026（11 月 23-27 日）
> **算力平台**：AutoDL RTX 5090
> **预算**：约 ¥180-250，60-70 GPU 小时
> **数据集**：MLL（主，21 类，171k 张，3678:1）+ PBC-LT（次，8 类，~7k 张，100:1）

---

## 一、已有可用实验数据

### 1.1 已训练模型（checkpoints/）

| 模型 | 状态 | 用途 |
|------|------|------|
| `game_v2_backbone_swin_t.pth` 等 | ✅ Swin-T，5+10+15 epoch | 旧版主实验，可作为论文里 "Ours" 的初版数据 |
| `improved_*_swin_t.pth` | ✅ Swin-T v1 框架 | v1 三阶段实验对比 |
| `improved_*_resnet50.pth` | ✅ ResNet50 v1 框架 | 骨干消融的 ResNet50 数据 |
| `finetune_{ce,wce,focal,nash}.pth` | ✅ ResNet50 端到端 | 早期基线对比，已不太用得上 |

### 1.2 已有结果文件（results/）

| 文件 | 内容 | 是否论文可用 |
|------|------|------------|
| `game_v2_results_swin_t.txt` | v2 主实验完整结果 | ✅ 可作初稿数据，AutoDL 重跑后替换 |
| `ablation_routing.txt` | 路由消融（Simple Avg / CE Gating / Bidding Game） | ✅ **关键消融，证明博弈机制有效** |
| `improved_results_swin_t.txt` | v1 三阶段路由 | ⚠️ 可作论文 "preliminary version" 引用 |
| `finetune_comparison.txt` | ResNet50 4 损失对比 | ⚠️ 部分有用（CE/Weighted CE/Focal） |

### 1.3 现有数据的局限

1. **Phase 1 只跑了 5 epoch** —— 骨干网络远未收敛
2. **缺关键长尾基线**：LDAM-DRW、Logit Adjustment、RIDE、SADE、GALA 等
3. **没有多 seed 实验** —— 审稿人会要求标准差
4. **没有出价成本 / α 系数的消融**
5. **PBC 数据集还没跑**
6. **没有 Grad-CAM 可视化**

---

## 二、AutoDL 上待做的实验清单

### 2.1 训练策略（统一应用）

| Phase | 最大 epoch | 早停 patience | 监控指标 |
|-------|-----------|--------------|---------|
| Phase 1（骨干网络） | 50 | 10 | val_macro_f1 |
| Phase 2（专家头） | 30 | 7 | val_macro_f1 |
| Phase 3（竞标博弈） | 30 | 7 | val_macro_f1 |

预计实际训练 epoch：P1 约 25-35，P2 约 12-18，P3 约 10-15。

### 2.2 实验 A：MLL 主实验（约 5h）

**目标**：用充分训练替代之前 5 epoch 的初版结果。

| 项目 | 内容 |
|------|------|
| 骨干 | Swin-Tiny |
| 数据集 | MLL（70/15/15 分层划分，seed=42） |
| 流程 | Phase 1 → Phase 2（A/B/C）→ Phase 3 竞标博弈 |
| 输出指标 | Acc, Macro F1, Wtd F1, Head F1, Tail F1, 21 类逐类 F1 |

### 2.3 实验 B：MLL 长尾基线对比（约 12-15h）

**对所有基线统一使用 Swin-Tiny 骨干 + 早停 + 完整训练**，确保公平。

| # | 基线 | 年份 | 实现要点 | 预估时间 |
|---|------|------|---------|---------|
| B1 | Standard CE | - | 标准训练 | 1.5h |
| B2 | Weighted CE | - | 损失权重 = 1/π_k | 1.5h |
| B3 | Focal Loss | ICCV 2017 | γ=2.0, α=1/π_k | 1.5h |
| B4 | LDAM-DRW | NeurIPS 2019 | LDAM 损失 + DRW 后期重加权 | 2h |
| B5 | Logit Adjustment | ICLR 2021 | logit + τ·log(π_k) | 1.5h |
| B6 | RIDE | ICLR 2021 | 3 专家 + 多样性损失 + gating | 3h |
| B7 | GALA | 2024 | 梯度感知的 Logit Adjustment | 2h |
| B8 | **SADE** | NeurIPS 2024 | 3 个不同分布训练的专家 + test-agnostic 路由 | 4h |

**可选**（如果时间充裕，约 5h）：

| # | 基线 | 年份 | 备注 |
|---|------|------|------|
| B9 | PaCo | ECCV 2022 | 参数化对比学习，需要 contrastive 预训练阶段 |
| B10 | GCL | CVPR 2022 | 高斯云 logit 调整 |

### 2.4 实验 C：PBC-LT 验证（约 3h）

**目标**：用第二个数据集证明方法泛化性，**只跑核心基线**。

| # | 方法 | 时间 |
|---|------|------|
| C1 | Standard CE | 0.5h |
| C2 | RIDE | 0.7h |
| C3 | SADE | 1h |
| C4 | **Ours** | 0.5h |

### 2.5 实验 D：消融实验（约 12h）

D1. **路由策略消融**（已有数据，AutoDL 重跑确认）

| 路由 | 备注 |
|------|------|
| Simple Average | 不学习，等权 |
| CE Gating | 学习路由，无 game loss |
| Bidding Game | 完整方法 |

D2. **出价成本 c_i 敏感性**（4 组配置）

| 配置 | $(c_1, c_2, c_3)$ | 含义 |
|------|------------------|------|
| 当前 | (0.1, 0.5, 1.0) | 非对称（默认） |
| 对称低 | (0.1, 0.1, 0.1) | 全部低成本 |
| 对称高 | (1.0, 1.0, 1.0) | 全部高成本 |
| 反向 | (1.0, 0.5, 0.1) | 头部专家成本最高 |

D3. **博弈损失权重 α 敏感性**

| α 值 | 0.0 | 0.1 | 0.5 | 1.0 | 2.0 |
|------|-----|-----|-----|-----|-----|
| 含义 | 无 game loss（=CE Gating） | 弱 | 默认 | 中 | 强 |

α=0.0 等价于 CE Gating，覆盖前面消融的同样数据点。

D4. **专家组合消融**（证明三个专家缺一不可）

| 配置 | 说明 |
|------|------|
| A only | 单专家基线 |
| A + B | 去掉极端尾部专家 |
| A + C | 去掉尾部专家 |
| B + C | 去掉头部专家 |
| A + B + C | 完整方法 |

### 2.6 实验 E：多 seed 鲁棒性（约 10h）

主实验（实验 A）和最关键基线（B6 RIDE、B8 SADE）用 3 个不同 seed 重跑：

| Seed | 主实验 | RIDE | SADE |
|------|--------|------|------|
| 42 | ✅ 已有 | ✅ B6 | ✅ B8 |
| 0 | 待跑 | 待跑 | 待跑 |
| 123 | 待跑 | 待跑 | 待跑 |

报告均值 ± 标准差。

### 2.7 实验 F：可视化（约 1h，仅推理）

| 图 | 用途 | 耗时 |
|---|------|------|
| F1: Grad-CAM 三专家对比 | 展示专家分工的视觉证据 | 30min |
| F2: 出价 vs 类别频率散点图 | 验证 Nash 均衡理论 $b_i^* \propto 1/c_i$ | 5min |
| F3: 专家分配权重直方图 | 展示路由决策分布 | 5min |
| F4: 混淆矩阵（21×21） | 错分模式分析 | 5min |
| F5: t-SNE 特征空间 | 三专家学到的特征结构 | 20min |

---

## 三、总时间和费用预算

| 实验组 | 时间 | 费用（¥3/h 估） |
|--------|------|----------------|
| A: MLL 主实验 | 5h | ¥15 |
| B: MLL 基线对比（8 个） | 17h | ¥51 |
| B optional: PaCo + GCL | 5h | ¥15 |
| C: PBC-LT 4 个方法 | 3h | ¥9 |
| D: 4 个消融实验 | 12h | ¥36 |
| E: 多 seed × 3 个核心实验 | 10h | ¥30 |
| F: 可视化（推理） | 1h | ¥3 |
| Debug + 补跑缓冲 | 10h | ¥30 |
| **合计** | **约 63h** | **约 ¥189** |
| 加 PaCo + GCL | 68h | ¥204 |

---

## 四、执行优先级（如果时间不够时砍）

### Tier 0（必做，论文骨架）

- A: MLL 主实验
- B1-B6: CE/WCE/Focal/LDAM/LA/RIDE 6 个基线
- B8: SADE（直接竞争对手，不能少）
- C: PBC-LT 4 方法
- D1: 路由消融（已有，确认即可）
- F1: Grad-CAM 可视化

**Tier 0 小计：约 30h，¥90**

### Tier 1（强烈推荐）

- B7: GALA（2024 最新基线）
- D2: 出价成本消融
- D4: 专家组合消融
- F2-F4: 其他可视化

**Tier 1 小计：约 15h，¥45**

### Tier 2（锦上添花）

- B9-B10: PaCo, GCL
- D3: α 敏感性
- E: 多 seed
- F5: t-SNE

**Tier 2 小计：约 18h，¥54**

---

## 五、代码改造任务（AutoDL 跑之前要先做）

### 5.1 必做改造

1. **数据/输出路径环境变量化**
   - `DATA_ROOT = os.environ.get("DATA_ROOT", os.path.dirname(__file__))`
   - 类似处理 `RESULTS_DIR`, `CKPT_DIR`

2. **早停策略加入所有 phase**
   - 当前 Phase 1 没有早停，需要加上

3. **断点续训**
   - 每个 epoch 保存 `{tag}_resume.pth`，包含 epoch 编号、模型状态、optimizer 状态、history
   - 启动时检查并加载

4. **完整日志保存**
   - 每个 phase 的 per-epoch 指标保存为 JSON
   - 训练曲线图自动生成
   - 测试集逐类 classification report 保存

### 5.2 新增代码

5. **8 个长尾基线方法实现**
   - LDAM-DRW、Logit Adjustment、RIDE、GALA、SADE 等
   - 统一接口，方便切换

6. **PBC 适配脚本**（已有 `run_pbc.py` 雏形，待完善）

7. **Grad-CAM 可视化脚本**

8. **结果汇总脚本**：扫描所有 result 文件，生成 LaTeX 表格代码

---

## 六、本地与 AutoDL 协作流程

```
本地（你的电脑）              AutoDL 5090
─────────────────            ──────────
Claude Code 改代码            git pull
  │                            │
  ├─ git push                  ├─ python run_xxx.py
  │                            │   (nohup 后台跑)
  │   ◄─────  scp 拉日志/结果  │
  └─ 查日志 + 改进              │
```

### Git 仓库内容

只 push 代码，**不 push** 数据集和大文件 checkpoint：

```
.gitignore 内容：
BMC_dataset/
checkpoints/
saved_features/
results/*.png
*.pth
*.npy
__pycache__/
```

---

## 七、推进顺序建议

### 第 1 天：代码改造（本地完成）

1. 路径环境变量化
2. 加早停 + 断点续训 + 日志
3. 实现 LDAM-DRW（最简单）和 Logit Adjustment（最简单）
4. 本地验证脚本能正常启动

### 第 2 天：AutoDL 上手 + Tier 0 实验

1. 租 5090，传数据，配环境
2. 跑实验 A（MLL 主实验）—— 约 5h
3. 同时本地写 RIDE、SADE 代码

### 第 3-4 天：Tier 0 基线 + PBC

1. 跑 B1-B8 基线 —— 约 17h
2. 跑 C 实验 —— 约 3h
3. 跑 F1 Grad-CAM —— 约 30min

### 第 5-6 天：消融 + 多 seed

1. 跑 D 系列消融
2. 跑 E 多 seed

### 第 7 天：补跑 + 整理

1. 处理 debug 中失败的实验
2. 用结果汇总脚本生成 LaTeX 表格

---

## 八、关键决策点（需要你确认）

| 决策 | 选项 | 我的建议 |
|------|------|---------|
| Tier 范围 | 0 / 0+1 / 全做 | **0+1，预算 ¥135** |
| PaCo / GCL 是否加 | 加 / 不加 | 不加（实现复杂，边际收益低） |
| 多 seed 数 | 1 / 3 / 5 | 3 个，对核心实验做即可 |
| Phase 1 max epoch | 30 / 50 / 80 | 50（早停会自动停） |
| ResNet50 消融是否做 | 是 / 否 | 不做（会扩大表格，分散重点） |

---

## 九、最终交付物（论文素材）

### 论文表格

- **Table 1**：MLL 主对比（10 行：CE/WCE/Focal/LDAM/LA/RIDE/GALA/SADE/Ours + Bidding Game vs CE Gating）
- **Table 2**：PBC-LT 泛化对比（4 行）
- **Table 3**：路由策略消融（3 行）
- **Table 4**：专家组合消融（5 行）
- **Table 5**：出价成本敏感性（4 行）

### 论文图

- **Fig 1**：方法总览图（架构示意，本地手画）
- **Fig 2**：Grad-CAM 三专家注意力对比
- **Fig 3**：纳什均衡验证散点图
- **Fig 4**：尾部类逐类 F1 改善对比柱状图
- **Fig 5**：训练曲线（loss/acc 收敛过程，可选）

### 附录材料

- 完整 21 类 / 8 类 classification report
- 多 seed 标准差表
- 超参数配置详细列表
