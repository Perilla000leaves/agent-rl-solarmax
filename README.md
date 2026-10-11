# Agent RL SolarMax

一个轻量的 SolarMax 风格策略游戏和第一版强化学习环境骨架。

## 运行游戏

```powershell
python -m http.server 8000
```

然后打开 http://localhost:8000/ 。从蓝色星球拖到目标星球释放，即可派出一半舰队。

画面使用原创 Canvas 渲染实现 SolarMax 风格的深空策略观感：发光星球、兵流粒子、拖拽箭头、发射闪光和占领冲击波。没有使用或复制官方素材。

## 网页 DQN 演示

网页右上角有 `DQN 演示` 按钮。点击后会启动一个观战 episode，由未训练的小型 DQN demo policy 自动选择合法动作并驱动 `SolarMaxEnv.step()`。右侧面板会显示：

- 当前 step
- done / winner
- 最近一步 reward、累计 reward 和 reward breakdown
- epsilon / policy 类型，以及该步是 explore 还是 greedy
- action index、source、target、sendRatio、是否合法
- observation/state 摘要：星球数、总兵力、总产能、舰队数、合法动作数
- 最近几步 step log

画布会短暂高亮 agent 最近选择的源星球和目标星球。再次点击 `停止演示` 可回到手动游玩；演示中也可以暂停或重新开始一局。

启动演示时它会尝试加载 `checkpoints/dqn-latest.json`（当前最强的 BC + DQfD 微调模型，见下文战绩），面板会显示 `loaded checkpoint DQN`；加载失败时会退回未训练的 demo baseline 并在面板显示原因。

## 运行测试

当前环境没有 `npm`，可以直接使用 Node 内置测试运行器：

```powershell
node --test
```

## 随机策略 baseline

```powershell
node scripts/random-baseline.js 25
```

第一个参数是 episode 数，第二个可选参数是每局最大 step 数：

```powershell
node scripts/random-baseline.js 100 600
```

## DQN baseline

当前 DQN baseline 不依赖 TensorFlow.js 或 PyTorch，而是一个纯 JavaScript 小型 MLP Q 网络，便于在没有 `npm` 的环境里端到端跑通。它包含经验回放、epsilon-greedy、target network，并且在选动作和计算 next-state target 时都会使用 `actionMask` 屏蔽非法动作。

```powershell
node scripts/train-dqn.js 50 300 normal aggressive 1
```

参数依次是：训练 episode 数、每局最大 step 数、难度、对手策略、基础 seed、checkpoint 输出路径、可选 resume checkpoint、epsilonStart、epsilonMin、epsilonDecay、learningRate、可选 gamma（默认 0.96；想学习更长视野的策略时可提高到 0.99）。难度支持 `easy`、`normal`、`hard`；对手策略支持 `aggressive`、`expansion`、`defensive`、`random`。脚本会逐局输出 reward、recentReward、epsilon、loss、当前对手、胜负统计，并默认保存到 `checkpoints/dqn-latest.json`。

```powershell
node scripts/train-dqn.js 50 300 normal random 1 checkpoints/dqn-latest.json
```

课程训练示例：

```powershell
node scripts/train-dqn.js 40 200 easy expansion 31 checkpoints/dqn-curriculum-stage1.json none 1 0.12 0.975 0.0015
node scripts/train-dqn.js 40 220 normal random 71 checkpoints/dqn-curriculum.json checkpoints/dqn-curriculum-stage1.json 0.45 0.10 0.975 0.0006
```

评估已保存 checkpoint：

```powershell
node scripts/evaluate-dqn.js checkpoints/dqn-latest.json 25 300 normal random 500
```

网页演示启动时会自动尝试加载 `./checkpoints/dqn-latest.json`。如果加载成功，面板会显示 `loaded checkpoint DQN`，epsilon 会降到接近 greedy；如果文件不存在或加载失败，会显示 fallback 原因并继续使用未充分训练的 demo baseline。

## 行为克隆热启动与 DQfD 微调

纯 DQN 从零训练在这个环境里样本效率不足（3000 局后贪心胜率仍是 0），因此项目的最优路线改为三段式：

1. **启发式老师**：`src/heuristic-policy.js` 提供共享规则策略（防守优先、攻击最弱目标、抢占中立、兜底支援），同时作为所有实验的对照基线
2. **行为克隆（BC）预训练**：`scripts/pretrain-bc.js` 用启发式生成示范数据，通过带动作掩码的 softmax 交叉熵把 expert 动作推成最高分
3. **DQfD 式微调**：`scripts/finetune-dqn.js` 从 BC checkpoint 热启动，重新注入示范样本（`trainStep` 中留一半批量做模仿更新以抵抗遗忘），支持 `easy,normal` 多难度按局交替训练，并用 held-out 种子周期性贪心评估

```powershell
node scripts/pretrain-bc.js 150 easy,normal random 101 checkpoints/dqn-bc.json 45 0.004
node scripts/finetune-dqn.js 1200 easy,normal random 8001 checkpoints/dqn-bc.json checkpoints/dqn-bc-tuned.json 40 0.0002
node scripts/heuristic-baseline.js 60 normal random 7000
```

- `pretrain-bc.js` 参数：每难度 episode 数、难度列表、对手策略、基础 seed、输出路径、训练 epoch 数、BC 学习率
- `finetune-dqn.js` 参数：微调 episode 数、难度列表（逗号分隔，按局交替）、对手策略、基础 seed、BC 输入路径、输出路径、每难度示范注入局数、微调学习率
- `heuristic-baseline.js` 参数：episode 数、难度、对手策略、基础 seed

### 三方对比（3 组 held-out 种子 × 60 局/难度，vs random）

| 策略 | easy | normal | 合计 |
|---|---|---|---|
| 启发式（老师） | 152/180 (84.4%) | 113/180 (62.8%) | 265/360 (73.6%) |
| BC 模型 | 150/180 (83.3%) | 67/180 (37.2%) | 217/360 (60.3%) |
| **BC + DQfD 微调** | **173/180 (96.1%)** | **118/180 (65.6%)** | **291/360 (80.8%)** |

微调模型在两组难度上都稳定超过示范，已同步为 `checkpoints/dqn-latest.json`，网页演示会直接加载。

## 环境随机化和难度

为了避免 agent 只记住单一地图和固定规则对手，环境现在在保持固定 observation/action shape 的前提下做了小幅随机化：

- 固定 8 个星球和 192 个离散动作，但不同 seed 会改变部分星球位置、中立星球兵力和 growth。
- 地图随机化是受控的：双方 home 仍在左右两侧，中立星球仍分布在中场附近，避免生成明显不可玩的开局。
- 对手策略包括 `aggressive`、`expansion`、`defensive`，也可以用 `random` 每局抽取策略。
- 难度会调整敌方决策频率、派兵比例和开局微优势。`hard` 会默认启用随机对手。

示例：

```powershell
node scripts/train-dqn.js 20 200 easy expansion 10
node scripts/train-dqn.js 20 200 hard random 10
```

## RL 环境接口

`src/rl-env.js` 导出 `SolarMaxEnv`：

- `reset(seed?) -> observation`
- `step(action) -> { observation, reward, done, info }`
- `observe() -> observation`
- `legalActions() -> actionMask`

动作空间固定为 `8 * 8 * 3 = 192`，编码为 `(sourcePlanet, targetPlanet, sendRatio)`，其中 `sendRatio` 为 `[0.25, 0.5, 0.75]`。非法动作会在 `actionMask` 中标为 `0`，传入非法动作会得到惩罚但环境仍会向前推进。

观测包含：

- `planets`：8 个星球的固定长度特征，包含归属、位置、半径、兵力、增长、来袭舰队。
- `fleets`：最多 32 支舰队的 padding 后固定长度特征。
- `global`：步数、星球占比、产能、总兵力、胜负标记。
- `actionMask`：长度 192 的合法动作掩码。
