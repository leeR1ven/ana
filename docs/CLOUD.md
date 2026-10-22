# 上服务器跑训练：操作手册

## 一、需要你提供的

1. **仓库访问**：`https://github.com/leeR1ven/ana` 目前是**私有**。
   两种做法任选：
   - 给一个只读/可写的 **GitHub token**（我用来 clone），或
   - 把仓库临时改成公开（`gh repo edit leeR1ven/ana --visibility public`）。
2. **机器**：Linux 优先（Windows 也行），**不需要 GPU**，CPU 核越多越好（现在本地 16 核）。
3. **后台运行方式**：`tmux` 或 `nohup` 都行。

## 二、环境（3 条命令）

```bash
git clone https://github.com/leeR1ven/ana.git && cd ana
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # numpy torch
pip install mujoco                     # 跑狗技能题需要
```

自检（必须全绿再开跑）：

```bash
python tests/test_ana.py               # 期望 17/17
python -m agi check --preset xl        # 期望 0 issues
python experiments/foundation_check.py # 期望"地基符合"
```

## 三、开跑（两条主线）

### A. 走路继续往上推（当前最好 0.052 米）

```bash
# 从本地最好模型起步，做局部搜索
python experiments/p29_walk_par.py --n 2000 --jobs $(nproc) --seed-kw walk_best.json
```

### B. 多技能演化（带技能清单、不遗忘）

```bash
python -m agi.dog_evolve --gens 20 --pop 12 --ticks 100 --init walk_best.json
```

输出：`dog_evolve_result.json`（技能清单 + 每代记录）。

## 四、跑完要带回来的三样东西

| 文件 | 内容 |
|---|---|
| `walk_best.json` | 最好的走路模型参数（位移、seed、k_near 等） |
| `dog_evolve_result.json` | 技能清单（第几代学会哪项）和每代成绩 |
| 控制台日志 | 每代"技能 k/5"和"最好位移" |

## 五、注意事项

- **别用更差的结果覆盖好模型**：`p29` 已经改成"只在更好时写入"，`dog_evolve` 的 `--init` 会从最好模型起步；
- **可复现**：同一 `seed` 结果一致；换机器不影响（CPU 推理，已固定单线程）；
- 本地我会继续改题库（加"朝红球走 / 转向 / 起身 / 避障"和更细的难度档），
  **云上跑同一套代码不受影响**——改完我会告诉你拉最新代码再跑。

## 六、当前基线（出发线）

| 项 | 值 |
|---|---|
| 走路最好位移 | **0.052 米**（100 拍） |
| 已获得技能 | `walk1`（>0.03）、`walk2`（>0.05） |
| 未获得 | `walk3`（>0.10）、`turn_to_ball`、`getup` |
| 单模型评估耗时 | 本地约 1.2 秒（12 进程并行时） |
