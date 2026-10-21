# 接口契约

> 换身体、换传感器、换某个区的专用网络时，只要守住这些，架构自检仍然能过。

## 1. 数据流

```
Body.observe() ──► Observation ──► Encoder.encode() ──► np.ndarray(n_in)
      ▲                                                        │
      │                                                  engine.set_input()
      │                                                        ▼
      └──── Body.step(forces) ◄── Brain.forces() ◄──── engine.step() × substeps
```

`Cortex.tick()` 负责把这条链走一遍，并在 `Brain` 上挂 `PlasticRule`。

## 2. `Engine`（agi/engine.py）

| 方法 | 说明 |
|---|---|
| `add_block(n, kind=..., region=..., depth=..., thr=, soft=, inh=, counted=, pos=)` | 加一块神经元，返回连续编号 |
| `connect(dst, src, w, group=0)` | 显式连边；`sgn` 由源是否是抑制细胞自动决定 |
| `link(src, dst, w, group)` | 1:1 专用线（长度必须相同） |
| `set_thr(ids, thr=, soft=, thr_frac=, soft_frac=)` | 门槛/带宽；构建期和运行期都能改 |
| `register_topk(ids, k)` | 这一块每拍精确选 k 个（必须是连续块） |
| `finalize()` | 进入可跑状态（去重、建后端张量） |
| `step()` | 走一小步（dt 由 `cfg.dt` 决定） |
| `drive()` | 当前驱动（标定用） |
| `set_input(v)` / `set_external(ids, v)` | 写感觉输入 / 给其它细胞加外部驱动 |
| `add_edges` / `drop_edges` / `rebuild_backend` | 运行期改边（可塑连接用） |
| `act_np` / `selected` / `thr` / `stats()` | 读 |

不变量：
- 感觉细胞（`input_size` 之前）**不能有入边**；
- 抑制必须用抑制细胞（`inh=True`），**不要写负权重**；
- 同一对 (src,dst) 只允许一条边（`finalize` 会去重）。

## 3. `Brain`（agi/build.py）

| 成员 | 说明 |
|---|---|
| `engine` / `spec` / `geo` | 引擎、规格、几何 |
| `layers: dict[str, LayerInfo]` | 每层：`region, depth, ids, k` |
| `pfc_groups[(region, depth)]` | 每层的前额叶组 |
| `pfc_ids` | 所有前额叶（含运动相关组） |
| `motor["force"][m]` | 肌肉 m 的力度神经元（10 个，门槛单调） |
| `motor["inh"] / ["deinh"]` | 门控 I / D |
| `ids["relay"][(region,depth)]` | 两级中继 `(r1, r2)` |
| `forces()` / `force_counts()` | 0..1 出力 / 原始计数 |
| `pfc_code()` | 当前活跃前额叶编号（int32） |
| `encoder` | 编码器 |

## 4. 换身体：`Body` 协议

只需要两个方法：

```python
class Body:
    def observe(self) -> Observation: ...     # 返回这一拍的感觉（值都在 0..1）
    def step(self, forces: np.ndarray) -> None: ...   # 12 个肌肉的出力档 0..1
```

`VirtualDog`（`agi/world.py`）是参考实现。接 Go2 时：
- `observe()` 从 MuJoCo 读 IMU/关节/接触/相机，转成 `Observation`；
- `step(forces)` 把力度档过一层"脊髓"（PD + 反射 + 力矩映射）再写 `data.ctrl`；
- **物理子步和大脑拍要分开**：物理 500 Hz，大脑 50 Hz。

## 5. 换传感器：`Observation` 字段

| 字段 | 形状 | 范围 | 对应区 |
|---|---|---|---|
| `vision` | `(px, px, 3)` | 0..1 | vision |
| `audio` | `(audio_bands, audio_frames)` | 0..1 | audio |
| `touch` | `(touch_ch,)` | 0..1 | touch |
| `proprio` | `(proprio_ch,)` | 0..1 | proprio |
| `balance` | `(balance_ch,)` | 0..1 | balance |

改形状必须同步改 `Spec`（`px`/`audio_bands`/...），否则 `Encoder` 会报通道数不符。

## 6. 换某个区的专用网络

守住：

| 项 | 要求 |
|---|---|
| 条带 | 在 `Geometry.bands` 里，互不重叠、都在 [0,1] |
| 编码 | 每通道 N 档正反一对，每档恰好亮一个 |
| 区内层 | 每层一个连续块，`pix` 优先排布（`i = 像素×R + 槽位`），R 为整数 |
| 就近 | 只用 (y,z) 的 3×3 像素邻域；门槛/带宽从入边和算 |
| 送额叶 | 每层神经元 1:1 → 中继1 → 中继2 → 该层前额叶组（group 30） |
| 回程 | 从该层前额叶组出发，两级中继回本层；不落 L1，边数 ≤ 前向一半，门槛可达 |
| 增删 | 稳定 ID（只加编号、不移位）；改完跑 `check()` |

## 7. 打开联想学习

```python
from agi import Cortex, PlasticConfig, PlasticRule
rule = PlasticRule(PlasticConfig(enabled=True, eta=0.02, topk_out=8))
ctx = Cortex(brain, plastic=rule)
for obs in stream:                 # 有时间结构的序列
    ctx.tick(obs)
    print(ctx.last_plastic)        # edges / grown / strengthened / pruned
```

- `apply()` 会把边加到 group 40，并**重建后端张量**（`rebuild_backend`）。
  实测 xl 规模（918k 边）重建一次约 **6.8 ms**——每拍都长边会吃掉三分之一预算。
  正式训练时建议：边先在 numpy 里攒着，每 5–10 拍重建一次（或在 P1 把后端改成
  容量可增长的缓冲，只追加不重建）。v0 不涉及，因为默认关闭。
- v0 自检要求 group 40 = 0；打开学习后跑 `check(brain, expect_no_plastic=False)`。

## 8. 稳定性红线（改任何东西都要守）

1. 每层每拍被选中数 == top-k（`engine.selected` 可查）；
2. 每条感觉通道每档恰好一个亮（`encoder` 可查）；
3. 前向边方向不能反（层 → 中继 → 额叶）；
4. 回程 ≤ 前向一半、不落 L1；
5. 力度门槛单调；
6. 单拍 ≤ 20 ms（`bench` 可查）；
7. 结构改动后 `check()` 必须 0 issues。
