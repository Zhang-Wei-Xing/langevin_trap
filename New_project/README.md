# Langevin Trap Simulation

用 Python 模拟一个在三维谐振势能井中的过阻尼布朗粒子。模型对应有限差分形式：

```text
r_i = r_{i-1} - (kappa / gamma) * r_{i-1} * dt + sqrt(2 * D * dt) * normal(0, 1)
```

其中 `kappa` 是三个方向的势阱刚度，`gamma` 是摩擦系数，`D` 是扩散系数。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python langevin_trap.py
```

程序会输出：

- 三维粒子轨迹和理论平衡椭球
- 基于随机游走采样的 `x-z` 平滑概率密度分布，以及由 Fokker-Planck 方程平衡解计算的 `x-y` 理论密度
- 位置自相关函数 ACF
- 平均平方位移 MSD 及理论平台值

默认图片保存到：

```text
outputs/langevin_trap.png
```

## 参数示例

```bash
python langevin_trap.py \
  --kappa 1.0,1.0,0.25 \
  --gamma 1.0 \
  --D 0.05 \
  --dt 0.005 \
  --steps 50000 \
  --initial-position 0.7,-0.25,0.9 \
  --seed 7 \
  --output outputs/langevin_trap.png
```

## 交互界面

```bash
python langevin_trap.py --ui
```

界面左侧可以调整 `kx`、`ky`、`kz`、`gamma`、`D`、`dt`、`steps`、密度图 `bins` 和 burn-in 比例；点击 `Run simulation` 后，右侧会重新计算并展示轨迹、两张密度分布图、ACF 和 MSD。
