# Burgers 方程 PINN 验证项目

## 1. 项目总体目标

本项目用于验证 Physics-Informed Neural Network（PINN）对一维黏性 Burgers 方程的重构能力。控制方程为：

```text
u_t + u u_x = ν u_xx
```

计算区域为 `x∈[-1,1]`、`t∈[0,1]`，黏性系数为 `ν=0.01/π`，初始条件为 `u(x,0)=-sin(πx)`，边界条件为 `u(-1,t)=u(1,t)=0`。

针对 Burgers 方程在小黏性条件下形成陡梯度结构、PINN 在激波附近误差偏大的问题，本版本采用高分辨率有限体积参考解，并在 PINN 训练中加入内部监督点。

## 2. 已经实现的目标

- 使用有限体积法生成高分辨率 Burgers 方程参考解。
- 对流项使用 Rusanov 数值通量，扩散项使用中心差分。
- 建立 TensorFlow PINN 模型，输入为 `(x,t)`，输出为 `u(x,t)`。
- 使用自动微分计算 `u_t`、`u_x`、`u_xx` 和方程残差。
- 损失函数同时包含初值损失、边界损失、方程残差损失和内部监督损失。
- 内部监督点从参考解中抽取，其中部分点按 `|u_x|` 加权采样，用于加强激波附近学习。
- 输出预测云图、参考解云图、误差云图、典型时刻剖面对比图和损失曲线。
- 输出整体误差、不同时刻误差和训练参数表。

## 3. 项目文件结构

```text
Burgers
├── config.py
├── generate_reference.py
├── pinn_model.py
├── train.py
├── evaluate.py
├── plot_results.py
├── main.py
├── README.md
├── 数据
├── 图片
├── 结果表格
└── 模型
```

## 4. 每个文件的任务和功能

### config.py

集中保存实验参数。

主要内容：

- 方程参数：黏性系数、空间范围、时间范围。
- 参考解参数：有限体积格式、空间点数、时间点数、CFL 数。
- 训练参数：初值点、边界点、残差点、监督点、激波监督点比例。
- 网络参数：隐藏层结构、激活函数、学习率、训练轮数。
- 损失权重：初值、边界、方程残差和监督损失权重。

主要函数：

```python
创建输出目录()
```

### generate_reference.py

生成高分辨率有限体积参考解。

主要函数：

```python
initial_condition(x)
```

定义初始条件 `u(x,0)=-sin(πx)`。

```python
rusanov_flux(u_left, u_right)
```

计算 Burgers 方程对流通量的 Rusanov 数值通量。

```python
finite_volume_rhs(u, dx, nu)
```

构造有限体积半离散右端项。

```python
stable_time_step(u, dx, nu)
```

根据对流和扩散稳定性约束计算显式推进时间步。

```python
generate_reference_solution()
```

生成参考解并保存为 `数据/参考解数据.npz`。

### pinn_model.py

定义 PINN 网络和 Burgers 方程残差。

主要函数：

```python
build_model()
predict_u(model, x, t)
pde_residual(model, x, t, nu=黏性系数)
```

### train.py

生成训练点并训练 PINN。

主要函数：

```python
generate_training_points(x_ref=None, t_ref=None, u_ref=None)
```

生成初值点、边界点、方程残差点和内部监督点。

```python
train_pinn(训练点)
```

执行 Adam 训练，记录总损失、初值损失、边界损失、方程残差损失和监督损失。

### evaluate.py

计算误差并生成论文表格。

主要函数：

```python
predict_on_reference_grid(model, x, t)
calculate_error_tables(x, t, u_ref, u_pred)
```

### plot_results.py

生成论文图。

主要函数：

```python
plot_all_results(x, t, u_ref, u_pred, loss_record, 训练点)
```

输出训练点分布、预测云图、参考解云图、误差云图、剖面对比图和损失曲线。

### main.py

项目总入口。

主要流程：

```text
生成有限体积参考解
生成训练点
训练 PINN
预测全场
计算误差表格
绘制论文图
保存模型
```

## 5. 运行方法

在 PowerShell 中执行：

```powershell
cd E:\本科毕设Part1\Burgers
python main.py
```

运行完成后检查：

- `数据/参考解数据.npz`
- `数据/PINN预测结果.npz`
- `数据/训练损失记录.csv`
- `图片/图1_训练采样点分布.png`
- `图片/图2_PINN预测解云图.png`
- `图片/图3_有限体积参考解云图.png`
- `图片/图4_绝对误差云图.png`
- `图片/图5_典型时刻速度剖面对比.png`
- `图片/图6_损失函数变化曲线.png`
- `结果表格/表1_整体误差指标.csv`
- `结果表格/表2_不同时刻剖面误差.csv`
- `结果表格/表3_训练参数设置.csv`
- `模型/Burgers_PINN模型.keras`

## 6. 主要技术点和技术流程

### 有限体积参考解

Burgers 方程写成守恒形式：

```text
u_t + F(u)_x = νu_xx
F(u)=0.5u^2
```

对流通量使用 Rusanov 格式，扩散项使用中心差分，时间推进采用二阶 Runge-Kutta。该方法比一阶迎风有限差分更适合处理激波附近的守恒结构。

### 监督点学习

传统 PINN 只依赖初边值和方程残差时，容易在陡梯度区域出现位置偏差。本项目从参考解中抽取内部监督点，并按 `|u_x|` 对激波附近区域加权采样，使网络在高误差区域获得更多约束。

### 损失函数

总损失为：

```text
Loss = Loss_ic + Loss_bc + Loss_f + λ_data Loss_data
```

其中 `Loss_data` 为内部监督点误差，默认权重为 `5.0`。

### 论文图表输出

程序输出速度云图、误差云图、典型时刻剖面对比图和损失曲线。误差表格包含整体误差、不同时刻剖面误差和训练参数设置，便于直接整理到论文中。

损失曲线图采用不同线型和不同标记区分各类损失，不只依赖颜色区分，黑白打印时仍可识别总损失、初值损失、边界损失、方程残差损失和监督损失。

论文图件的中文字符采用宋体，英文字符采用 Times New Roman。图标题、图例、轴标题、坐标轴刻度、colorbar 标题和 colorbar 刻度均进行了字号放大，便于论文排版和打印阅读。

损失函数变化曲线的横轴和纵轴主刻度采用科学计数法显示，用于突出训练轮数和损失值的数量级变化。
