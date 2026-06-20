# 2D 顶盖驱动方腔流动 PINN 重构项目

## 1. 项目总体目标

本项目基于 OpenFOAM 计算得到的二维顶盖驱动方腔流动数据，使用 Physics-Informed Neural Network（PINN）重构瞬态速度场和压力场。模型输入为 `(x,y,t)`，输出为 `(u,v,p)`，训练过程同时使用稀疏内部监督数据、边界条件和二维不可压 Navier-Stokes 方程残差。

控制方程为：

```text
u_t + u u_x + v u_y + p_x - ν(u_xx + u_yy) = 0
v_t + u v_x + v v_y + p_y - ν(v_xx + v_yy) = 0
u_x + v_y = 0
```

其中 OpenFOAM 的 `p` 按 icoFoam 中的运动压力处理，密度取 `ρ=1`。

## 2. 已经实现的目标

- 读取 `OF_data/merged_time_steps.mat` 中的 OpenFOAM 内部场数据。
- 读取 `OF_data/all_boundaries.mat` 中的边界点数据。
- 从 OpenFOAM 全场数据中抽取稀疏监督点。
- 基于 `movingWall` 和 `fixedWalls` 构造速度边界约束。
- 在计算域内部随机生成方程残差点。
- 使用 TensorFlow 自动微分计算 NS 方程残差。
- 输出速度云图、压力云图、误差云图、流线图、中心线剖面对比图和损失曲线。
- 输出整体误差、不同时刻误差、中心线剖面误差和训练参数表。

## 3. 项目文件结构

```text
2D_LidDriven
├── OF_data
├── config.py
├── data_loader.py
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

`OF_data` 为已有 OpenFOAM 数据目录，程序不会修改该目录中的原始数据。

## 4. 每个文件的任务和功能

### config.py

功能：集中保存路径、物理参数、采样参数、网络参数和输出设置。

主要函数：

```python
创建输出目录()
设置随机种子()
```

### data_loader.py

功能：读取 MAT 数据，整理 OpenFOAM 内部场，生成训练点。

主要函数：

```python
read_time_values(time_count)
```

从 `time_step_to_vtk_files.csv` 读取时间步；读取失败时使用 `0` 到 `1` 的均匀时间序列。

```python
load_openfoam_field()
```

读取 `center_x`、`center_y`、`pressure`、`velocity_x`、`velocity_y`，并保存 `数据/整理后的OpenFOAM数据.csv`。

```python
load_boundary_points()
```

读取 `movingWall` 和 `fixedWalls` 的二维边界坐标。

```python
calculate_output_statistics(field_data)
```

计算 `u`、`v`、`p` 的均值和标准差，用于网络输出尺度设置。

```python
generate_training_points(field_data)
```

生成监督点、边界点和残差点。

### pinn_model.py

功能：定义 PINN 网络结构和 NS 方程残差。

主要函数：

```python
build_model(output_mean, output_std)
```

构建输入为 `(x,y,t)`、输出为 `(u,v,p)` 的全连接神经网络。

```python
predict_uvp(model, x, y, t)
```

预测速度和压力。

```python
ns_residual(model, points, nu=运动黏性系数)
```

计算二维不可压 Navier-Stokes 方程的两个动量残差和连续性残差。

### train.py

功能：执行小批量 Adam 训练，并记录损失。

主要函数：

```python
train_pinn(训练点, output_mean, output_std)
```

训练 PINN 模型，输出 `数据/训练损失记录.csv` 和 `模型/LidDriven_PINN模型.keras`。

### evaluate.py

功能：在完整 OpenFOAM 网格上预测并计算误差。

主要函数：

```python
predict_full_field(model, field_data)
```

在全部时间步和全部网格点上预测 `u`、`v`、`p`，保存 `数据/PINN重构结果.npz`。

```python
calculate_error_tables(field_data, pred_data)
```

生成整体误差、不同时刻误差、中心线误差和训练参数表。

### plot_results.py

功能：生成论文图。

主要函数：

```python
plot_training_points(训练点)
plot_cloud(x, y, value, title, cbar_label, filename, cmap="viridis")
plot_streamline_comparison(x, y, u_ref, v_ref, u_pred, v_pred)
plot_centerline_profiles(x, y, u_ref, v_ref, u_pred, v_pred)
plot_loss_curve(loss_record)
plot_all_results(field_data, pred_data, loss_record, 训练点)
```

### main.py

功能：项目总入口，按顺序执行数据读取、采样、训练、预测、评估和绘图。

主要函数：

```python
main()
```

## 5. 运行方法

在 PowerShell 中执行：

```powershell
cd E:\本科毕设Part1\2D_LidDriven
python main.py
```

运行完成后会生成：

- `数据/整理后的OpenFOAM数据.csv`
- `数据/训练损失记录.csv`
- `数据/PINN重构结果.npz`
- `图片/图1_训练采样点分布.png`
- `图片/图2_速度大小参考解云图.png`
- `图片/图3_速度大小PINN重构云图.png`
- `图片/图4_速度大小绝对误差云图.png`
- `图片/图5_压力参考解云图.png`
- `图片/图6_压力PINN重构云图.png`
- `图片/图7_流线对比图.png`
- `图片/图8_中心线速度剖面对比.png`
- `图片/图9_损失函数变化曲线.png`
- `结果表格/表1_整体误差指标.csv`
- `结果表格/表2_不同时刻误差指标.csv`
- `结果表格/表3_中心线剖面误差.csv`
- `结果表格/表4_训练参数设置.csv`
- `模型/LidDriven_PINN模型.keras`

## 6. 主要技术点和技术流程

### OpenFOAM 数据整理

`merged_time_steps.mat` 中每个变量的形状为 `[时间步,1,网格数]`。程序将其压缩为 `[时间步,网格数]`，再展开为长表。内部场变量包括空间坐标、速度分量和压力。

### 稀疏监督重构

每个时间步随机抽取部分内部点作为监督数据。模型不是直接拟合全部 OpenFOAM 网格，而是在稀疏数据约束下结合 NS 方程残差进行重构。

### 边界条件约束

`movingWall` 设置为 `u=10, v=0`，`fixedWalls` 设置为 `u=0, v=0`。`frontAndBack` 是二维空边界，不参与速度边界约束。

### 物理残差约束

模型通过 TensorFlow 自动微分计算一阶和二阶导数，并构造动量方程残差和连续性残差。动量残差和连续性残差在损失中按特征尺度归一化，避免量纲差异影响训练。

### 论文图表输出

最终时刻默认取 `t=1.0` 输出速度大小、压力、误差和流线图。同时输出中心线速度剖面对比图，用于分析方腔主涡结构和中心速度分布的重构能力。

论文图件的中文字符采用宋体，英文字符采用 Times New Roman。图标题、图例、轴标题、坐标轴刻度、colorbar 标题和 colorbar 刻度均进行了字号放大，便于论文排版和打印阅读。损失函数曲线同时采用不同线型和标记区分各类损失，不只依赖颜色区分。

### 网格与边界插图

`plot_mesh_boundary.py` 用于生成论文中描述计算域和边界条件的插图。该脚本从 `OF_data/cavity/system/blockMeshDict` 读取网格数量，按照真实的 `128×128` 结构网格绘制计算网格划分图，并根据 OpenFOAM 算例设置绘制边界条件示意图。

运行方法：

```powershell
python plot_mesh_boundary.py
```

输出图片：

- `图片/图10_计算网格划分图.png`
- `图片/图11_边界条件示意图.png`
- `图片/图12_方腔网格与顶盖运动示意图.png`

边界条件示意图中，运动顶盖和固定壁面使用不同线型与标记区分，不只依赖颜色，黑白打印时仍可识别。

`图12_方腔网格与顶盖运动示意图.png` 采用黑色方腔背景和红色顶盖运动箭头，同时显示结构网格、横纵坐标轴和坐标轴标题，不设置图标题，适合作为论文中的计算模型示意图。
