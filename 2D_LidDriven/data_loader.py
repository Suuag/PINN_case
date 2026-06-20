"""
OpenFOAM MAT 数据读取、整理和训练点采样。
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from config import (
    合并场数据文件,
    边界数据文件,
    时间索引文件,
    数据目录,
    方腔边长,
    顶盖速度,
    时间最小值,
    时间最大值,
    每个时间步监督点数量,
    每类边界点数量,
    方程残差点数量,
    随机种子,
)


def read_time_values(time_count):
    """
    读取 OpenFOAM 输出时间。

    参数:
        time_count: merged_time_steps.mat 中的时间步数量

    返回:
        按升序排列的时间数组
    """
    if os.path.exists(时间索引文件):
        try:
            时间表 = pd.read_csv(时间索引文件)
            if "Time_Step" in 时间表.columns:
                时间值 = 时间表["Time_Step"].astype(float).to_numpy()
                if 时间值.size == time_count:
                    return np.sort(时间值)
        except Exception:
            pass

    return np.linspace(时间最小值, 时间最大值, time_count)


def _squeeze_time_cell_array(array):
    """
    将 MAT 中的变量从 [时间步, 1, 网格数] 压缩为 [时间步, 网格数]。
    """
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"无法识别的数据形状: {array.shape}")


def load_openfoam_field():
    """
    读取 OpenFOAM 合并后的内部场数据。

    返回:
        字典，包含 x、y、t、u、v、p 以及长表 DataFrame。
    """
    if not os.path.exists(合并场数据文件):
        raise FileNotFoundError(f"未找到内部场数据文件: {合并场数据文件}")

    mat = loadmat(合并场数据文件)
    需要变量 = ["center_x", "center_y", "pressure", "velocity_x", "velocity_y"]
    for 变量名 in 需要变量:
        if 变量名 not in mat:
            raise KeyError(f"merged_time_steps.mat 中缺少变量: {变量名}")

    x = _squeeze_time_cell_array(mat["center_x"]).astype(np.float32)
    y = _squeeze_time_cell_array(mat["center_y"]).astype(np.float32)
    p = _squeeze_time_cell_array(mat["pressure"]).astype(np.float32)
    u = _squeeze_time_cell_array(mat["velocity_x"]).astype(np.float32)
    v = _squeeze_time_cell_array(mat["velocity_y"]).astype(np.float32)

    时间步数量, 网格数量 = x.shape
    t = read_time_values(时间步数量).astype(np.float32)
    t_grid = np.repeat(t[:, None], 网格数量, axis=1)

    长表 = pd.DataFrame(
        {
            "时间": t_grid.ravel(),
            "空间坐标x": x.ravel(),
            "空间坐标y": y.ravel(),
            "速度u": u.ravel(),
            "速度v": v.ravel(),
            "压力p": p.ravel(),
        }
    )

    os.makedirs(数据目录, exist_ok=True)
    长表.to_csv(os.path.join(数据目录, "整理后的OpenFOAM数据.csv"), index=False, encoding="utf-8-sig")

    return {
        "x": x,
        "y": y,
        "t": t,
        "u": u,
        "v": v,
        "p": p,
        "long_table": 长表,
    }


def load_boundary_points():
    """
    读取边界点坐标和边界类型标签。

    返回:
        字典，包含 movingWall 和 fixedWalls 两类二维边界点。
    """
    if not os.path.exists(边界数据文件):
        raise FileNotFoundError(f"未找到边界数据文件: {边界数据文件}")

    mat = loadmat(边界数据文件)
    coordinates = np.asarray(mat["coordinates"], dtype=np.float32)
    labels = np.char.strip(np.asarray(mat["boundary_labels"]).astype(str).ravel())

    边界点 = {}
    for 标签名 in ["movingWall", "fixedWalls"]:
        mask = labels == 标签名
        coords = coordinates[mask, :2]
        if coords.size == 0:
            raise ValueError(f"边界数据中没有找到 {标签名}")

        # 去除重复点，避免同一个角点被过度采样。
        coords = np.unique(np.round(coords, decimals=8), axis=0).astype(np.float32)
        边界点[标签名] = coords

    return 边界点


def calculate_output_statistics(field_data):
    """
    计算 u、v、p 的均值和标准差，用于网络输出尺度设置。
    """
    outputs = np.column_stack(
        [
            field_data["u"].ravel(),
            field_data["v"].ravel(),
            field_data["p"].ravel(),
        ]
    ).astype(np.float32)

    mean = outputs.mean(axis=0).astype(np.float32)
    std = outputs.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    return mean, std


def _random_choice(rng, total_count, sample_count):
    """生成随机采样序号，样本不足时允许重复采样。"""
    replace = total_count < sample_count
    return rng.choice(total_count, size=sample_count, replace=replace)


def generate_training_points(field_data):
    """
    生成监督点、边界点和方程残差点。

    参数:
        field_data: load_openfoam_field 返回的数据字典

    返回:
        训练点字典
    """
    rng = np.random.default_rng(随机种子)
    t = field_data["t"]
    x = field_data["x"]
    y = field_data["y"]
    u = field_data["u"]
    v = field_data["v"]
    p = field_data["p"]

    监督输入列表 = []
    监督输出列表 = []
    for 时间序号, 当前时刻 in enumerate(t):
        网格数量 = x.shape[1]
        采样序号 = _random_choice(rng, 网格数量, 每个时间步监督点数量)
        当前输入 = np.column_stack(
            [
                x[时间序号, 采样序号],
                y[时间序号, 采样序号],
                np.full(采样序号.size, 当前时刻, dtype=np.float32),
            ]
        )
        当前输出 = np.column_stack(
            [
                u[时间序号, 采样序号],
                v[时间序号, 采样序号],
                p[时间序号, 采样序号],
            ]
        )
        监督输入列表.append(当前输入.astype(np.float32))
        监督输出列表.append(当前输出.astype(np.float32))

    监督输入 = np.vstack(监督输入列表).astype(np.float32)
    监督输出 = np.vstack(监督输出列表).astype(np.float32)

    边界点 = load_boundary_points()
    边界输入列表 = []
    边界输出列表 = []

    for 标签名, 目标速度 in [("movingWall", (顶盖速度, 0.0)), ("fixedWalls", (0.0, 0.0))]:
        coords = 边界点[标签名]
        采样序号 = _random_choice(rng, coords.shape[0], 每类边界点数量)
        当前坐标 = coords[采样序号]
        当前时间 = rng.uniform(时间最小值, 时间最大值, (每类边界点数量, 1)).astype(np.float32)
        当前输入 = np.column_stack([当前坐标[:, 0], 当前坐标[:, 1], 当前时间[:, 0]])
        当前输出 = np.column_stack(
            [
                np.full(每类边界点数量, 目标速度[0], dtype=np.float32),
                np.full(每类边界点数量, 目标速度[1], dtype=np.float32),
            ]
        )
        边界输入列表.append(当前输入.astype(np.float32))
        边界输出列表.append(当前输出.astype(np.float32))

    边界输入 = np.vstack(边界输入列表).astype(np.float32)
    边界速度 = np.vstack(边界输出列表).astype(np.float32)

    残差输入 = np.column_stack(
        [
            rng.uniform(0.0, 方腔边长, 方程残差点数量),
            rng.uniform(0.0, 方腔边长, 方程残差点数量),
            rng.uniform(时间最小值, 时间最大值, 方程残差点数量),
        ]
    ).astype(np.float32)

    return {
        "监督输入": 监督输入,
        "监督输出": 监督输出,
        "边界输入": 边界输入,
        "边界速度": 边界速度,
        "残差输入": 残差输入,
        "边界点": 边界点,
    }
