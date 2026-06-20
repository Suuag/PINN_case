"""
PINN 重构结果评估与论文表格生成。
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from config import (
    数据目录,
    表格目录,
    方腔边长,
    顶盖速度,
    运动黏性系数,
    雷诺数,
    每个时间步监督点数量,
    每类边界点数量,
    方程残差点数量,
    网络隐藏层结构,
    学习率,
    训练轮数,
    误差统计时刻,
)


def predict_full_field(model, field_data):
    """
    在完整 OpenFOAM 网格上预测 u、v、p。
    """
    x = field_data["x"]
    y = field_data["y"]
    t = field_data["t"]
    time_count, cell_count = x.shape

    pred = np.zeros((time_count, cell_count, 3), dtype=np.float32)
    for time_id, current_time in enumerate(t):
        xyt = np.column_stack(
            [
                x[time_id],
                y[time_id],
                np.full(cell_count, current_time, dtype=np.float32),
            ]
        ).astype(np.float32)
        pred[time_id] = model(xyt).numpy().astype(np.float32)

    u_pred = pred[:, :, 0]
    v_pred = pred[:, :, 1]
    p_pred = pred[:, :, 2]

    os.makedirs(数据目录, exist_ok=True)
    np.savez(
        os.path.join(数据目录, "PINN重构结果.npz"),
        x=x,
        y=y,
        t=t,
        u_ref=field_data["u"],
        v_ref=field_data["v"],
        p_ref=field_data["p"],
        u_pred=u_pred,
        v_pred=v_pred,
        p_pred=p_pred,
    )
    return {"u": u_pred, "v": v_pred, "p": p_pred}


def _relative_l2(error, reference):
    """计算相对 L2 误差。"""
    denom = np.linalg.norm(reference.ravel())
    if denom < 1.0e-12:
        return np.linalg.norm(error.ravel())
    return np.linalg.norm(error.ravel()) / denom


def _metric_row(name, reference, prediction):
    """生成单个变量的误差统计行。"""
    error = prediction - reference
    abs_error = np.abs(error)
    return {
        "变量名称": name,
        "相对L2误差": _relative_l2(error, reference),
        "均方误差": np.mean(error**2),
        "平均绝对误差": np.mean(abs_error),
        "最大绝对误差": np.max(abs_error),
    }


def _centerline_profile(x, y, values, line_type, point_count=200):
    """
    提取中心线剖面。

    参数:
        line_type: horizontal 表示 y=L/2，vertical 表示 x=L/2
    """
    axis = np.linspace(0.0, 方腔边长, point_count)
    if line_type == "horizontal":
        query = np.column_stack([axis, np.full_like(axis, 方腔边长 / 2.0)])
    else:
        query = np.column_stack([np.full_like(axis, 方腔边长 / 2.0), axis])

    profile = griddata(np.column_stack([x, y]), values, query, method="linear")
    if np.isnan(profile).any():
        nearest = griddata(np.column_stack([x, y]), values, query, method="nearest")
        profile = np.where(np.isnan(profile), nearest, profile)
    return axis, profile


def calculate_error_tables(field_data, pred_data):
    """
    计算整体误差、不同时刻误差、中心线误差和训练参数表。
    """
    os.makedirs(表格目录, exist_ok=True)

    u_ref = field_data["u"]
    v_ref = field_data["v"]
    p_ref = field_data["p"]
    u_pred = pred_data["u"]
    v_pred = pred_data["v"]
    p_pred = pred_data["p"]

    speed_ref = np.sqrt(u_ref**2 + v_ref**2)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)

    overall = pd.DataFrame(
        [
            _metric_row("速度u", u_ref, u_pred),
            _metric_row("速度v", v_ref, v_pred),
            _metric_row("压力p", p_ref, p_pred),
            _metric_row("速度大小", speed_ref, speed_pred),
        ]
    )
    overall.to_csv(os.path.join(表格目录, "表1_整体误差指标.csv"), index=False, encoding="utf-8-sig")

    time_rows = []
    for target_time in 误差统计时刻:
        time_id = int(np.argmin(np.abs(field_data["t"] - target_time)))
        current_speed_ref = speed_ref[time_id]
        current_speed_pred = speed_pred[time_id]
        row = _metric_row("速度大小", current_speed_ref, current_speed_pred)
        row["目标时刻"] = target_time
        row["实际取样时刻"] = field_data["t"][time_id]
        time_rows.append(row)

    time_table = pd.DataFrame(time_rows)
    cols = ["目标时刻", "实际取样时刻", "变量名称", "相对L2误差", "均方误差", "平均绝对误差", "最大绝对误差"]
    time_table = time_table[cols]
    time_table.to_csv(os.path.join(表格目录, "表2_不同时刻误差指标.csv"), index=False, encoding="utf-8-sig")

    final_id = int(np.argmin(np.abs(field_data["t"] - 1.0)))
    x_final = field_data["x"][final_id]
    y_final = field_data["y"][final_id]
    axis_x, u_ref_line = _centerline_profile(x_final, y_final, u_ref[final_id], "horizontal")
    _, u_pred_line = _centerline_profile(x_final, y_final, u_pred[final_id], "horizontal")
    axis_y, v_ref_line = _centerline_profile(x_final, y_final, v_ref[final_id], "vertical")
    _, v_pred_line = _centerline_profile(x_final, y_final, v_pred[final_id], "vertical")

    center_rows = [
        _metric_row("水平中心线u速度", u_ref_line, u_pred_line),
        _metric_row("垂直中心线v速度", v_ref_line, v_pred_line),
    ]
    center_table = pd.DataFrame(center_rows)
    center_table.insert(0, "剖面说明", ["y=L/2", "x=L/2"])
    center_table.to_csv(os.path.join(表格目录, "表3_中心线剖面误差.csv"), index=False, encoding="utf-8-sig")

    param_table = pd.DataFrame(
        [
            {"参数名称": "方腔边长", "参数取值": 方腔边长},
            {"参数名称": "顶盖速度", "参数取值": 顶盖速度},
            {"参数名称": "运动黏性系数", "参数取值": 运动黏性系数},
            {"参数名称": "雷诺数", "参数取值": 雷诺数},
            {"参数名称": "每个时间步监督点数量", "参数取值": 每个时间步监督点数量},
            {"参数名称": "每类边界点数量", "参数取值": 每类边界点数量},
            {"参数名称": "方程残差点数量", "参数取值": 方程残差点数量},
            {"参数名称": "网络隐藏层结构", "参数取值": str(网络隐藏层结构)},
            {"参数名称": "学习率", "参数取值": 学习率},
            {"参数名称": "训练轮数", "参数取值": 训练轮数},
        ]
    )
    param_table.to_csv(os.path.join(表格目录, "表4_训练参数设置.csv"), index=False, encoding="utf-8-sig")

    return overall, time_table, center_table, param_table
