"""
误差评估与论文表格生成。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    表格目录,
    黏性系数,
    参考解方法,
    参考解空间点数,
    参考解时间点数,
    初值点数量,
    边界点数量,
    方程残差点数量,
    监督点数量,
    激波监督点比例,
    监督损失权重,
    网络隐藏层结构,
    学习率,
    训练轮数,
    剖面对比时刻,
)


def predict_on_reference_grid(model, x, t):
    """
    在参考解网格上计算 PINN 预测值。

    参数:
        model: 训练后的 PINN 模型
        x: 空间网格
        t: 时间网格

    返回:
        u_pred: PINN 预测解，形状为 [时间点数, 空间点数]
    """
    X, T = np.meshgrid(x, t)
    输入坐标 = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
    u_pred = model(tf.convert_to_tensor(输入坐标, dtype=tf.float32)).numpy()
    return u_pred.reshape(T.shape)


def _relative_l2(error, reference):
    """计算相对 L2 误差，参考范数过小时退化为绝对 L2 误差。"""
    denom = np.linalg.norm(reference.ravel())
    if denom < 1.0e-12:
        return np.linalg.norm(error.ravel())
    return np.linalg.norm(error.ravel()) / denom


def calculate_error_tables(x, t, u_ref, u_pred):
    """
    计算整体误差和不同时刻剖面误差，并保存为中文 CSV 表格。

    参数:
        x: 空间网格
        t: 时间网格
        u_ref: 参考解
        u_pred: PINN 预测解
    """
    del x
    os.makedirs(表格目录, exist_ok=True)

    误差 = u_pred - u_ref
    绝对误差 = np.abs(误差)

    整体误差表 = pd.DataFrame(
        [
            {
                "相对L2误差": _relative_l2(误差, u_ref),
                "均方误差": np.mean(误差**2),
                "平均绝对误差": np.mean(绝对误差),
                "最大绝对误差": np.max(绝对误差),
            }
        ]
    )
    整体误差表.to_csv(os.path.join(表格目录, "表1_整体误差指标.csv"), index=False, encoding="utf-8-sig")

    剖面误差记录 = []
    for 目标时刻 in 剖面对比时刻:
        时间序号 = int(np.argmin(np.abs(t - 目标时刻)))
        当前误差 = u_pred[时间序号] - u_ref[时间序号]
        当前绝对误差 = np.abs(当前误差)
        剖面误差记录.append(
            {
                "目标时刻": 目标时刻,
                "实际取样时刻": t[时间序号],
                "相对L2误差": _relative_l2(当前误差, u_ref[时间序号]),
                "均方误差": np.mean(当前误差**2),
                "平均绝对误差": np.mean(当前绝对误差),
                "最大绝对误差": np.max(当前绝对误差),
            }
        )

    剖面误差表 = pd.DataFrame(剖面误差记录)
    剖面误差表.to_csv(os.path.join(表格目录, "表2_不同时刻剖面误差.csv"), index=False, encoding="utf-8-sig")

    训练参数表 = pd.DataFrame(
        [
            {"参数名称": "参考解方法", "参数取值": 参考解方法},
            {"参数名称": "参考解空间点数", "参数取值": 参考解空间点数},
            {"参数名称": "参考解时间点数", "参数取值": 参考解时间点数},
            {"参数名称": "黏性系数", "参数取值": 黏性系数},
            {"参数名称": "初值点数量", "参数取值": 初值点数量},
            {"参数名称": "边界点数量", "参数取值": 边界点数量},
            {"参数名称": "方程残差点数量", "参数取值": 方程残差点数量},
            {"参数名称": "内部监督点数量", "参数取值": 监督点数量},
            {"参数名称": "激波监督点比例", "参数取值": 激波监督点比例},
            {"参数名称": "监督损失权重", "参数取值": 监督损失权重},
            {"参数名称": "网络隐藏层结构", "参数取值": str(网络隐藏层结构)},
            {"参数名称": "初始学习率", "参数取值": 学习率},
            {"参数名称": "训练轮数", "参数取值": 训练轮数},
        ]
    )
    训练参数表.to_csv(os.path.join(表格目录, "表3_训练参数设置.csv"), index=False, encoding="utf-8-sig")

    return 整体误差表, 剖面误差表, 训练参数表
