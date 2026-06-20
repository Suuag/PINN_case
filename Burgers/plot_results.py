"""
论文图件绘制函数。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from config import 图片目录, 图片分辨率, 剖面对比时刻


标题字号 = 21
轴标题字号 = 18
刻度字号 = 15
图例字号 = 15
色条标题字号 = 18


def set_chinese_style():
    """设置 Matplotlib 中文字体和基础图形风格。"""
    # 英文优先使用 Times New Roman，中文回退到宋体。
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (7.2, 5.0)
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["font.size"] = 刻度字号
    plt.rcParams["axes.titlesize"] = 标题字号
    plt.rcParams["axes.labelsize"] = 轴标题字号
    plt.rcParams["xtick.labelsize"] = 刻度字号
    plt.rcParams["ytick.labelsize"] = 刻度字号
    plt.rcParams["legend.fontsize"] = 图例字号


def save_current_figure(filename):
    """保存当前图片并关闭画布。"""
    os.makedirs(图片目录, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(图片目录, filename), dpi=图片分辨率, bbox_inches="tight")
    plt.close()


def set_ascii_tick_format(ax, colorbar=None):
    """
    将坐标轴刻度改成普通 ASCII 数字格式。

    部分中文字体不包含数学负号 U+2212，统一格式化后可以避免负号显示成方框。
    """
    数字格式 = FuncFormatter(lambda value, pos: f"{value:g}")
    ax.xaxis.set_major_formatter(数字格式)
    ax.yaxis.set_major_formatter(数字格式)
    ax.tick_params(axis="both", which="major", labelsize=刻度字号)
    if colorbar is not None:
        colorbar.ax.yaxis.set_major_formatter(数字格式)
        colorbar.ax.tick_params(labelsize=刻度字号)
        colorbar.update_ticks()


def set_scientific_tick_format(ax):
    """
    将横纵坐标主刻度统一设置为科学计数法。

    用于损失曲线这类跨数量级变化的图，便于论文中统一表达数量级。
    """
    科学计数格式 = FuncFormatter(lambda value, pos: f"{value:.0e}")
    ax.xaxis.set_major_formatter(科学计数格式)
    ax.yaxis.set_major_formatter(科学计数格式)
    ax.tick_params(axis="both", which="major", labelsize=刻度字号)


def plot_training_points(训练点):
    """绘制初值点、边界点和方程残差点分布。"""
    set_chinese_style()
    plt.figure()
    plt.scatter(训练点["x方程"], 训练点["t方程"], s=4, c="#9aa0a6", alpha=0.45, label="方程残差点")
    if "x监督" in 训练点:
        plt.scatter(训练点["x监督"], 训练点["t监督"], s=5, c="#2ca02c", alpha=0.45, label="内部监督点")
    plt.scatter(训练点["x初值"], 训练点["t初值"], s=16, c="#d62728", label="初值点")
    plt.scatter(训练点["x边界"], 训练点["t边界"], s=10, c="#1f77b4", label="边界点")
    plt.xlabel("空间坐标 x")
    plt.ylabel("时间 t")
    plt.title("PINN 训练采样点分布")
    plt.legend(frameon=False, fontsize=图例字号)
    set_ascii_tick_format(plt.gca())
    save_current_figure("图1_训练采样点分布.png")


def plot_cloud(x, t, u, title, colorbar_label, filename, cmap="viridis"):
    """绘制 x-t 平面上的速度或误差云图。"""
    set_chinese_style()
    plt.figure()
    色图 = plt.pcolormesh(x, t, u, shading="auto", cmap=cmap)
    plt.xlabel("空间坐标 x")
    plt.ylabel("时间 t")
    plt.title(title, fontsize=标题字号)
    cbar = plt.colorbar(色图)
    cbar.set_label(colorbar_label, fontsize=色条标题字号)
    set_ascii_tick_format(plt.gca(), cbar)
    save_current_figure(filename)


def plot_profile_comparison(x, t, u_ref, u_pred):
    """绘制典型时刻速度剖面对比图。"""
    set_chinese_style()
    plt.figure(figsize=(8.0, 5.2))
    颜色 = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    for 序号, 目标时刻 in enumerate(剖面对比时刻):
        时间序号 = int(np.argmin(np.abs(t - 目标时刻)))
        当前颜色 = 颜色[序号 % len(颜色)]
        plt.plot(x, u_ref[时间序号], color=当前颜色, linewidth=1.8, label=f"参考解 t={t[时间序号]:.2f}")
        plt.plot(x, u_pred[时间序号], "--", color=当前颜色, linewidth=1.5, label=f"PINN t={t[时间序号]:.2f}")

    plt.xlabel("空间坐标 x")
    plt.ylabel("速度 u")
    plt.title("典型时刻速度剖面对比")
    plt.legend(frameon=False, ncol=2, fontsize=图例字号)
    set_ascii_tick_format(plt.gca())
    save_current_figure("图5_典型时刻速度剖面对比.png")


def plot_loss_curve(loss_record):
    """绘制训练损失变化曲线。"""
    set_chinese_style()
    plt.figure(figsize=(7.6, 5.0))

    # 科研论文中曲线不能只依赖颜色区分，因此这里同时使用线型和标记。
    曲线样式 = [
        ("总损失", "-", "o", 1.9),
        ("初值损失", "--", "s", 1.5),
        ("边界损失", "-.", "^", 1.5),
        ("方程残差损失", ":", "D", 1.6),
        ("监督损失", (0, (5, 2, 1, 2)), "v", 1.5),
    ]
    标记间隔 = max(len(loss_record) // 12, 1)

    for 列名, 线型, 标记, 线宽 in 曲线样式:
        if 列名 not in loss_record.columns:
            continue
        plt.semilogy(
            loss_record["训练轮数"],
            loss_record[列名],
            label=列名,
            linestyle=线型,
            marker=标记,
            markevery=标记间隔,
            markersize=4.0,
            linewidth=线宽,
            markerfacecolor="white",
            markeredgewidth=0.9,
        )

    plt.xlabel("训练轮数")
    plt.ylabel("损失值")
    plt.title("PINN 损失函数变化曲线", fontsize=标题字号)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend(frameon=False, fontsize=图例字号)
    set_scientific_tick_format(plt.gca())
    save_current_figure("图6_损失函数变化曲线.png")


def plot_all_results(x, t, u_ref, u_pred, loss_record, 训练点):
    """
    统一绘制所有论文图。

    参数:
        x: 空间网格
        t: 时间网格
        u_ref: 参考解
        u_pred: PINN 预测解
        loss_record: 训练损失记录
        训练点: PINN 训练采样点
    """
    绝对误差 = np.abs(u_pred - u_ref)

    plot_training_points(训练点)
    plot_cloud(x, t, u_pred, "PINN 预测解云图", "速度 u", "图2_PINN预测解云图.png", cmap="viridis")
    plot_cloud(x, t, u_ref, "有限体积参考解云图", "速度 u", "图3_有限体积参考解云图.png", cmap="viridis")
    plot_cloud(x, t, 绝对误差, "PINN 预测绝对误差云图", "绝对误差", "图4_绝对误差云图.png", cmap="magma")
    plot_profile_comparison(x, t, u_ref, u_pred)
    plot_loss_curve(loss_record)
