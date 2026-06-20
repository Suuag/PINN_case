"""
论文图件绘制。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata

from config import 图片目录, 图片分辨率, 方腔边长, 最终绘图时刻, 插值网格数量


标题字号 = 21
轴标题字号 = 18
刻度字号 = 15
图例字号 = 15
色条标题字号 = 18


def set_chinese_style():
    """设置中文字体和科研图基础格式。"""
    # 英文优先使用 Times New Roman，中文回退使用宋体。
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.figsize"] = (7.2, 5.4)
    plt.rcParams["font.size"] = 刻度字号
    plt.rcParams["axes.titlesize"] = 标题字号
    plt.rcParams["axes.labelsize"] = 轴标题字号
    plt.rcParams["xtick.labelsize"] = 刻度字号
    plt.rcParams["ytick.labelsize"] = 刻度字号
    plt.rcParams["legend.fontsize"] = 图例字号


def set_ascii_tick_format(ax, colorbar=None):
    """避免负号字体缺失导致刻度显示异常。"""
    formatter = FuncFormatter(lambda value, pos: f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", which="major", labelsize=刻度字号)
    if colorbar is not None:
        colorbar.ax.yaxis.set_major_formatter(formatter)
        colorbar.ax.tick_params(labelsize=刻度字号)
        colorbar.update_ticks()


def save_current_figure(filename):
    """保存当前图片。"""
    os.makedirs(图片目录, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(图片目录, filename), dpi=图片分辨率, bbox_inches="tight")
    plt.close()


def interpolate_to_grid(x, y, value):
    """
    将非结构化或展平网格数据插值到规则网格，便于绘制云图和流线图。
    """
    grid_x = np.linspace(0.0, 方腔边长, 插值网格数量)
    grid_y = np.linspace(0.0, 方腔边长, 插值网格数量)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = griddata(np.column_stack([x, y]), value, (X, Y), method="linear")
    if np.isnan(Z).any():
        Z_nearest = griddata(np.column_stack([x, y]), value, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), Z_nearest, Z)
    return X, Y, Z


def plot_training_points(训练点):
    """绘制训练采样点分布。"""
    set_chinese_style()
    plt.figure()
    sup = 训练点["监督输入"]
    bc = 训练点["边界输入"]
    res = 训练点["残差输入"]

    plt.scatter(res[:, 0], res[:, 1], s=3, c="#9aa0a6", alpha=0.35, label="方程残差点")
    plt.scatter(sup[:, 0], sup[:, 1], s=3, c="#1f77b4", alpha=0.35, label="内部监督点")
    plt.scatter(bc[:, 0], bc[:, 1], s=8, c="#d62728", alpha=0.75, label="边界约束点")
    plt.xlabel("空间坐标 x/m")
    plt.ylabel("空间坐标 y/m")
    plt.title("PINN 训练采样点分布")
    plt.axis("equal")
    plt.legend(frameon=False, fontsize=图例字号)
    set_ascii_tick_format(plt.gca())
    save_current_figure("图1_训练采样点分布.png")


def plot_cloud(x, y, value, title, cbar_label, filename, cmap="viridis"):
    """绘制二维云图。"""
    set_chinese_style()
    X, Y, Z = interpolate_to_grid(x, y, value)
    plt.figure()
    im = plt.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
    plt.xlabel("空间坐标 x/m")
    plt.ylabel("空间坐标 y/m")
    plt.title(title, fontsize=标题字号)
    plt.axis("equal")
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label, fontsize=色条标题字号)
    set_ascii_tick_format(plt.gca(), cbar)
    save_current_figure(filename)


def plot_streamline_comparison(x, y, u_ref, v_ref, u_pred, v_pred):
    """绘制参考解和 PINN 重构流线对比图。"""
    set_chinese_style()
    X, Y, U_ref = interpolate_to_grid(x, y, u_ref)
    _, _, V_ref = interpolate_to_grid(x, y, v_ref)
    _, _, U_pred = interpolate_to_grid(x, y, u_pred)
    _, _, V_pred = interpolate_to_grid(x, y, v_pred)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    for ax, U, V, title in [
        (axes[0], U_ref, V_ref, "OpenFOAM 参考流线"),
        (axes[1], U_pred, V_pred, "PINN 重构流线"),
    ]:
        speed = np.sqrt(U**2 + V**2)
        ax.streamplot(X, Y, U, V, color=speed, cmap="viridis", density=1.5, linewidth=0.8)
        ax.set_xlabel("空间坐标 x/m")
        ax.set_ylabel("空间坐标 y/m")
        ax.set_title(title, fontsize=标题字号)
        ax.set_aspect("equal")
        set_ascii_tick_format(ax)
    plt.savefig(os.path.join(图片目录, "图7_流线对比图.png"), dpi=图片分辨率, bbox_inches="tight")
    plt.close(fig)


def _centerline_profile(x, y, value, line_type, point_count=200):
    """提取中心线剖面数据。"""
    axis = np.linspace(0.0, 方腔边长, point_count)
    if line_type == "horizontal":
        query = np.column_stack([axis, np.full_like(axis, 方腔边长 / 2.0)])
    else:
        query = np.column_stack([np.full_like(axis, 方腔边长 / 2.0), axis])
    profile = griddata(np.column_stack([x, y]), value, query, method="linear")
    if np.isnan(profile).any():
        nearest = griddata(np.column_stack([x, y]), value, query, method="nearest")
        profile = np.where(np.isnan(profile), nearest, profile)
    return axis, profile


def plot_centerline_profiles(x, y, u_ref, v_ref, u_pred, v_pred):
    """绘制中心线速度剖面对比。"""
    set_chinese_style()
    axis_x, u_ref_line = _centerline_profile(x, y, u_ref, "horizontal")
    _, u_pred_line = _centerline_profile(x, y, u_pred, "horizontal")
    axis_y, v_ref_line = _centerline_profile(x, y, v_ref, "vertical")
    _, v_pred_line = _centerline_profile(x, y, v_pred, "vertical")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
    axes[0].plot(axis_x, u_ref_line, label="OpenFOAM", linewidth=1.8)
    axes[0].plot(axis_x, u_pred_line, "--", label="PINN", linewidth=1.6)
    axes[0].set_xlabel("空间坐标 x/m")
    axes[0].set_ylabel("u 速度/(m/s)")
    axes[0].set_title("水平中心线 u 速度剖面")
    axes[0].legend(frameon=False, fontsize=图例字号)

    axes[1].plot(axis_y, v_ref_line, label="OpenFOAM", linewidth=1.8)
    axes[1].plot(axis_y, v_pred_line, "--", label="PINN", linewidth=1.6)
    axes[1].set_xlabel("空间坐标 y/m")
    axes[1].set_ylabel("v 速度/(m/s)")
    axes[1].set_title("垂直中心线 v 速度剖面")
    axes[1].legend(frameon=False, fontsize=图例字号)

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        set_ascii_tick_format(ax)

    plt.savefig(os.path.join(图片目录, "图8_中心线速度剖面对比.png"), dpi=图片分辨率, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(loss_record):
    """绘制训练损失曲线。"""
    set_chinese_style()
    plt.figure(figsize=(7.8, 5.2))

    曲线样式 = [
        ("总损失", "-", "o", 1.8),
        ("数据损失", "--", "s", 1.5),
        ("边界损失", "-.", "^", 1.5),
        ("动量方程损失", ":", "D", 1.6),
        ("连续性损失", (0, (5, 2, 1, 2)), "v", 1.5),
    ]
    标记间隔 = max(len(loss_record) // 12, 1)

    for col, linestyle, marker, linewidth in 曲线样式:
        if col not in loss_record.columns:
            continue
        plt.semilogy(
            loss_record["训练轮数"],
            loss_record[col],
            label=col,
            linestyle=linestyle,
            marker=marker,
            markevery=标记间隔,
            markersize=4.0,
            linewidth=linewidth,
            markerfacecolor="white",
            markeredgewidth=0.9,
        )
    plt.xlabel("训练轮数")
    plt.ylabel("损失值")
    plt.title("PINN 损失函数变化曲线", fontsize=标题字号)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    plt.legend(frameon=False, fontsize=图例字号)
    set_ascii_tick_format(plt.gca())
    save_current_figure("图9_损失函数变化曲线.png")


def plot_all_results(field_data, pred_data, loss_record, 训练点):
    """统一生成所有论文图。"""
    os.makedirs(图片目录, exist_ok=True)

    time_id = int(np.argmin(np.abs(field_data["t"] - 最终绘图时刻)))
    x = field_data["x"][time_id]
    y = field_data["y"][time_id]
    u_ref = field_data["u"][time_id]
    v_ref = field_data["v"][time_id]
    p_ref = field_data["p"][time_id]
    u_pred = pred_data["u"][time_id]
    v_pred = pred_data["v"][time_id]
    p_pred = pred_data["p"][time_id]

    speed_ref = np.sqrt(u_ref**2 + v_ref**2)
    speed_pred = np.sqrt(u_pred**2 + v_pred**2)
    speed_error = np.abs(speed_pred - speed_ref)

    plot_training_points(训练点)
    plot_cloud(x, y, speed_ref, "速度大小参考解云图", "速度大小/(m/s)", "图2_速度大小参考解云图.png")
    plot_cloud(x, y, speed_pred, "速度大小 PINN 重构云图", "速度大小/(m/s)", "图3_速度大小PINN重构云图.png")
    plot_cloud(x, y, speed_error, "速度大小绝对误差云图", "绝对误差/(m/s)", "图4_速度大小绝对误差云图.png", cmap="magma")
    plot_cloud(x, y, p_ref, "压力参考解云图", "压力", "图5_压力参考解云图.png", cmap="coolwarm")
    plot_cloud(x, y, p_pred, "压力 PINN 重构云图", "压力", "图6_压力PINN重构云图.png", cmap="coolwarm")
    plot_streamline_comparison(x, y, u_ref, v_ref, u_pred, v_pred)
    plot_centerline_profiles(x, y, u_ref, v_ref, u_pred, v_pred)
    plot_loss_curve(loss_record)
