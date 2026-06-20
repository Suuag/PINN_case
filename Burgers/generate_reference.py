"""
用有限体积方法生成一维黏性 Burgers 方程参考解。

守恒型有限体积格式比简单有限差分更适合处理 Burgers 方程中的陡梯度结构。
本程序使用 Rusanov 数值通量处理非线性对流项，扩散项使用中心差分。
"""

import numpy as np

from config import (
    x最小值,
    x最大值,
    t最小值,
    t最大值,
    黏性系数,
    参考解空间点数,
    参考解时间点数,
    参考解CFL,
    参考解文件,
    创建输出目录,
)


def initial_condition(x):
    """
    定义 Burgers 方程初始条件。

    参数:
        x: 空间坐标数组

    返回:
        u(x, 0) 的初始速度分布
    """
    return -np.sin(np.pi * x)


def rusanov_flux(u_left, u_right):
    """
    计算 Burgers 方程对流项的 Rusanov 数值通量。

    Burgers 方程通量为 F(u)=0.5*u^2，局部最大波速为 max(|u_left|, |u_right|)。
    """
    flux_left = 0.5 * u_left**2
    flux_right = 0.5 * u_right**2
    wave_speed = np.maximum(np.abs(u_left), np.abs(u_right))
    return 0.5 * (flux_left + flux_right) - 0.5 * wave_speed * (u_right - u_left)


def finite_volume_rhs(u, dx, nu):
    """
    计算有限体积半离散右端项。

    参数:
        u: 网格中心速度
        dx: 网格尺度
        nu: 黏性系数

    返回:
        du/dt
    """
    # 两侧 ghost cell 用 Dirichlet 边界 u=0。
    u_ext = np.zeros(u.size + 2, dtype=np.float64)
    u_ext[1:-1] = u

    u_left = u_ext[:-1]
    u_right = u_ext[1:]
    conv_flux = rusanov_flux(u_left, u_right)
    conv_term = -(conv_flux[1:] - conv_flux[:-1]) / dx

    diffusion_term = nu * (u_ext[2:] - 2.0 * u_ext[1:-1] + u_ext[:-2]) / (dx * dx)
    return conv_term + diffusion_term


def stable_time_step(u, dx, nu):
    """
    根据对流和扩散约束估计显式推进时间步。
    """
    max_speed = max(float(np.max(np.abs(u))), 1.0e-8)
    dt_conv = dx / max_speed
    dt_diff = 0.5 * dx * dx / max(nu, 1.0e-12)
    return 参考解CFL * min(dt_conv, dt_diff)


def generate_reference_solution():
    """
    生成 Burgers 方程高分辨率有限体积参考解。

    返回:
        x: 网格中心坐标
        t: 输出时间
        u_record: 参考解矩阵，形状为 [时间点数, 空间点数]
    """
    创建输出目录()

    dx = (x最大值 - x最小值) / 参考解空间点数
    x = np.linspace(x最小值 + 0.5 * dx, x最大值 - 0.5 * dx, 参考解空间点数)
    t = np.linspace(t最小值, t最大值, 参考解时间点数)

    u = initial_condition(x).astype(np.float64)
    u_record = np.zeros((t.size, x.size), dtype=np.float64)
    u_record[0] = u

    current_time = t最小值
    output_id = 1

    while output_id < t.size:
        target_time = t[output_id]
        while current_time < target_time - 1.0e-14:
            dt = min(stable_time_step(u, dx, 黏性系数), target_time - current_time)

            # 二阶 Runge-Kutta 时间推进，提高参考解时间精度。
            k1 = finite_volume_rhs(u, dx, 黏性系数)
            u_mid = u + dt * k1
            k2 = finite_volume_rhs(u_mid, dx, 黏性系数)
            u = 0.5 * u + 0.5 * (u_mid + dt * k2)

            current_time += dt

        u_record[output_id] = u
        output_id += 1

    np.savez(参考解文件, x=x, t=t, u=u_record, nu=黏性系数, dx=dx, method="finite_volume_rusanov")
    return x, t, u_record


if __name__ == "__main__":
    generate_reference_solution()
    print("参考解已生成：", 参考解文件)
