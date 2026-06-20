"""
PINN 训练相关函数。

本版本在传统 PINN 的初值、边界和方程残差约束基础上，增加有限体积参考解的内部监督点。
监督点重点覆盖激波附近区域，用来降低 Burgers 方程陡梯度位置误差。
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    x最小值,
    x最大值,
    t最小值,
    t最大值,
    初值点数量,
    边界点数量,
    方程残差点数量,
    监督点数量,
    激波监督点比例,
    随机种子,
    学习率,
    训练轮数,
    打印间隔,
    学习率衰减表,
    初值损失权重,
    边界损失权重,
    方程损失权重,
    监督损失权重,
)
from generate_reference import initial_condition
from pinn_model import build_model, predict_u, pde_residual


def _sample_supervised_points(x_ref, t_ref, u_ref, rng):
    """
    从参考解中抽取内部监督点。

    一部分点按 |u_x| 加权采样，重点学习激波附近的陡梯度结构；
    另一部分点均匀采样，避免只拟合局部区域。
    """
    X, T = np.meshgrid(x_ref, t_ref)
    dx = float(x_ref[1] - x_ref[0])
    ux_abs = np.abs(np.gradient(u_ref, dx, axis=1))

    # 避免初始时刻和边界点重复承担监督功能，内部监督主要关注 t>0 的演化场。
    valid_mask = np.ones_like(u_ref, dtype=bool)
    valid_mask[0, :] = False

    flat_x = X[valid_mask].ravel()
    flat_t = T[valid_mask].ravel()
    flat_u = u_ref[valid_mask].ravel()
    flat_weight = ux_abs[valid_mask].ravel()

    shock_count = int(监督点数量 * 激波监督点比例)
    uniform_count = 监督点数量 - shock_count

    weight = flat_weight + 1.0e-8
    weight = weight / np.sum(weight)

    shock_idx = rng.choice(flat_x.size, size=shock_count, replace=False, p=weight)
    uniform_idx = rng.choice(flat_x.size, size=uniform_count, replace=False)
    idx = np.concatenate([shock_idx, uniform_idx])
    rng.shuffle(idx)

    return (
        flat_x[idx, None].astype(np.float32),
        flat_t[idx, None].astype(np.float32),
        flat_u[idx, None].astype(np.float32),
    )


def generate_training_points(x_ref=None, t_ref=None, u_ref=None):
    """
    生成 PINN 训练点，包括初值点、边界点、方程残差点和内部监督点。

    参数:
        x_ref, t_ref, u_ref: 参考解网格和参考解矩阵。若提供，则抽取内部监督点。

    返回:
        字典，包含各类训练点及对应约束值。
    """
    rng = np.random.default_rng(随机种子)

    x初值 = rng.uniform(x最小值, x最大值, (初值点数量, 1)).astype(np.float32)
    t初值 = np.zeros_like(x初值, dtype=np.float32)
    u初值 = initial_condition(x初值).astype(np.float32)

    t边界 = rng.uniform(t最小值, t最大值, (边界点数量, 1)).astype(np.float32)
    x左边界 = np.full_like(t边界, x最小值, dtype=np.float32)
    x右边界 = np.full_like(t边界, x最大值, dtype=np.float32)
    x边界 = np.vstack([x左边界, x右边界]).astype(np.float32)
    t边界 = np.vstack([t边界, t边界]).astype(np.float32)
    u边界 = np.zeros_like(x边界, dtype=np.float32)

    x方程 = rng.uniform(x最小值, x最大值, (方程残差点数量, 1)).astype(np.float32)
    t方程 = rng.uniform(t最小值, t最大值, (方程残差点数量, 1)).astype(np.float32)

    训练点 = {
        "x初值": x初值,
        "t初值": t初值,
        "u初值": u初值,
        "x边界": x边界,
        "t边界": t边界,
        "u边界": u边界,
        "x方程": x方程,
        "t方程": t方程,
    }

    if x_ref is not None and t_ref is not None and u_ref is not None and 监督点数量 > 0:
        x监督, t监督, u监督 = _sample_supervised_points(x_ref, t_ref, u_ref, rng)
        训练点.update({"x监督": x监督, "t监督": t监督, "u监督": u监督})

    return 训练点


def _update_learning_rate(optimizer, epoch):
    """
    按配置表分段调整学习率。
    """
    current_lr = 学习率
    for start_epoch, lr in sorted(学习率衰减表.items()):
        if epoch >= start_epoch:
            current_lr = lr
    optimizer.learning_rate.assign(current_lr)
    return current_lr


def train_pinn(训练点):
    """
    使用 Adam 优化器训练 PINN。

    参数:
        训练点: generate_training_points 生成的训练点字典

    返回:
        model: 训练后的 PINN 模型
        loss_record: 损失记录 DataFrame
    """
    np.random.seed(随机种子)
    tf.random.set_seed(随机种子)

    model = build_model()
    优化器 = tf.keras.optimizers.Adam(learning_rate=学习率)

    x初值 = tf.convert_to_tensor(训练点["x初值"], dtype=tf.float32)
    t初值 = tf.convert_to_tensor(训练点["t初值"], dtype=tf.float32)
    u初值 = tf.convert_to_tensor(训练点["u初值"], dtype=tf.float32)

    x边界 = tf.convert_to_tensor(训练点["x边界"], dtype=tf.float32)
    t边界 = tf.convert_to_tensor(训练点["t边界"], dtype=tf.float32)
    u边界 = tf.convert_to_tensor(训练点["u边界"], dtype=tf.float32)

    x方程 = tf.convert_to_tensor(训练点["x方程"], dtype=tf.float32)
    t方程 = tf.convert_to_tensor(训练点["t方程"], dtype=tf.float32)

    有监督点 = all(key in 训练点 for key in ["x监督", "t监督", "u监督"])
    if 有监督点:
        x监督 = tf.convert_to_tensor(训练点["x监督"], dtype=tf.float32)
        t监督 = tf.convert_to_tensor(训练点["t监督"], dtype=tf.float32)
        u监督 = tf.convert_to_tensor(训练点["u监督"], dtype=tf.float32)
    else:
        x监督 = tf.zeros((1, 1), dtype=tf.float32)
        t监督 = tf.zeros((1, 1), dtype=tf.float32)
        u监督 = tf.zeros((1, 1), dtype=tf.float32)

    @tf.function
    def 单步训练():
        """执行一次梯度下降更新。"""
        with tf.GradientTape() as 参数求导带:
            u初值预测 = predict_u(model, x初值, t初值)
            u边界预测 = predict_u(model, x边界, t边界)
            f方程 = pde_residual(model, x方程, t方程)
            u监督预测 = predict_u(model, x监督, t监督)

            初值损失 = tf.reduce_mean(tf.square(u初值预测 - u初值))
            边界损失 = tf.reduce_mean(tf.square(u边界预测 - u边界))
            方程损失 = tf.reduce_mean(tf.square(f方程))
            监督损失 = tf.reduce_mean(tf.square(u监督预测 - u监督))

            总损失 = (
                初值损失权重 * 初值损失
                + 边界损失权重 * 边界损失
                + 方程损失权重 * 方程损失
                + 监督损失权重 * 监督损失
            )

        梯度 = 参数求导带.gradient(总损失, model.trainable_variables)
        优化器.apply_gradients(zip(梯度, model.trainable_variables))
        return 总损失, 初值损失, 边界损失, 方程损失, 监督损失

    损失记录 = []
    for epoch in range(1, 训练轮数 + 1):
        当前学习率 = _update_learning_rate(优化器, epoch)
        总损失, 初值损失, 边界损失, 方程损失, 监督损失 = 单步训练()

        if epoch == 1 or epoch % 10 == 0 or epoch == 训练轮数:
            损失记录.append(
                {
                    "训练轮数": epoch,
                    "学习率": 当前学习率,
                    "总损失": float(总损失.numpy()),
                    "初值损失": float(初值损失.numpy()),
                    "边界损失": float(边界损失.numpy()),
                    "方程残差损失": float(方程损失.numpy()),
                    "监督损失": float(监督损失.numpy()),
                }
            )

        if epoch == 1 or epoch % 打印间隔 == 0 or epoch == 训练轮数:
            print(
                f"第 {epoch:5d} 轮 | 学习率={当前学习率:.1e} | 总损失={总损失.numpy():.4e} | "
                f"初值={初值损失.numpy():.4e} | 边界={边界损失.numpy():.4e} | "
                f"方程={方程损失.numpy():.4e} | 监督={监督损失.numpy():.4e}"
            )

    return model, pd.DataFrame(损失记录)
