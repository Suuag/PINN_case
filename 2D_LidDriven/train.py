"""
PINN 训练过程。
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    数据目录,
    模型目录,
    随机种子,
    学习率,
    训练轮数,
    打印间隔,
    监督点批量,
    边界点批量,
    残差点批量,
    数据损失权重,
    边界损失权重,
    动量损失权重,
    连续性损失权重,
    动量残差尺度,
    连续性残差尺度,
)
from pinn_model import build_model, ns_residual


def _sample_batch(rng, array, batch_size):
    """从数组中随机抽取一个小批量。"""
    total = array.shape[0]
    replace = total < batch_size
    idx = rng.choice(total, size=batch_size, replace=replace)
    return array[idx]


def train_pinn(训练点, output_mean, output_std):
    """
    使用 Adam 优化器训练 PINN。

    参数:
        训练点: data_loader.generate_training_points 生成的字典
        output_mean: u、v、p 均值
        output_std: u、v、p 标准差

    返回:
        model: 训练后的模型
        loss_record: 损失记录表
    """
    np.random.seed(随机种子)
    tf.random.set_seed(随机种子)
    rng = np.random.default_rng(随机种子)

    model = build_model(output_mean=output_mean, output_std=output_std)
    optimizer = tf.keras.optimizers.Adam(learning_rate=学习率)

    output_std_tf = tf.constant(output_std.reshape(1, 3), dtype=tf.float32)
    velocity_std_tf = tf.constant(output_std[:2].reshape(1, 2), dtype=tf.float32)

    @tf.function
    def train_step(supervised_xyt, supervised_uvp, boundary_xyt, boundary_uv, residual_xyt):
        """执行一次小批量训练。"""
        with tf.GradientTape() as tape:
            supervised_pred = model(supervised_xyt)
            data_loss = tf.reduce_mean(tf.square((supervised_pred - supervised_uvp) / output_std_tf))

            boundary_pred = model(boundary_xyt)[:, 0:2]
            boundary_loss = tf.reduce_mean(tf.square((boundary_pred - boundary_uv) / velocity_std_tf))

            momentum_x, momentum_y, continuity = ns_residual(model, residual_xyt)
            momentum_loss = tf.reduce_mean(
                tf.square(momentum_x / 动量残差尺度) + tf.square(momentum_y / 动量残差尺度)
            )
            continuity_loss = tf.reduce_mean(tf.square(continuity / 连续性残差尺度))

            total_loss = (
                数据损失权重 * data_loss
                + 边界损失权重 * boundary_loss
                + 动量损失权重 * momentum_loss
                + 连续性损失权重 * continuity_loss
            )

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss, data_loss, boundary_loss, momentum_loss, continuity_loss

    supervised_xyt = 训练点["监督输入"].astype(np.float32)
    supervised_uvp = 训练点["监督输出"].astype(np.float32)
    boundary_xyt = 训练点["边界输入"].astype(np.float32)
    boundary_uv = 训练点["边界速度"].astype(np.float32)
    residual_xyt = 训练点["残差输入"].astype(np.float32)

    loss_rows = []
    for epoch in range(1, 训练轮数 + 1):
        sup_idx = rng.choice(supervised_xyt.shape[0], size=监督点批量, replace=supervised_xyt.shape[0] < 监督点批量)
        sup_xyt_batch = tf.convert_to_tensor(supervised_xyt[sup_idx], dtype=tf.float32)
        sup_uvp_batch = tf.convert_to_tensor(supervised_uvp[sup_idx], dtype=tf.float32)

        bc_idx = rng.choice(boundary_xyt.shape[0], size=边界点批量, replace=boundary_xyt.shape[0] < 边界点批量)
        boundary_xyt_batch = tf.convert_to_tensor(boundary_xyt[bc_idx], dtype=tf.float32)
        boundary_uv_batch = tf.convert_to_tensor(boundary_uv[bc_idx], dtype=tf.float32)

        res_xyt_batch = tf.convert_to_tensor(_sample_batch(rng, residual_xyt, 残差点批量), dtype=tf.float32)

        total_loss, data_loss, boundary_loss, momentum_loss, continuity_loss = train_step(
            sup_xyt_batch,
            sup_uvp_batch,
            boundary_xyt_batch,
            boundary_uv_batch,
            res_xyt_batch,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == 训练轮数:
            loss_rows.append(
                {
                    "训练轮数": epoch,
                    "总损失": float(total_loss.numpy()),
                    "数据损失": float(data_loss.numpy()),
                    "边界损失": float(boundary_loss.numpy()),
                    "动量方程损失": float(momentum_loss.numpy()),
                    "连续性损失": float(continuity_loss.numpy()),
                }
            )

        if epoch == 1 or epoch % 打印间隔 == 0 or epoch == 训练轮数:
            print(
                f"第 {epoch:5d} 轮 | 总损失={total_loss.numpy():.4e} | "
                f"数据={data_loss.numpy():.4e} | 边界={boundary_loss.numpy():.4e} | "
                f"动量={momentum_loss.numpy():.4e} | 连续性={continuity_loss.numpy():.4e}"
            )

    loss_record = pd.DataFrame(loss_rows)
    os.makedirs(数据目录, exist_ok=True)
    os.makedirs(模型目录, exist_ok=True)
    loss_record.to_csv(os.path.join(数据目录, "训练损失记录.csv"), index=False, encoding="utf-8-sig")
    model.save(os.path.join(模型目录, "LidDriven_PINN模型.keras"))

    return model, loss_record
