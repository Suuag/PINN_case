"""
PINN 模型结构和二维不可压 Navier-Stokes 方程残差计算。
"""

import tensorflow as tf

from config import (
    方腔边长,
    时间最小值,
    时间最大值,
    运动黏性系数,
    网络隐藏层结构,
    激活函数,
)


@tf.keras.utils.register_keras_serializable(package="LidDrivenPINN")
class CoordinateScaleLayer(tf.keras.layers.Layer):
    """将 x、y、t 从物理区间缩放到 [-1, 1]。"""

    def __init__(self, lower_bound, upper_bound, **kwargs):
        super().__init__(**kwargs)
        self.lower_bound = list(lower_bound)
        self.upper_bound = list(upper_bound)

    def call(self, inputs):
        lower = tf.constant(self.lower_bound, dtype=inputs.dtype)
        upper = tf.constant(self.upper_bound, dtype=inputs.dtype)
        return 2.0 * (inputs - lower) / (upper - lower) - 1.0

    def get_config(self):
        config = super().get_config()
        config.update({"lower_bound": self.lower_bound, "upper_bound": self.upper_bound})
        return config


@tf.keras.utils.register_keras_serializable(package="LidDrivenPINN")
class OutputScaleLayer(tf.keras.layers.Layer):
    """将网络标准化输出还原为物理量 u、v、p。"""

    def __init__(self, output_mean, output_std, **kwargs):
        super().__init__(**kwargs)
        self.output_mean = list(output_mean)
        self.output_std = list(output_std)

    def call(self, inputs):
        mean = tf.constant(self.output_mean, dtype=inputs.dtype)
        std = tf.constant(self.output_std, dtype=inputs.dtype)
        return inputs * std + mean

    def get_config(self):
        config = super().get_config()
        config.update({"output_mean": self.output_mean, "output_std": self.output_std})
        return config


def build_model(output_mean, output_std):
    """
    构建 PINN 神经网络。

    参数:
        output_mean: u、v、p 的均值
        output_std: u、v、p 的标准差

    返回:
        Keras 模型，输入为 [x, y, t]，输出为 [u, v, p]
    """
    inputs = tf.keras.Input(shape=(3,), name="input_x_y_t")
    z = CoordinateScaleLayer(
        lower_bound=[0.0, 0.0, 时间最小值],
        upper_bound=[方腔边长, 方腔边长, 时间最大值],
        name="coordinate_scale",
    )(inputs)

    for neurons in 网络隐藏层结构:
        z = tf.keras.layers.Dense(
            neurons,
            activation=激活函数,
            kernel_initializer="glorot_normal",
        )(z)

    raw_outputs = tf.keras.layers.Dense(3, activation=None, name="raw_u_v_p")(z)
    outputs = OutputScaleLayer(output_mean=output_mean, output_std=output_std, name="physical_u_v_p")(raw_outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def predict_uvp(model, x, y, t):
    """
    预测速度和压力。

    参数:
        model: PINN 模型
        x, y, t: 坐标张量，形状均为 [N, 1]

    返回:
        u, v, p 三个张量
    """
    inputs = tf.concat([x, y, t], axis=1)
    output = model(inputs)
    return output[:, 0:1], output[:, 1:2], output[:, 2:3]


def ns_residual(model, points, nu=运动黏性系数):
    """
    计算二维不可压 Navier-Stokes 方程残差。

    参数:
        model: PINN 模型
        points: [x, y, t] 输入点，形状为 [N, 3]
        nu: 运动黏性系数

    返回:
        x 方向动量残差、y 方向动量残差、连续性残差
    """
    points = tf.convert_to_tensor(points, dtype=tf.float32)
    x = points[:, 0:1]
    y = points[:, 1:2]
    t = points[:, 2:3]

    with tf.GradientTape(persistent=True) as second_tape:
        second_tape.watch([x, y, t])
        with tf.GradientTape(persistent=True) as first_tape:
            first_tape.watch([x, y, t])
            u, v, p = predict_uvp(model, x, y, t)

        u_x = first_tape.gradient(u, x)
        u_y = first_tape.gradient(u, y)
        u_t = first_tape.gradient(u, t)
        v_x = first_tape.gradient(v, x)
        v_y = first_tape.gradient(v, y)
        v_t = first_tape.gradient(v, t)
        p_x = first_tape.gradient(p, x)
        p_y = first_tape.gradient(p, y)

    u_xx = second_tape.gradient(u_x, x)
    u_yy = second_tape.gradient(u_y, y)
    v_xx = second_tape.gradient(v_x, x)
    v_yy = second_tape.gradient(v_y, y)

    del first_tape
    del second_tape

    momentum_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y

    return momentum_x, momentum_y, continuity
