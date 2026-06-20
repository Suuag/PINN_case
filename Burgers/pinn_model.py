"""
PINN 网络结构与 Burgers 方程残差计算。
"""

import tensorflow as tf

from config import (
    x最小值,
    x最大值,
    t最小值,
    t最大值,
    黏性系数,
    网络隐藏层结构,
    激活函数,
)


@tf.keras.utils.register_keras_serializable(package="BurgersPINN")
class CoordinateScaleLayer(tf.keras.layers.Layer):
    """
    将输入坐标从物理区间线性缩放到 [-1, 1]。

    这里单独写成 Keras Layer，是为了保证模型可以正常保存为 .keras 文件。
    """

    def __init__(self, lower_bound, upper_bound, **kwargs):
        super().__init__(**kwargs)
        self.lower_bound = list(lower_bound)
        self.upper_bound = list(upper_bound)

    def call(self, inputs):
        下界 = tf.constant(self.lower_bound, dtype=inputs.dtype)
        上界 = tf.constant(self.upper_bound, dtype=inputs.dtype)
        return 2.0 * (inputs - 下界) / (上界 - 下界) - 1.0

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
            }
        )
        return config


def build_model():
    """
    构建 PINN 全连接神经网络。

    返回:
        TensorFlow Keras 模型，输入为 [x, t]，输出为 u
    """
    # Keras 的内部层名称只能使用 ASCII 安全字符，中文说明放在注释中。
    输入层 = tf.keras.Input(shape=(2,), name="input_x_t")

    # 将 x 和 t 统一缩放到 [-1, 1]，有利于 tanh 网络训练。
    z = CoordinateScaleLayer(
        lower_bound=[x最小值, t最小值],
        upper_bound=[x最大值, t最大值],
        name="coordinate_scale",
    )(输入层)

    for 神经元数量 in 网络隐藏层结构:
        z = tf.keras.layers.Dense(
            神经元数量,
            activation=激活函数,
            kernel_initializer="glorot_normal",
        )(z)

    输出层 = tf.keras.layers.Dense(1, activation=None, name="predict_u")(z)
    return tf.keras.Model(inputs=输入层, outputs=输出层)


def predict_u(model, x, t):
    """
    计算网络预测速度。

    参数:
        model: PINN 模型
        x: 空间坐标张量，形状为 [N, 1]
        t: 时间坐标张量，形状为 [N, 1]

    返回:
        预测速度 u，形状为 [N, 1]
    """
    输入坐标 = tf.concat([x, t], axis=1)
    return model(输入坐标)


def pde_residual(model, x, t, nu=黏性系数):
    """
    计算 Burgers 方程残差 f = u_t + u u_x - nu u_xx。

    参数:
        model: PINN 模型
        x: 空间坐标张量
        t: 时间坐标张量
        nu: 黏性系数

    返回:
        方程残差张量
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as 二阶求导带:
        二阶求导带.watch(x)
        二阶求导带.watch(t)

        with tf.GradientTape(persistent=True) as 一阶求导带:
            一阶求导带.watch(x)
            一阶求导带.watch(t)
            u = predict_u(model, x, t)

        u_x = 一阶求导带.gradient(u, x)
        u_t = 一阶求导带.gradient(u, t)

    u_xx = 二阶求导带.gradient(u_x, x)

    del 一阶求导带
    del 二阶求导带

    return u_t + u * u_x - nu * u_xx
