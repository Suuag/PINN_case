"""
二维顶盖驱动方腔流动 PINN 重构项目总入口。

直接运行:
    python main.py
"""

import pandas as pd
import tensorflow as tf

from config import 创建输出目录, 设置随机种子, 模型目录
from data_loader import load_openfoam_field, calculate_output_statistics, generate_training_points
from train import train_pinn
from evaluate import predict_full_field, calculate_error_tables
from plot_results import plot_all_results
import pinn_model


def main():
    """执行完整 PINN 重构流程。"""
    创建输出目录()
    设置随机种子()

    print("步骤1：读取 OpenFOAM MAT 数据")
    field_data = load_openfoam_field()
    output_mean, output_std = calculate_output_statistics(field_data)
    print("输出变量均值 u,v,p:", output_mean)
    print("输出变量标准差 u,v,p:", output_std)

    print("步骤2：生成稀疏监督点、边界点和方程残差点")
    training_points = generate_training_points(field_data)

    print("步骤3：训练 PINN 模型")
    model, loss_record = train_pinn(training_points, output_mean, output_std)

    print("步骤4：在完整 OpenFOAM 网格上重构流场")
    pred_data = predict_full_field(model, field_data)

    print("步骤5：计算误差表格并绘制论文图")
    overall_table, _, _, _ = calculate_error_tables(field_data, pred_data)
    plot_all_results(field_data, pred_data, loss_record, training_points)

    # 验证保存模型能否重新加载，避免后续复现实验时才发现序列化问题。
    tf.keras.models.load_model(f"{模型目录}/LidDriven_PINN模型.keras")

    print("全部计算完成。整体误差指标如下：")
    print(overall_table.to_string(index=False))


if __name__ == "__main__":
    main()
