"""
Burgers 方程 PINN 验证项目总入口。

直接运行:
    python main.py
"""

import os
import numpy as np

from config import 创建输出目录, 数据目录, 模型目录
from generate_reference import generate_reference_solution
from train import generate_training_points, train_pinn
from evaluate import predict_on_reference_grid, calculate_error_tables
from plot_results import plot_all_results


def main():
    """执行完整的高精度 PINN 验证流程。"""
    创建输出目录()

    print("步骤1：重新生成高分辨率有限体积参考解")
    x, t, u_ref = generate_reference_solution()

    print("步骤2：生成 PINN 初值点、边界点、残差点和内部监督点")
    训练点 = generate_training_points(x, t, u_ref)

    print("步骤3：训练带监督点约束的 PINN 模型")
    model, 损失记录 = train_pinn(训练点)
    损失记录.to_csv(os.path.join(数据目录, "训练损失记录.csv"), index=False, encoding="utf-8-sig")

    print("步骤4：在参考解网格上进行预测")
    u_pred = predict_on_reference_grid(model, x, t)
    np.savez(os.path.join(数据目录, "PINN预测结果.npz"), x=x, t=t, u_pred=u_pred, u_ref=u_ref)

    print("步骤5：保存模型、误差表格和论文图")
    model.save(os.path.join(模型目录, "Burgers_PINN模型.keras"))
    整体误差表, _, _ = calculate_error_tables(x, t, u_ref, u_pred)
    plot_all_results(x, t, u_ref, u_pred, 损失记录, 训练点)

    print("全部计算完成。整体误差指标如下：")
    print(整体误差表.to_string(index=False))


if __name__ == "__main__":
    main()
