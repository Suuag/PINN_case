import os
import pyvista as pv
import subprocess
import numpy as np
import matplotlib.pyplot as plt


class FoamMeshVisualizer:
    """Openfoam网格可视化工具类，支持压力、速度分量的独立图像渲染"""

    def __init__(self, case_dir, openfoam_bashrc="/opt/openfoam9/etc/bashrc"):
        self.case_dir = case_dir
        self.openfoam_bashrc = openfoam_bashrc
        self.vtk_dir = os.path.join(case_dir, "VTK")

        # 渲染控制开关
        self.show_pressure = False  # 默认不显示压力
        self.show_u_velocity = False  # 默认不显示U速度分量
        self.show_v_velocity = False  # 默认不显示V速度分量

        # 颜色映射配置
        self.pressure_cmap = "viridis"
        self.u_cmap = "coolwarm"
        self.v_cmap = "plasma"

        # 存储VTK网格对象
        self.mesh = None

    def convert_foam_to_vtk(self):
        """转换Openfoam案例为VTK格式"""
        #这里其实也是一个命令行工具，这里只是调用了
        if not os.path.exists(os.path.join(self.case_dir, "constant")):
            raise FileNotFoundError(f"未找到案例目录下的constant文件夹: {self.case_dir}")

        if not os.path.exists(self.openfoam_bashrc):
            raise FileNotFoundError(f"Openfoam配置文件不存在: {self.openfoam_bashrc}")

        try:
            cmd = f"source {self.openfoam_bashrc} && cd {self.case_dir} && foamToVTK"
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable="/bin/bash"
            )
            print(f"网格已转换为VTK格式，保存至: {self.vtk_dir}")

            # 打印VTK目录结构
            print("VTK目录内容：")
            for root, dirs, files in os.walk(self.vtk_dir):
                level = root.replace(self.vtk_dir, "").count(os.sep)
                indent = " " * 4 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 4 * (level + 1)
                for f in files:
                    print(f"{subindent}{f}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"foamToVTK转换失败:\n错误输出: {e.stderr}\n标准输出: {e.stdout}")

        return self.vtk_dir

    def _load_mesh(self):
        """加载VTK网格文件（内部方法）"""
        vtk_files = []
        for root, _, files in os.walk(self.vtk_dir):
            for file in files:
                if file.endswith((".vtk", ".vtu")):
                    vtk_files.append(os.path.join(root, file))

        if not vtk_files:
            raise FileNotFoundError(f"在{self.vtk_dir}中未找到VTK格式文件（.vtk或.vtu）")

        # 选择最大的VTK文件（通常是包含完整数据的文件）
        vtk_file = max(vtk_files, key=lambda x: os.path.getsize(x))
        print(f"加载网格文件：{vtk_file}")
        self.mesh = pv.read(vtk_file)
        return self.mesh

    def set_render_flags(self, show_pressure=False, show_u=False, show_v=False):
        """设置渲染开关"""
        self.show_pressure = show_pressure
        self.show_u_velocity = show_u
        self.show_v_velocity = show_v

    def _create_base_plotter(self, is_2d):
        """创建基础绘图器（含基础网格）"""
        plotter = pv.Plotter()
        # 基础网格渲染（浅灰色带边缘）
        plotter.add_mesh(
            self.mesh,
            color="lightgray",
            show_edges=True,
            edge_color="black",
            line_width=0.5,
            opacity=0.5  # 基础网格半透明
        )
        # 配置基础属性
        plotter.set_background("white")
        # 设置视角
        if is_2d:
            plotter.view_xy()
        else:
            plotter.view_isometric()
        return plotter

    def visualize(self, output_dir=None):
        """可视化网格及物理量，每个启用的标志生成独立图像"""
        if not self.mesh:
            self._load_mesh()

        # 判断网格维度（2D/3D）
        points = self.mesh.points
        z_coords = points[:, 2]
        is_2d = np.allclose(z_coords, z_coords[0], atol=1e-6)

        # 收集需要渲染的标志
        render_tasks = []
        if self.show_pressure:
            render_tasks.append(("pressure", "Pressure", "p", self.pressure_cmap))
        if self.show_u_velocity:
            render_tasks.append(("u_velocity", "U Velocity (x)", "U_x", self.u_cmap))
        if self.show_v_velocity:
            render_tasks.append(("v_velocity", "V Velocity (y)", "U_y", self.v_cmap))

        if not render_tasks:
            print("警告：未启用任何渲染标志，无法生成图像")
            return

        # 为每个任务生成独立图像
        for task in render_tasks:
            task_name, title, scalar_name, cmap = task
            plotter = self._create_base_plotter(is_2d)

            # 添加对应物理量
            if task_name == "pressure" and "p" in self.mesh.point_data:
                plotter.add_mesh(
                    self.mesh,
                    scalars="p",
                    cmap=cmap,
                    show_edges=False,
                    label=title,
                    scalar_bar_args={"title": title}
                )
            elif task_name == "pressure":
                print("警告：VTK文件中未找到压力数据（'p'），跳过压力渲染")
                continue

            elif task_name == "u_velocity" and "U" in self.mesh.point_data:
                u_data = self.mesh.point_data["U"][:, 0]
                self.mesh.point_data["U_x"] = u_data
                plotter.add_mesh(
                    self.mesh,
                    scalars="U_x",
                    cmap=cmap,
                    show_edges=False,
                    label=title,
                    scalar_bar_args={"title": title}
                )
            elif task_name == "u_velocity":
                print("警告：VTK文件中未找到速度数据（'U'），跳过U速度渲染")
                continue

            elif task_name == "v_velocity" and "U" in self.mesh.point_data:
                v_data = self.mesh.point_data["U"][:, 1]
                self.mesh.point_data["U_y"] = v_data
                plotter.add_mesh(
                    self.mesh,
                    scalars="U_y",
                    cmap=cmap,
                    show_edges=False,
                    label=title,
                    scalar_bar_args={"title": title}
                )
            elif task_name == "v_velocity":
                print("警告：VTK文件中未找到速度数据（'U'），跳过V速度渲染")
                continue

            # 设置标题
            plotter.add_title(f"Openfoam Mesh - {title}")

            # 显示或保存
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{task_name}.png")
                plotter.screenshot(output_path, dpi=300)
                print(f"图像已保存至: {output_path}")
            else:
                print(f"显示 {title} 图像（关闭窗口后显示下一个）")
                plotter.show()

    def remove_vtk_files(self):
        """删除生成的VTK文件"""
        if os.path.exists(self.vtk_dir):
            for root, dirs, files in os.walk(self.vtk_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.vtk_dir)
            print(f"已删除VTK文件目录：{self.vtk_dir}")
        else:
            print(f"VTK文件目录不存在：{self.vtk_dir}")


# 使用示例
if __name__ == "__main__":
    # 配置案例路径
    case_dir = "./cavity"
    openfoam_bashrc = "/opt/openfoam9/etc/bashrc"

    # 初始化可视化器
    visualizer = FoamMeshVisualizer(case_dir, openfoam_bashrc)

    # 转换Openfoam案例为VTK
    visualizer.convert_foam_to_vtk()

    # 设置需要渲染的物理量（每个True对应一个独立图像）
    visualizer.set_render_flags(
        show_pressure=True,
        show_u=True,
        show_v=True  # 这里3个True会生成3个独立图像
    )

    # 可视化
    visualizer.visualize()

    # 可选：删除VTK文件
    # visualizer.remove_vtk_files()
