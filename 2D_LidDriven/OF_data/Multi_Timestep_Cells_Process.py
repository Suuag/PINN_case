import os
import pyvista as pv
import pandas as pd
import subprocess
import scipy.io as sio  # 用于MAT文件处理
from mesh_visualise import FoamMeshVisualizer


class TimeStepDataConverter(FoamMeshVisualizer):
    """处理多时间步OpenFOAM结果，支持转换为VTK、CSV和MAT格式"""

    def __init__(self, case_dir, openfoam_bashrc="/opt/openfoam9/etc/bashrc"):
        super().__init__(case_dir, openfoam_bashrc)
        self.onestep_VTKnames = []  # 初始化空列表
        self.csv_output_root = os.path.join(case_dir, "time_step_csv")  # CSV输出目录
        self.mat_output_root = os.path.join(case_dir, "time_step_mat")  # MAT输出目录
        self.time_steps = []  # 存储识别的时间步

    def clean_previous_results(self):
        """删除之前生成的VTK文件和time_step_mat文件夹及其内容"""
        # 删除VTK目录及其中所有文件
        if os.path.exists(self.vtk_dir):
            try:
                for root, dirs, files in os.walk(self.vtk_dir, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                os.rmdir(self.vtk_dir)
                print(f"已删除VTK目录: {self.vtk_dir}")
            except Exception as e:
                print(f"删除VTK目录时出错: {e}")
        
        # 删除time_step_mat目录及其中所有文件
        if os.path.exists(self.mat_output_root):
            try:
                for file in os.listdir(self.mat_output_root):
                    file_path = os.path.join(self.mat_output_root, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(self.mat_output_root)
                print(f"已删除time_step_mat目录: {self.mat_output_root}")
            except Exception as e:
                print(f"删除time_step_mat目录时出错: {e}")

    def _identify_time_steps(self):
        """识别并按时间顺序排序所有时间步文件夹"""
        all_entries = os.listdir(self.case_dir)
        time_steps = []
        for entry in all_entries:
            entry_path = os.path.join(self.case_dir, entry)
            if os.path.isdir(entry_path):
                try:
                    time_value = float(entry)
                    time_steps.append((entry, time_value))
                except ValueError:
                    continue

        # 按时间值排序，解决浮点数精度问题
        time_steps.sort(key=lambda x: (len(str(x[1])), x[1]))
        self.time_steps = [ts[0] for ts in time_steps]

        if not self.time_steps:
            raise ValueError(f"在{self.case_dir}中未找到时间步文件夹")

        print(f"已识别{len(self.time_steps)}个时间步，按顺序为: {self.time_steps}")
        return self.time_steps

    def _get_cell_centers_and_data(self, vtk_file):
        """获取网格中心和物理数据"""
        # 复用父类的网格加载逻辑
        self.mesh = pv.read(vtk_file)
        # 这个 pyvista 读取 VTK 文件 ， 读取的mesh 是一个 pyvista.PolyData 对象 , 比较复杂
        cell_centers = self.mesh.cell_centers().points
        # 这个读取的 cell_centers() 返回的是一个 numpy 数组 这里是( 9200 , 3 ) , 也就是 cell_numbers ， 3
        if cell_centers.size == 0:
            print(f"警告：{vtk_file} 中未找到网格单元数据")
            return None

        data = {
            "center_x": cell_centers[:, 0],
            "center_y": cell_centers[:, 1],
            #"center_z": cell_centers[:, 2],
            "cell_index" : list(range(cell_centers.shape[0]))
        }

        # 提取压力数据
        if "p" in self.mesh.cell_data:
            data["pressure"] = self.mesh.cell_data["p"]

        # 提取速度分量数据
        if "U" in self.mesh.cell_data:
            velocity = self.mesh.cell_data["U"]
            data["velocity_x"] = velocity[:, 0]
            data["velocity_y"] = velocity[:, 1]
            data["velocity_z"] = velocity[:, 2]
        elif "U" in self.mesh.point_data:
            cell_vel = self.mesh.interpolate("U", point_data_to_cell_data=True).cell_data["U"]
            data["velocity_x"] = cell_vel[:, 0]
            data["velocity_y"] = cell_vel[:, 1]
            data["velocity_z"] = cell_vel[:, 2]

        return data

    def _convert_time_step_to_vtk(self, time_step):
        """转换指定时间步到VTK格式"""
        try:
            ## 这里是调用Openfoam的命令 foamtoVTK 其实可以打开 terminal 手动输入命令
            cmd = f"source {self.openfoam_bashrc} && cd {self.case_dir} && foamToVTK -time {time_step}"
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                executable="/bin/bash",
                encoding='utf-8'
            )
            print(f"时间步 {time_step} 已转换为VTK格式，保存至: {self.vtk_dir}")
            # 直接找到最新且文件名最大的VTK文件，而不遍历所有文件
            latest_vtk_file = None
            for root, _, files in os.walk(self.vtk_dir):
                vtk_files = [f for f in files if f.endswith(".vtk")]
                if vtk_files:
                    # 从文件名中提取时间步数值进行比较
                    def extract_time_value(filename):
                        # 移除扩展名和后缀，提取时间步数值
                        name_without_ext = filename.replace('.vtk', '')
                        parts = name_without_ext.split('_')
                        try:
                            return float(parts[-1])  # 返回最后一个下划线后的数值
                        except (ValueError, IndexError):
                            return 0.0

                    latest_vtk_file = max(vtk_files, key=extract_time_value)
                    break

            # 保存这一个时间步骤下的实例的名称
            if latest_vtk_file:
                self.onestep_VTKnames.append(latest_vtk_file)
            else:
                self.onestep_VTKnames.append([])
            return True
        except subprocess.CalledProcessError as e:  # 这种报错基本不发生
            print(f"警告：时间步 {time_step} 转换失败，跳过:\n错误输出: {e.stderr}")
            return False

    def convert_to_mat(self, output_dir=None, include_csv=False):
        """
        将所有时间步数据转换为MAT格式
        output_dir: 自定义MAT输出目录
        include_csv: 是否同时生成CSV文件
        """
        # 设置输出目录
        if output_dir:
            self.mat_output_root = output_dir
        os.makedirs(self.mat_output_root, exist_ok=True)
        if include_csv:
            os.makedirs(self.csv_output_root, exist_ok=True)

        # 识别时间步
        self._identify_time_steps()  # 返回所有识别的时间的列表
        # 创建一个 timesteps[] 到 vtk_files[] 的对应索引列表
        timesteps_to_vtk_files = []

        for time_step in self.time_steps:   # 遍历所有时间步
            print(f"\n处理时间步: {time_step}") # 打印当前处理的时间步

            # 转换当前时间步到VTK
            if not self._convert_time_step_to_vtk(time_step):
                continue

            # 创建时间步和对应VTK文件名的索引列表
            timesteps_to_vtk_files.append((time_step, self.onestep_VTKnames[-1]))  # 添加当前时间步和其VTK文件名
            print((time_step, self.onestep_VTKnames[-1]))

            # 提取数据
            vtk_filename = self.onestep_VTKnames[-1]
            if vtk_filename:
                data_dict = self._get_cell_centers_and_data(os.path.join(self.vtk_dir, vtk_filename))
            else:
                print(f"警告：时间步 {time_step} 没有找到VTK文件")
                continue
            if not data_dict:
                continue

            # 保存为MAT文件
            mat_filename = f"{time_step}.mat"
            mat_path = os.path.join(self.mat_output_root, mat_filename)
            sio.savemat(mat_path, data_dict)
            print(f"已保存MAT文件: {mat_path} (包含 {len(data_dict['center_x'])} 个网格单元)")

            # 可选：同时保存为CSV
            if include_csv:
                csv_filename = f"{time_step}.csv"
                csv_path = os.path.join(self.csv_output_root, csv_filename)
                pd.DataFrame(data_dict).to_csv(csv_path, index=False)
                print(f"已保存CSV文件: {csv_path}")

        print(f"\n所有时间步转换完成，MAT文件保存至: {self.mat_output_root}")
        # 保存 timesteps_to_vtk_files 列表 csv 文件
        timesteps_to_vtk_files_names = f"time_step_to_vtk_files.csv"
        csv_path = os.path.join(self.case_dir, timesteps_to_vtk_files_names)
        df = pd.DataFrame(timesteps_to_vtk_files, columns=['Time_Step', 'VTK_File'])
        df.to_csv(csv_path, index=False)
        print(f"已保存时间步与VTK文件对应关系: {csv_path}")

        return self.mat_output_root


#
if __name__ == "__main__":
    # 初始化转换器
    converter = TimeStepDataConverter(case_dir="./cavity")
    
    # 删除之前的运行结果
    converter.clean_previous_results()

    # 仅转换为MAT格式
    converter.convert_to_mat()
