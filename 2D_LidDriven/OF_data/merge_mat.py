import os
import scipy.io
import numpy as np


class MatFileMerger:
    """
    Mat文件合并器：用于合并多个.mat文件中的变量，按时间步顺序拼接    处理并保存为一个合并后的.mat文件，同时支持打印合并后数据的前几行。
    """

    def __init__(self, input_dir='./cavity/time_step_mat', output_file='./merged_time_steps.mat'):
        """
        初始化合并器

        参数:
            input_dir (str): 输入.mat文件所在目录，默认值为'./of_result/time_step_mat'
            output_file (str): 合并后文件的保存路径，默认值为'./merged_time_steps.mat'
        """
        self.input_dir = input_dir  # 输入目录
        self.output_file = output_file  # 输出文件路径
        self.all_vars = None  # 存储所有变量的时间序列数据
        self.time_steps = []  # 存储时间步名称（文件名）
        self.merged_data = None  # 合并后的数据集

    def _get_sorted_mat_files(self):
        """获取目录中所有.mat文件并按文件名数值排序"""
        # 筛选出所有.mat文件
        mat_files = [f for f in os.listdir(self.input_dir) if f.endswith('.mat')]
        # 按文件名（不含扩展名）的数值排序
        mat_files.sort(key=lambda x: float(os.path.splitext(x)[0]))
        return mat_files

    def load_and_merge(self):
        """加载并合并所有.mat文件中的变量"""
        # 遍历每个.mat文件处理
        mat_files = self._get_sorted_mat_files()
        for idx, file in enumerate(mat_files):
            file_path = os.path.join(self.input_dir, file)
            try:
                # 加载mat文件数据
                data = scipy.io.loadmat(file_path)

                # 提取有效变量（排除mat文件自带的特殊变量，就是前面三个key ，）
                var_names = [k for k in data.keys() if not k.startswith('__')]

                # 初始化变量存储结构（第一次处理时）
                if self.all_vars is None:
                    self.all_vars = {v: [] for v in var_names}
                # 为每个变量添加时间维度并存储
                for v in var_names:
                    arr = data[v]  # 原始形状假设为(1, 9200)
                    # 新增时间维度（变为(1, 1, 9200)），方便后续按时间维度拼接
                    self.all_vars[v].append(arr[np.newaxis, ...])

                self.time_steps.append(os.path.splitext(file)[0])  # 记录时间步
                print(f"已处理 {idx + 1}/{len(mat_files)}: {file}")

            except Exception as e:
                print(f"处理文件 {file} 出错：{e}")
                continue

        # 合并所有时间步数据
        if self.all_vars is not None and self.time_steps:
            self.merged_data = {}
            for v in self.all_vars:
                # 按时间维度（axis=0）拼接，形状变为(时间步数, 1, 9200)
                self.merged_data[v] = np.concatenate(self.all_vars[v], axis=0)
                print(f"变量 {v} 合并后形状：{self.merged_data[v].shape}")
            return True
        else:
            print("未找到有效数据进行合并")
            return False

    def save_merged_data(self):
        """保存合并后的数据到.mat文件"""
        if self.merged_data is None:
            print("没有可保存的合并数据，请先执行load_and_merge()")
            return
            
        try:
            # 保存合并后的数据到.mat文件
            scipy.io.savemat(self.output_file, self.merged_data)
            print(f"合并后的数据已保存至: {self.output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")

    def print_first_rows(self, var_name, rows=5):
        """
        打印指定变量合并后的前几行数据

        参数:
            var_name (str): 要打印的变量名
            rows (int): 要打印的行数，默认5行
        """
        if self.merged_data is None:
            print("没有合并数据，请先执行load_and_merge()")
            return

        if var_name not in self.merged_data:
            print(f"变量 {var_name} 不存在，可用变量：{list(self.merged_data.keys())}")
            return

        data = self.merged_data[var_name]
        print(f"\n变量 {var_name} 的前 {rows} 行数据：")
        # 打印前rows行（根据实际形状调整显示方式）
        print(data[:rows])

    def print_file_info(self, file_path):
        """
        打印单个.mat文件的信息，包括形状、大小和键
        
        参数:
            file_path (str): .mat文件的路径
        """
        try:
            # 加载mat文件数据
            data = scipy.io.loadmat(file_path)
            
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            # 打印文件基本信息
            print(f"\n文件: {os.path.basename(file_path)}")
            print(f"文件大小: {file_size} 字节 ({file_size / 1024:.2f} KB)")
            print(f"变量键名: {[k for k in data.keys() if not k.startswith('__')]}")
            
            # 打印各变量的形状
            print("变量形状:")
            for key, value in data.items():
                if not key.startswith('__'):  # 排除内置变量
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)} (非数组类型)")
                        
        except Exception as e:
            print(f"读取文件 {file_path} 信息时出错: {e}")


if __name__ == "__main__":
    # 用法
    merger = MatFileMerger()

    # 加载并合并数据
    if merger.load_and_merge():
        # 保存合并结果
        merger.save_merged_data()

        # 打印第一个变量的前5行数据

        first_var = next(iter(merger.merged_data.keys()))
        merger.print_first_rows(first_var, rows=5)
            
        # 打印合并后数据的整体信息
        print("\n=== 合并后数据信息 ===")
        print(f"时间步数量: {len(merger.time_steps)}")
        print(f"变量名称: {list(merger.merged_data.keys())}")
        print("各变量形状:")
        for var_name, var_data in merger.merged_data.items():
            print(f"  {var_name}: {var_data.shape}")
                
        # 显示单个原始文件的信息作为示例
        print("\n=== 单个原始文件信息示例 ===")
        mat_files = merger._get_sorted_mat_files()
        if mat_files:
            sample_file = os.path.join(merger.input_dir, mat_files[0]) # 打印排序后的第一个文件 ， 就是时间最开始的那一个
            merger.print_file_info(sample_file)
