import os
import pyvista as pv
import numpy as np
import scipy.io as sio


def process_boundary_vtks(root_dir, output_file):
    """
    处理所有边界文件夹中的VTK文件，合并到一个mat文件
    root_dir: VTK文件根目录
    output_file: 输出的mat文件路径
    """
    # 定义边界文件夹
    boundaries = ["fixedWalls","frontAndBack","movingWall"]

    # 存储所有坐标数据和边界标签
    all_coords = []
    all_labels = []

    for boundary in boundaries:
        boundary_dir = os.path.join(root_dir, boundary)
        # 获取文件夹中的所有VTK文件并排序
        vtk_files = [f for f in os.listdir(boundary_dir) if f.endswith('.vtk')]


        # 按照文件名排序并选择第一个VTK文件进行处理
        vtk_files.sort()
        selected_vtk = vtk_files[0]
        vtk_path = os.path.join(boundary_dir, selected_vtk)

        # 读取VTK数据
        print(f"处理 {boundary} 边界的 {selected_vtk} 文件...")
        coords = vtk_to_data(vtk_path)
        
        # 添加坐标和标签
        all_coords.append(coords)
        labels = np.array([boundary] * coords.shape[0])
        all_labels.append(labels)

        # 合并所有数据

        combined_coords = np.vstack(all_coords)
        combined_labels = np.hstack(all_labels)
        
        # 创建新的数据结构
        flattened_data = {
            'coordinates': combined_coords,
            'boundary_labels': combined_labels
        }
        
        # 保存为mat文件
        sio.savemat(output_file, flattened_data)
        print(f"已生成合并的mat文件: {output_file}")
        
        # 打印每个键的前几行数据
        print("\n生成的MAT文件内容预览:")
        print_key_preview(flattened_data)



def print_key_preview(data, indent=0):
    """
    递归打印字典中每个键的前几行数据
    """
    spacing = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{spacing}{key}:")
                print_key_preview(value, indent + 1)
            elif isinstance(value, np.ndarray):
                print(f"{spacing}{key}: {value.shape} array")
                if value.size > 0:
                    if value.ndim == 1:
                        print(f"{spacing}  前5个元素: {value[:5]}")
                    elif value.ndim == 2:
                        rows = min(5, value.shape[0])
                        print(f"{spacing}  前{rows}行:")
                        for i in range(rows):
                            if value.shape[1] > 10:
                                print(f"{spacing}    [{i}]: {value[i][:10]}...")
                            else:
                                print(f"{spacing}    [{i}]: {value[i]}")
            else:
                print(f"{spacing}{key}: {value}")


def vtk_to_data(vtk_file_path):
    """
    读取VTK文件并返回其中的数据
    vtk_file_path: VTK文件路径
    返回: 包含点坐标、单元格信息、点数据和单元格数据的字典
    """
    # 读取VTK文件 ， 这里方法和前面文件中的一样 ，但是后面读取的坐标和数据有区别
    mesh = pv.read(vtk_file_path)

    # 提取点坐标，这个坐标和Cells不一样 ， BC都是面 ， Cells是体 ， 这里没有读取中心坐标 ，读取的就是网格点坐标
    points_np = mesh.points

    # 由网格点的坐标构建中心坐标，需要知道网格节点的连接关系 ， 这里没有读取 ， 也很难处理
    # 如果直接使用计算中心坐标 ，比如取中间值的方法的花 ， 只能对平面的边界 ，或者说直线边界有效 ，也很唐
    # 这里就默认使用点坐标 ， 但是这里注意一下 ， 同一个网格节点可以属于多个边界 ， 然后点的数量和 VTK 文件中的边界面的单元数量 或者说 mesh.faces 数量是不一致的

    boudary_data = (points_np)

    return boudary_data






if __name__ == "__main__":
    # 定义输入输出路径
    vtk_root_dir = "./cavity/VTK"
    mat_output_file = "./all_boundaries.mat"  # 输出单个mat文件

    # 创建输出目录（
    os.makedirs(os.path.dirname(mat_output_file), exist_ok=True)

    # 处理边界VTK文件
    process_boundary_vtks(vtk_root_dir, mat_output_file)
    print("所有边界文件处理完成")