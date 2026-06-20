import scipy.io
import h5py
import numpy as np
import os

# 文件路径
file_path = '/home/ych/桌面/PINN_version3/all_boundaries.mat'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
    exit()

# 检查文件大小
file_size = os.path.getsize(file_path)
print(f"文件大小: {file_size} 字节")

# 首先尝试使用scipy.io.loadmat (适用于旧版MATLAB格式)
try:
    mat = scipy.io.loadmat(file_path)
    print("成功使用 scipy.io.loadmat 读取文件 (旧版MATLAB格式)")
    print("文件中的变量名：", [k for k in mat.keys() if not k.startswith('__')])
    
    # 显示变量信息
    for key in mat.keys():
        if not key.startswith('__'):
            data = mat[key]
            print(f"变量 {key} 的类型: {type(data)}")
            if hasattr(data, 'shape'):
                print(f"变量 {key} 的形状: {data.shape}")
except Exception as e:
    print(f"scipy.io.loadmat 读取失败: {e}")
    
    # 如果scipy失败，则尝试使用h5py (适用于MATLAB v7.3格式)
    try:
        mat = h5py.File(file_path, 'r')
        print("成功使用 h5py 读取文件 (MATLAB v7.3格式)")
        print("文件中的变量名：", list(mat.keys()))
        
        # 显示变量信息
        for key in mat.keys():
            if not key.startswith('#'):
                data = mat[key]
                if isinstance(data, h5py.Dataset):
                    print(f"变量 {key} 的形状：{data.shape}")
                    print(f"变量 {key} 的数据类型：{data.dtype}")
                else:
                    print(f"变量 {key} 的类型：{type(data)}")
        
        mat.close()
    except Exception as e2:
        print(f"h5py 读取也失败了: {e2}")
        print("文件可能已损坏或不是有效的MATLAB文件")