#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:25:30 2024

@author: dayang
"""

## save one dictionary, search for others
from tqdm import tqdm
import itertools
import sys
import numpy as np
from scipy.spatial.distance import cityblock
from sklearn.neighbors import KDTree
import multiprocessing
import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
import matplotlib
matplotlib.use('TkAgg')
import subprocess
from decimal import Decimal, getcontext

getcontext().prec = 30

def trans2byte(bitstring):
    output = int(bitstring, 2)
    return output

def rebound_func(node):
    result = np.abs(node - 2*np.floor((node+1)/2))
    return result

def define_global(codebook_num, Uint_i, k, k_lossless):

    global side_length, K_lossless, dimension, Max, tree, Try, K
    n = codebook_num
    side_length = n + 1
    K_lossless = k_lossless
    K = k
    uint_i = Uint_i

    a = np.linspace(0, K - 1, K)
    b = 1 / (np.pi + a)

    Max_List_Path = f'./lib/uint{uint_i}_list.txt'
    Max = []
    with open(Max_List_Path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line[-1] == '\n':
                Max.append(int(line[:-1]))
            else:
                Max.append(int(line))

    dimension = [i for i in range(2, len(Max)+2)]

    R = 10  # 1/R : 步长 100

    Allval = np.arange(n)
    Allval = Allval.reshape((n,1))

    Cn = b/R  # 方向向量
    Cn = Cn.reshape((1,K))

    Target = np.random.uniform(0,1,K)
    Target = Target.reshape((1,K))
    i_target = 0

    # hyperfunction_1
    #Try = Allval @ Cn
    #Try, integral_part = np.modf(Try)  # 向量化

    # hyperfunction_2 rebound
    #Try = rebound_func(Try)

    # hyperfunction_3 grids
    Try = []
    n_line = int(np.floor(np.sqrt(n)))
    gap = np.linspace(0, 1, n_line)
    for i in gap:
        for j in gap:
            Try.append([i,j])
    Try = np.array(Try)


    x = Try[:, 0]
    y = Try[:, 1]
    #plt.scatter(x, y, color='red', marker='o', s=0.1)
    #plt.show()

    ## get a new tree for query
    tree = KDTree(Try, leaf_size=100)

    # Example arrays A and B
    # A = np.random.rand(1000, 5)
    Ba = 32
    Target = np.random.rand(Ba, K)

    return Max, dimension


def get_Try(tan_alpha, step_size, n, node_ld, rect_l):
    Allval = np.arange(n)
    Allval = Allval.reshape((n,1))

    Cn = np.array([rect_l/tan_alpha, rect_l])
    Cn = Cn/np.linalg.norm(Cn)
    Cn = Cn.reshape((1,2))*step_size

    Try_array = Allval @ Cn
    Try_array = Try_array % rect_l
    Try_array = Try_array + node_ld

    x = Try_array[:, 0]
    y = Try_array[:, 1]

    return Try_array


def read_uintn_list_from_bin(file_path, n):
    # 确定每个整数需要多少字节
    num_bytes = (n + 7) // 8  # 向上取整到完整的字节数

    with open(file_path, 'rb') as f:
        data_list = []
        while True:
            byte_chunk = f.read(num_bytes)  # 读取指定字节数
            if not byte_chunk:
                break
            number = 0
            for i in range(num_bytes):
                byte = int.from_bytes(byte_chunk[i:i + 1], byteorder='big')
                number = (number << 8) | byte
            data_list.append(number)
        return data_list


def Decode_Params(Save_CompressedResult_RootPath, codebook_num, Uint_i, k, k_lossless):

    define_global(codebook_num, Uint_i, k, k_lossless)
    print("#"*50,'\n',"Start Decoding",'\n',"#"*50)

    decode_params_list = []
    infor_path = Save_CompressedResult_RootPath + 'infor.bin'
    compressed_num_zero_path = Save_CompressedResult_RootPath + 'compressed_num_zero.bin'
    FloatInfor_path = Save_CompressedResult_RootPath + 'float.bin'
    Cha_path = Save_CompressedResult_RootPath + 'Cha.bin'
    Size_tar_path = Save_CompressedResult_RootPath + 'Size_tar.bin'
    if Uint_i % 8 != 0:
        Bit_zeros_path = Save_CompressedResult_RootPath + 'bit_zeros.bin'

    infor = np.fromfile(infor_path, dtype=np.uint8)
    compress_zeros = np.fromfile(compressed_num_zero_path, dtype=np.uint64)
    FloatInfor = np.fromfile(FloatInfor_path, dtype=np.float32).reshape([-1,2])
    Cha = np.fromfile(Cha_path, dtype=np.int32)
    Size_tar = np.fromfile(Size_tar_path, dtype=np.int32).reshape([-1,2])
    if Uint_i % 8 != 0:
        Bit_zeros = np.fromfile(Bit_zeros_path, dtype=np.uint8)

    w = infor[0]
    N = infor[1]
    uint_i = infor[3]

    k_lossless = infor[2]

    l = Max[dimension.index(N)] + 1
    back_step_1 = decompression_2(compress_zeros, l, N)

    if w != 0:
        back_step_1 = back_step_1[:-w]

    r"""
        decode layers' params
    """

    # 获取 root文件夹中所有文件名中不带"_"的文件名
    file_names = os.listdir(Save_CompressedResult_RootPath)
    if uint_i % 8 == 0:
        num_layers = len(file_names) - 5
    else:
        num_layers = len(file_names) - 6


    # 构建 layer_i 与 文件地址的dict
    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')]) # 第几层的参数
            shape_str = file_name[file_name.index('_')+1 : -4].split('_')
            shape = []
            for j in shape_str:
                shape.append(int(j))
            shape = np.array(shape) # 参数的shape
            dict_layer_files[i_layer] = [Save_CompressedResult_RootPath + file_name ,shape]
        except:
            # except 说明是“非layer params”文件
            continue

    print(f"一共 {num_layers} layers，成功获取{len(list(dict_layer_files.values()))}个文件")

    #index_count = {}  # 模型所有index中的最大值
    if num_layers == len(list(dict_layer_files.values())):
        # 遍历到每一个文件名和shape
        for i in tqdm(range(num_layers)):
            # layer_param_index decode 为 float params
            cha = Cha[i]
            size_tar = tuple(Size_tar[i])
            mean = FloatInfor[i][0]
            max_abs = FloatInfor[i][1]

            if mean==0 and max_abs==0 and cha==0 and size_tar==tuple([0,0]): # 当时压缩失败的参数直接load
                layer_path = dict_layer_files[i][0]  # 文件地址
                layer_shape = dict_layer_files[i][1]  # 恢复的shape
                param1 = np.fromfile(layer_path, dtype=np.float32)
                param1 = np.array(param1).reshape(layer_shape)
                decode_params_list.append(param1)

            else:
                layer_path = dict_layer_files[i][0]  # 文件地址
                layer_shape = dict_layer_files[i][1] # 恢复的shape
                if uint_i % 8 == 0:
                    layer_param_int = read_uintn_list_from_bin(layer_path, uint_i)
                else:
                    bit_save = np.fromfile(layer_path, dtype=np.uint8)
                    results8bits = "".join(format(i, f'0{8}b') for i in bit_save)[Bit_zeros[i]:]
                    with multiprocessing.Pool(processes=10) as pool:
                        layer_param_int = pool.starmap(trans2byte, [[results8bits[i:i + uint_i]] for i in range(0, len(results8bits), uint_i)])

                with multiprocessing.Pool(processes=10) as pool:
                    # 使用进程池的 starmap 方法并行处理列表中的每个元素和索引
                    layer_param_index = pool.starmap(back2node, [
                        [result, Max[dimension.index(k_lossless)] + 1, k_lossless]
                        for result in layer_param_int])

                layer_param_index = list(itertools.chain.from_iterable(layer_param_index))

                # layer_param_index 为 整数index结果
                if int(back_step_1[i]) != 0:
                    layer_param_index = layer_param_index[:-int(back_step_1[i])]

                '''''
                ########## 统计 index ##########
                for i in layer_param_int:
                    if i in list(index_count.keys()):
                        index_count[i] += 1
                    else:
                        index_count[i] = 0
                ########## 统计 index ##########
                '''''

                # param1 : 将整数index结果decode为新的float参数
                param1 = decompression_1d(np.array(layer_param_index).reshape([-1,1]), cha, size_tar, Try)
                param1 = restore_data(param1, mean, max_abs)
                param1 = np.array(param1).reshape(layer_shape)

                decode_params_list.append(param1)

    else:
        print("Something Wrong During Decoding!")

    print("#" * 50, '\n', "Finish Decoding", '\n', "#" * 50)
    return decode_params_list


def Save_Num_Zero(Num_Zero, root_dir, K_lossless, uint_i):
    save_name_infor = "infor.bin"
    save_path_infor = root_dir + save_name_infor
    save_name_compressed_num_zero = "compressed_num_zero.bin"
    save_path_compressed_num_zero = root_dir + save_name_compressed_num_zero

    _, N = check_comp(Num_Zero) # 可压最大倍数

    side_length_zero = Max[dimension.index(N)] + 1
    new_Num_Zero, w = add_zero(np.array(Num_Zero), N)
    Reshape_new_Num_Zero = np.array(new_Num_Zero).reshape(-1, N).tolist()
    
    with multiprocessing.Pool(processes=8) as pool:
        #使用进程池的 starmap 方法并行处理列表中的每个元素和索引
        compress_zeros = pool.starmap(process, [[Reshape_new_Num_Zero[index], index, side_length_zero, len(Reshape_new_Num_Zero)] for index in range(len(Reshape_new_Num_Zero))])

    infor = np.array([w, N, K_lossless, uint_i])
    
    np.array(compress_zeros).astype(np.uint64).tofile(save_path_compressed_num_zero)
    np.array(infor).astype(np.uint8).tofile(save_path_infor)
    print("ok")


'''''
def checkdic1(Target,diction=Try):
    ## Target [B,5]
    distances = np.linalg.norm(diction[:, np.newaxis, :] - Target, axis=2)
    closest_row_indices = np.argmin(distances, axis=0)
    return closest_row_indices[:, np.newaxis] ## (B,1)
'''''

def checkdic(Target,tree):
    ## Target [B,5]
    dist, closest_row_indices = tree.query(Target, k=1)
    return closest_row_indices ## (B,1)


# # Compute the Euclidean distances between each row in A and each row in B
# distances = np.linalg.norm(Try[:, np.newaxis, :] - Target, axis=2)
# # Find the index of the row in A with the minimum distance for each row in B
# closest_row_indices = np.argmin(distances, axis=0)
# print(closest_row_indices.shape)
    
'''''
#%% Integar compression
    
from decimal import Decimal, getcontext
from tqdm import tqdm
import sys

getcontext().prec = 1000

def f(x,l):
    x = np.array(x)
    y = x % l
    return y


def find_center_coordinate(l, point):

    for i in range(len(point)):
        point[i] = min(int(point[i]), l - 1) + 0.5

    return point


def add_zero(list, K):
    n = len(list)
    num_zero = K - n % K
    new_list = list.tolist() + [0]*num_zero

    return new_list, num_zero



def compute_S(node,l):
    K = len(node) #坐标维度
    L = [1] # S计算向量
    for t in range(K-1):
        num = L[0]
        L.insert(0, num*l)
    S = 0
    for i in range(len(node)):
        S += node[i] * L[i]
    return S


def back2node(S, l, K):
    l = int(l)
    L = [Decimal(l)]
    for t in range(K - 1):
        num = L[0]
        L.insert(0, num / l)
    b = L  # 方向向量

    R = Decimal(0)
    for i in range(len(b)):
        R += b[i] ** 2
    R = R.sqrt()

    b_1 = b
    for i in range(len(b_1)):
        b_1[i] = b_1[i] / R

    step_size = R / l

    B = b_1
    for i in range(len(B)):
        B[i] = B[i] * step_size

    start = [0] * K
    for i in range(len(start)):
        start[i] = start[i] + (step_size / 2) * b_1[i]

    step_of_S = start  # x = f(S * B + start, l)
    for i in range(len(step_of_S)):
        step_of_S[i] = step_of_S[i] + S * B[i]
    x = f(step_of_S, l)

    node = find_center_coordinate(l, x)
    node = np.array(node) - np.array([0.5] * K)
    result = node.tolist()
    return result


def decompression_2(Result, l, K_lossless):
    param1 = []
    num_zero = Result[-1]
    Ready2Compress = Result[:-1]

    for r in Ready2Compress:
        param1.append(back2node(r, l, K_lossless))

    back_flattened_array = []
    for w in param1:
        back_flattened_array += w
    if num_zero != 0:
        back_flattened_array = back_flattened_array[:-num_zero]

    return back_flattened_array



def compress_decompre_scale_deep_2(inputx): ## input array for testing and recover the same parameter, numpy array type

    K_lossless = 6

    flattened_array = inputx.flatten()

    if len(flattened_array) % K_lossless != 0: # 补0
        new_array, num_zero = add_zero(flattened_array, K_lossless)
    else:
        new_array = flattened_array
        num_zero = 0

    new_array = np.array(new_array).astype(np.int64)
    Ready2Compress = new_array.reshape(-1, K_lossless)
    Ready2Compress = Ready2Compress.tolist()

# =============================================================================
#     for i in tqdm(range(len(Ready2Compress))):
#         Ready2Compress[i] = compute_S(Ready2Compress[i], max(new_array)+1)
# =============================================================================
    Ready2Compress = [compute_S(x,max(new_array)+1) for x in Ready2Compress]
    # Ready2Compress is the result of compression
    Result = Ready2Compress + [num_zero]
# =============================================================================
#     print("\n")
#     print("len of Result:", len(Result))
#     print("Max of Result:", max(Result))
#     print("If max < sys.maxsize :", max(Result)<sys.maxsize)
#     print("Bytes of Result:", np.array(Result).nbytes)
#     print("Bytes of inputx:", np.array(inputx).nbytes)
# =============================================================================

    # decompression:
    param1 = decompression_2(Result, max(new_array)+1, K_lossless)
    param1 = np.array(param1).astype(np.float64).reshape(inputx.shape)  # output_idx.shape= (100,12)
# =============================================================================
#     print("if_same:", (param1 == inputx).all())
# =============================================================================

    return param1
    
'''''

def f(x,l):
    x = np.array(x)
    y = x % l
    return y


def find_center_coordinate(l, point):

    for i in range(len(point)):
        point[i] = min(int(point[i]), l - 1) + 0.5

    return point


def add_zero(list, K):
    n = len(list)
    if n % K != 0:
        num_zero = K - n % K
        new_list = list.tolist() + [0]*num_zero
    else:
        new_list = list
        num_zero = 0

    return new_list, num_zero


def compute_S(node,l): #输入node的array坐标, l:网格大小
    K = len(node) #坐标维度
    L = [1] # S计算向量
    for t in range(K-1):
        num = L[0]
        L.insert(0, num*l)

    '''''
    ##### 精度测试一 #####
    getcontext().prec = 1000
    T1_start = time.perf_counter()
    S = 0
    for i in range(len(node)):
        S += node[i] * L[i]
    T1_end = time.perf_counter()
    T1 = T1_end - T1_start

    ##### 精度测试二 #####
    getcontext().prec = 20
    T2_start = time.perf_counter()
    S = 0
    for i in range(len(node)):
        S += node[i] * L[i]
    T2_end = time.perf_counter()
    T2 = T2_end - T2_start
    print(f"调整精度后，速度可直接提升{round((T1-T2)/T1, 4)*100}%")
    t_increase = T2/T1
    '''''

    getcontext().prec = 30
    '''''
    ##### 计算S版本一 #######
    t1_start = time.perf_counter()
    S = 0
    for i in range(len(node)):
        S += node[i] * L[i]
    t1_end = time.perf_counter()
    t1 = t1_end - t1_start
    print(f"计算S  版本一耗时{t1_end - t1_start}")
    '''''

    ##### 计算S版本二 #######
    #t2_start = time.perf_counter()
    L2 = np.array(L, dtype=np.uint64).reshape((1,len(L)))
    node2 = np.array(node, dtype=np.uint64).reshape((len(node),1))
    S2 = (L2 @ node2)[0][0]
    #t2_end = time.perf_counter()
    #t2 = t2_end - t2_start
    #print(f"计算S  版本二耗时{'{:.15f}'.format(t2_end - t2_start)}")
    #print(f"计算S  是否有提速：{t2 < t1}")
    #print(f"计算S  速度提升{round((t1-t2)/t1, 4)*100}%")

    #Node = np.array(node, dtype=np.int64)
    #L = np.array(L, dtype=np.int64)
    #S = np.dot(L, Node)

    return S2

def back2node(S, l, K):
    l = int(l)
    L = [Decimal(l)]
    for t in range(K - 1):
        num = L[0]
        L.insert(0, num / l)
    b = L  # 方向向量

    R = Decimal(0)
    for i in range(len(b)):
        R += b[i] ** 2
    R = R.sqrt()

    b_1 = b
    for i in range(len(b_1)):
        b_1[i] = b_1[i] / R

    step_size = R / l

    B = b_1
    for i in range(len(B)):
        B[i] = B[i] * step_size

    start = [0] * K
    for i in range(len(start)):
        start[i] = start[i] + (step_size / 2) * b_1[i]

    step_of_S = start  # x = f(S * B + start, l)
    for i in range(len(step_of_S)):
        step_of_S[i] = step_of_S[i] + S * B[i]
    x = f(step_of_S, l)

    node = find_center_coordinate(l, x)
    node = np.array(node) - np.array([0.5] * K)
    result = node.tolist()
    return result

def check_comp(inn):
    m = max(inn)  #当前集合中的最大数字
    for n in range(len(Max)):
        if Max[n] < m :
            max_dimension = dimension[n-1] #当前理论上的最大倍数
            break
    if not 'max_dimension' in locals():
        max_dimension = 65

    if max_dimension >= len(inn): #说明能压
        return True, max_dimension
    else:
        return False, max_dimension

def compress_dis(param_dis):
    data_index = param_dis
    ## s2:
    s2, s3, s4 = [], [], []
    while len(data_index) != 0:
        #print("len of data_index: ", len(data_index))
        encode_inn = []
        for ele in range(len(data_index)): # 当前为data_index的第ele位
            if check_comp( encode_inn + [data_index[ele]] )[0]: # 说明可以加入encode_inn
                encode_inn.append(data_index[ele])
                if ele == len(data_index)-1: #已经到最后一个了，直接对目前的encode_inn压缩
                    s3.append(check_comp(encode_inn)[1]) #此时的encode_inn已经是将新元素进入之后的结果了，[1]是N维空间
                    l = Max[dimension.index(check_comp(encode_inn)[1])] + 1
                    S = compute_S(encode_inn, l)  # 找到的S
                    s2.append(S)
                    data_index = data_index[len(encode_inn):]  # 编码完成的data删除
                    break

            else: #不能加入的话就直接压缩或丢入s4
                if sum(encode_inn) >= 8: #可以压
                    s3.append(check_comp(encode_inn)[1])  # 此时的encode_inn已经是将新元素进入之后的结果了，[1]是N维空间
                    l = Max[dimension.index(check_comp(encode_inn)[1])] + 1
                    S = compute_S(encode_inn, l)  # 找到的S
                    s2.append(S)
                    if S > sys.maxsize:
                        print("ok")
                    data_index = data_index[len(encode_inn):]  # 编码完成的data删除
                    break

                else: #不可以压，补0后丢入s4
                    s4.append(encode_inn)
                    data_index = [0] + data_index[len(encode_inn):]
    return S_save, K_save


def decompression_2(Ready2Back, l, k_lossless):  # 将Result的list每个数字decode后输出为flatten list
    param1 = []

    for r in Ready2Back:
        p = back2node(r, l, k_lossless)
        param1.append(p)

    back_flattened_array = []
    for w in param1:
        back_flattened_array += w

    #new_Num_Zero = np.fromfile('./compressed_result/test/Compressed_Dir/new_Num_Zero.bin', dtype=np.int64)

    return back_flattened_array

def process(node, index, side_length_set, total_num): # 传入 元素的index和元素本身
    #now = 100*(index/total_num)
    #print(f"进度：{now:.2f}%")
    compute_S_result = compute_S(node, side_length_set)
    #back = back2node(compute_S_result, side_length_set, 27)
    #print(node==back)
    return compute_S_result

def process_dynamic(node, l):
    return compute_S(node, l+1)


def compress_decompre_scale_deep_2(inputx): ## input array for testing and recover the same parameter, numpy array type

    flattened_array = inputx.flatten()

    if len(flattened_array) % K_lossless != 0: # 补0
        new_array, num_zero = add_zero(flattened_array, K_lossless)
    else:
        new_array = flattened_array
        num_zero = 0

    new_array = np.array(new_array).astype(np.uint64)
    Ready2Compress = new_array.reshape(-1, K_lossless)
    Ready2Compress = Ready2Compress.tolist()
    #side_length = 1000+1

# =============================================================================
#     for i in tqdm(range(len(Ready2Compress))):
#         Ready2Compress[i] = compute_S(Ready2Compress[i], max(new_array)+1)
# =============================================================================

    #Ready2Compress = [compute_S(x,side_length) for x in tqdm(Ready2Compress)]



    with multiprocessing.Pool(processes=8) as pool:
        # 使用进程池的 starmap 方法并行处理列表中的每个元素和索引
        Result = pool.starmap(process, [[Ready2Compress[index], index, side_length, len(Ready2Compress)] for index in range(len(Ready2Compress))])



# =============================================================================
#     print("\n")
#     print("len of Result:", len(Result))
#     print("Max of Result:", max(Result))
#     print("If max < sys.maxsize :", max(Result)<sys.maxsize)
#     print("Bytes of Result:", np.array(Result).nbytes)
#     print("Bytes of inputx:", np.array(inputx).nbytes)
# =============================================================================

    # decompression:
    #param1 = []
    #Result_try = [num_zero] + Result
    #param1 = decompression_2(Result_try, max(new_array)+1, K_lossless)
    #param1 = np.array(param1).astype(np.float64).reshape(inputx.shape)  # output_idx.shape= (100,12)
# =============================================================================
    #print("if_same:", (param1 == inputx).all())
# =============================================================================
    return Result, num_zero


#%%
## Deal with matrix size of [500,100]

import time

def batch_compression(TargetX, Tree, Ba=128):
    K = 2
    bat,cha = TargetX.shape
    newch = int(np.ceil(cha/K))
    newba = int(np.ceil(bat/Ba))

    pad_size = newch*K - cha
    if pad_size > 0:
      TargetX = np.pad(TargetX, ((0, 0), (0, pad_size)), constant_values=0.5)

    output_idx = np.zeros((bat,newch))
    for i in range(newba):
        for j in range(newch):
            hi = i*Ba
            wi = j*K
            tmp_res = checkdic(TargetX[hi:hi+Ba,wi:wi+K],Tree)
            # print(tmp_res.shape)
            output_idx[hi:hi+Ba,j] = tmp_res.reshape(-1)

    return output_idx, (bat,cha)



def decompression(output_idx,size_tar,diction):
    K = 2
    bat,ch = output_idx.shape
    outputx = np.zeros((bat,ch*K))

    for c in range(ch):
        outputx[:,c*K:c*K+K] = diction[output_idx[:,c].astype(int)]

    return outputx[:,:size_tar[1]]

def batch_compression_1d(TargetX, tree, Ba):

    """
    cha : The length of the original params
    results.shape[0]*2 - cha : pad_size
    """

    K = 2
    cha = len(TargetX)
    newch = int(np.ceil(cha/2))

    pad_size = newch*K - cha
    if pad_size > 0:
      TargetX = np.pad(TargetX, (0, pad_size), constant_values=0.5)

    TargetX = TargetX.reshape(-1, 2)
    results, sizere =  batch_compression(TargetX, Tree=tree, Ba=128)

    return results, cha, sizere


def decompression_1d(output_idx, cha, sizere, Try):

    outputx = decompression(output_idx,sizere,diction=Try)
    # print(outputx.shape)
    outputx = outputx.flatten()[:cha]
    return outputx

def scale_data(data):
    # Step 1: Subtract the mean to make the average zero
    mean = np.mean(data)
    data_centered = data - mean
    # Step 2: Scale the data to be in the range [-0.5, 0.5]
    max_abs = np.max(np.abs(data_centered))
    scaled_data = data_centered / (2 * max_abs)  # Scale to [-0.5, 0.5]
    # Step 3: Shift the scaled data to have an average of 0.5
    scaled_data = scaled_data + 0.5

    return scaled_data, mean, max_abs


def restore_data(scaled_data, mean, max_abs):
    # Step 1: Shift the scaled data to remove the average of 0.5
    shifted_data = scaled_data - 0.5
    # Step 2: Restore the original scale
    restored_data = shifted_data * (2 * max_abs)
    # Step 3: Add back the mean
    original_data = restored_data + mean

    return original_data

def compress_decom_v2(inputx): ## most easy form to deal with all shapes, only need to deal with 1d
    ## scale the data to 0 - 1
# =============================================================================
#     max_value = inputx.max()
#     min_value = inputx.min()
#     #min_value, max_value = np.percentile(inputx, [.5, 99.5]) #[2.5, 97.5]
#     inputx = (inputx - min_value) / (max_value - min_value)
# =============================================================================
    inputx, mean, max_abs = scale_data(inputx)

    ori_shape = inputx.shape
    inputx = inputx.flatten()
    output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree, Ba=128)
    param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    param1 = param1.reshape(ori_shape)

    ## scale value back
# =============================================================================
#     param1 = param1 * (max_value - min_value) + min_value
# =============================================================================
    # Restore the scaled data
    param1 = restore_data(param1, mean, max_abs)
    return param1

from scipy import sparse

def small_rect(inputx, rect_l):  # 以 inputx的质心为中心，rect_l为边长的正方形中的点作为原参数
    nodes = inputx.reshape(-1,2)
    x_node = nodes[:,0]
    y_node = nodes[:,1]
    x_center = np.mean(x_node) # 质心的x
    y_center = np.mean(y_node) # 质心的y
    node_center = np.array([x_center, y_center]) # 质心
    x1, y1 = x_center - rect_l/2, y_center + rect_l/2
    x2, y2 = x_center + rect_l/2, y_center + rect_l/2
    x3, y3 = x_center + rect_l/2, y_center - rect_l/2
    x4, y4 = x_center - rect_l/2, y_center - rect_l/2
    plt.figure(figsize=(6, 6))
    #plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1])  # 画内矩形的边界


    # 画圆
    radius = rect_l/2
    x_center, y_center = node_center[0], node_center[1]
    theta = np.linspace(0, 2 * np.pi, 100)
    x = radius * np.cos(theta) + x_center
    y = radius * np.sin(theta) + y_center
    plt.plot(x, y, color='black')

    plt.scatter(x_node, y_node, color='red', marker='o', s=5)
    plt.scatter(x_center, y_center, color='black', marker='*', s=50)
    plt.xlim(np.min(x_node), np.max(x_node))
    plt.ylim(np.min(y_node), np.max(y_node))
    #plt.show()

    inputx = inputx.reshape(-1,2)
    inputx_rect = []
    for node in tqdm(inputx):
        if x4 <= node[0] <= x3 and y4 <= node[1] <= y1:
            inputx_rect.append(node)

    return np.array(inputx_rect), [x4, y4]

def find_inner_outer(inputx, rect_l1):
    K = 2
    cha = len(inputx)
    newch = int(np.ceil(cha / K))

    pad_size = newch * K - cha
    if pad_size > 0:
        inputx = np.pad(inputx, (0, pad_size), constant_values=np.mean(inputx))

    nodes = inputx.reshape(-1,2)
    x_node = nodes[:,0]
    y_node = nodes[:,1]
    x_center = np.mean(x_node) # 质心的x
    y_center = np.mean(y_node) # 质心的y
    center_node = np.array([x_center, y_center]) # 质心

    # 定义矩形的左下角坐标 (x, y)，以及矩形的宽度和高度
    x_ld, y_ld = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2
    x_lu, y_lu = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_ru, y_ru = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_rd, y_rd = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2


    t2_start = time.perf_counter()

    dis_matrix = np.linalg.norm(np.abs(nodes - center_node), axis=1).astype(np.float64)

    farthest_dis_matrix = np.max(dis_matrix)


    farthest_node_matrix = nodes[np.where(dis_matrix == farthest_dis_matrix)[0][0]]


    # 判断node是否在inner内部
    check_inner = (nodes[:,0] >= x_lu) & (nodes[:,0] <= x_ru) & (nodes[:,1] >= y_ld) & (nodes[:,1] <= y_lu)
    inner_nodes_index = np.where(check_inner == True)[0]
    outer_nodes_index = np.where(check_inner == False)[0]

    inputx_rec_inner_matrix = nodes[inner_nodes_index]
    inputx_rec_outer_matrix = nodes[outer_nodes_index]

    t2_end = time.perf_counter()
    t2 = t2_end - t2_start

    return nodes, inputx_rec_inner_matrix, inputx_rec_outer_matrix, center_node, farthest_dis_matrix, farthest_node_matrix, pad_size


def scale_rect_outer(inputx_rect_outer, center_node, factor):

    scaled_inputx_outer = []
    for i in inputx_rect_outer: # i is C
        OC = i - center_node
        OB = OC*factor
        B = center_node + OB
        scaled_inputx_outer.append(B)
    scaled_rect_outer = np.array(scaled_inputx_outer)

    return scaled_rect_outer


def rescale_rect_outer(scaled_rect_outer, factor, center_node):
    rescaled_rect_outer = []
    for i in scaled_rect_outer: # i is B
        OB = i - center_node
        OC = OB/factor
        C = OC + center_node
        rescaled_rect_outer.append(C)
    rescaled_rect_outer = np.array(rescaled_rect_outer)

    return rescaled_rect_outer


def define_nodes_on_circle(center_node, r, n):
    cx, cy = center_node[0], center_node[1]  # 圆心坐标

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)

    return x, y


def find_factors_closest_pair(A):
    for x in range(int(math.sqrt(A)), 0, -1):
        if A % x == 0:
            y = A // x
            return x, y


def test_node_class(outer_node, center_node, dis_list):


    dis = np.linalg.norm(np.abs(outer_node-center_node)).astype(np.float64)

    for class_dis in dis_list:
        if dis <= class_dis:
            node_class = np.where(dis_list==class_dis)[0][0]+1
            break


    return node_class


def decompress_by_KDTree(Try_rect_inner, tree_rect_inner, B):
    output_idx_node, cha_inner, size_tar_inner = batch_compression_1d(B, tree_rect_inner, 128)

    B_star = decompression_1d(output_idx_node, cha_inner, size_tar_inner, Try_rect_inner)
    index = output_idx_node

    return B_star, index


def test_ClassLoss(Try_rect_inner, tree_rect_inner, inputx_rect_outer, rect_l1, each_dis, center_node, farthest_node, num_class, inner_MAE_loss):

    farthest_dis = np.linalg.norm(np.abs(farthest_node-center_node)).astype(np.float64)
    start_class = num_class
    dis_list = np.array([(i+1)*each_dis+(rect_l1/2) for i in range(num_class)], dtype=np.float64)
    dis_list[-1] = farthest_dis
    factor_list = (rect_l1/2)/dis_list

    center_node_tile = np.tile(center_node, (inputx_rect_outer.shape[0], 1))
    dis_tile = np.linalg.norm(inputx_rect_outer - center_node_tile, axis=1).reshape(-1,1)
    node_class_1 = np.argmax(dis_tile <= np.tile(dis_list, (dis_tile.shape[0],1)), axis=1) + 1
    node_factor_1 = factor_list[node_class_1 - 1].reshape(-1,1)
    OC_1 = inputx_rect_outer - np.tile(center_node, (inputx_rect_outer.shape[0],1))
    OB_1 = OC_1 * node_factor_1
    B_1 = OB_1 + np.tile(center_node, (inputx_rect_outer.shape[0],1))
    B_star_1, index_1 = decompress_by_KDTree(Try_rect_inner, tree_rect_inner, B_1.flatten())
    B_star_1 = B_star_1.reshape(-1,2)
    OB_star_1 = B_star_1 - np.tile(center_node, (inputx_rect_outer.shape[0],1))
    OC_star_1 = OB_star_1 / node_factor_1
    C_star_1 = OC_star_1 + np.tile(center_node, (inputx_rect_outer.shape[0],1))

    tensor_loss_1 = np.append(inner_MAE_loss, np.abs(C_star_1.flatten() - inputx_rect_outer.flatten()))
    MAE_tensor_loss_1 = np.mean(tensor_loss_1)


    return MAE_tensor_loss_1, start_class



def scale_rect_outer_class(Try_rect_inner, tree_rect_inner, node, center_node, farthest_dis, best_class, rect_l1):
    each_dis = (farthest_dis - (rect_l1/2)) / best_class
    dis_list = np.linspace(1, best_class, best_class, endpoint=True) * each_dis + (rect_l1/2)
    dis_list[-1] = farthest_dis
    factor_list = (rect_l1/2) / dis_list
    node_class = test_node_class(node, center_node, dis_list)
    node_factor = factor_list[node_class - 1]
    OC = node - center_node
    OB = OC * node_factor
    B = OB + center_node
    B_star, index = decompress_by_KDTree(Try_rect_inner, tree_rect_inner, B)
    OB_star = B_star - center_node
    OC_star = OB_star / node_factor
    C_star = OC_star + center_node
    loss_B = np.mean(np.abs(B-B_star))
    loss_C = np.mean(np.abs(node-C_star))
    loss_C_comp_1 = np.mean(np.abs(B-B_star)/node_factor)
    loss_C_comp_2 = np.mean( np.abs(B - B_star) ) * (2*dis_list[node_class - 1] / rect_l1)

    return C_star, B_star, B, node_class, index



def multi_compression_decompression_1d(node_inner, tree_rect_inner, Try_rect_inner):
    """
        Input:
        node_inner
        tree_rect_inner
        Try_rect_inner


        Return:
        output_idx_node : 输入node的index
        param_inner ： 还原后的lossy node
        inner_MAE_loss : MAE_loss

    """

    output_idx_node, cha_inner, size_tar_inner = batch_compression_1d(node_inner, tree_rect_inner, 128)

    # param_inner : 还原后的node
    param_inner = decompression_1d(output_idx_node, cha_inner, size_tar_inner, Try_rect_inner)
    inner_MAE_loss = np.abs(node_inner - param_inner)


    return [output_idx_node, param_inner, inner_MAE_loss]


def get_max_threads():

    cpu_cores = os.cpu_count()

    try:
        max_processes = int(subprocess.check_output(['ulimit', '-u']).strip())
    except Exception as e:
        print(f"Error fetching max processes: {e}")
        max_processes = cpu_cores * 10  # 默认值

    max_threads = min(cpu_cores * 2, max_processes)

    return max_threads



def decompress_v3(index_array,
                  num_inner_index_i,
                  class_list_i,
                  if_padding_i,
                  center_node_i,
                  farthest_node_i,
                  rect_l1,
                  num_inner_list
                  ):

    n = num_inner_list[num_inner_index_i]

    d_inner = rect_l1 / (np.ceil(np.sqrt(n)))
    l1 = np.sqrt(rect_l1 ** 2 + d_inner ** 2)
    tan_alpha = np.ceil(np.sqrt(n))
    sin_alpha = rect_l1 / l1
    step_size_inner = (rect_l1 / sin_alpha) / np.ceil(np.sqrt(n))


    Try_rect_inner = np.array([center_node_i])

    Try_rect_inner = np.append(Try_rect_inner, get_Try(tan_alpha, step_size=step_size_inner,
                                                       n=n,
                                                       node_ld=center_node_i - np.array([rect_l1 / 2, rect_l1 / 2]),
                                                       rect_l=rect_l1)).reshape([-1, 2])
    tree_rect_inner = KDTree(Try_rect_inner, leaf_size=100)



    if class_list_i != 0: # K != 0
        farthest_dis = np.linalg.norm(center_node_i-farthest_node_i).astype(np.float64)
        index_class = np.floor(index_array/(n+1)) # C
        each_dis = (farthest_dis - (rect_l1/2))/class_list_i
        dis_list = rect_l1/2 + index_class * each_dis
        factor = ((rect_l1/2) / dis_list).reshape(-1,1)
        index_array_i = index_array - index_class*(n+1)
        B_star = decompression_1d(index_array_i.reshape(-1,1), index_array_i.shape[0]*2, tuple((index_array_i.shape[0],2)), Try_rect_inner)
        B_star = B_star.reshape(-1,2)
        center_node = np.tile(np.array([center_node_i]), (B_star.shape[0],1))
        OB_star = B_star - center_node
        OC_star = OB_star/factor
        C_star = OC_star + center_node
        param1 = C_star.flatten()
        if if_padding_i != 0:
            param1 = param1[:-int(if_padding_i)]

    else: # K == 0
        index_array_i = index_array
        C_star = decompression_1d(index_array_i.reshape(-1, 1), index_array_i.shape[0] * 2, tuple((index_array_i.shape[0], 2)),
                         Try_rect_inner)
        param1 = C_star.flatten()
        if if_padding_i != 0:
            param1 = param1[:-int(if_padding_i)]

    return param1




def compress_decom_v3(inputx,
                      num_ori,
                      rect_l1,
                      num_inner_list,
                      class_max,
                      loss_max,
                      loss_shredhold):
    """
    inputx : ori_param of i-layer
    num_ori : the i of i-layer
    rect_l1 : inner side
    num_inner_list : 可选的num_inner的list
    """

    origin_inputx = inputx

    plt.clf()
    ori_shape = inputx.shape
    inputx = inputx.flatten()


    for num_inner in num_inner_list:

        d_inner = rect_l1/(np.ceil(np.sqrt(num_inner)))
        l1 = np.sqrt(rect_l1**2 + d_inner**2)
        tan_alpha = np.ceil(np.sqrt(num_inner))
        sin_alpha = rect_l1/l1
        step_size_inner = (rect_l1/sin_alpha)/np.ceil(np.sqrt(num_inner))

        padding_nodes, inputx_rect_inner, inputx_rect_outer, center_node, farthest_dis, farthest_node, pad_size = find_inner_outer(inputx, rect_l1)

        x_ld_inner, y_ld_inner = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2
        x_lu_inner, y_lu_inner = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2
        x_ru_inner, y_ru_inner = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2
        x_rd_inner, y_rd_inner = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2
        width_inner, height_inner = rect_l1, rect_l1



        """ Create the KDTree """
        Try_rect_inner = np.array([center_node])

        Try_rect_inner = np.append(Try_rect_inner, get_Try(tan_alpha, step_size=step_size_inner,
                                                           n=num_inner,
                                                           node_ld=center_node - np.array([rect_l1 / 2, rect_l1 / 2]),
                                                           rect_l=rect_l1)).reshape([-1, 2])
        plt.scatter(Try_rect_inner[:, 0], Try_rect_inner[:, 1], color='black', marker='o', s=10)
        tree_rect_inner = KDTree(Try_rect_inner, leaf_size=100)


        inner_MAE_loss = []
        if inputx_rect_inner.shape[0] != 0:
            output_idx_node, cha_inner, size_tar_inner = batch_compression_1d(inputx_rect_inner.flatten(), tree_rect_inner, 128)
            param_inner_list = decompression_1d(output_idx_node, cha_inner, size_tar_inner, Try_rect_inner)
            inner_MAE_loss = np.abs(param_inner_list - inputx_rect_inner.flatten())



        """ Test the best class """
        if inputx_rect_outer.shape[0] == 0:
            best_class = 0
            best_MAE_tensor_loss = np.mean(inner_MAE_loss)

        else:
            best_MAE_tensor_loss = 100
            best_class = None

            for num_class in range(1,class_max+1):
                each_dis = (farthest_dis-(rect_l1/2))/num_class

                MAE_tensor_loss, start_class = test_ClassLoss(Try_rect_inner, tree_rect_inner, inputx_rect_outer, rect_l1, each_dis, center_node, farthest_node, num_class, inner_MAE_loss) # tensor_loss : 类别为 num_class 时的loss

                if MAE_tensor_loss <= best_MAE_tensor_loss:
                    best_MAE_tensor_loss = MAE_tensor_loss
                    best_class = num_class

                if MAE_tensor_loss <= loss_shredhold:
                    best_class = num_class
                    best_MAE_tensor_loss = MAE_tensor_loss
                    break

                if num_class == class_max:
                    each_dis = np.array((farthest_dis - (rect_l1/2)) / best_class)
                    MAE_tensor_loss, start_class = test_ClassLoss(Try_rect_inner, tree_rect_inner, inputx_rect_outer, rect_l1, each_dis, center_node, farthest_node, best_class, inner_MAE_loss)
                    best_MAE_tensor_loss = MAE_tensor_loss


        if best_MAE_tensor_loss > loss_max and num_inner_list.index(num_inner) != len(num_inner_list) - 1:
            continue
        else:
            break



    if best_class == 0:

        output_idx_node_1, cha_inner_1, size_tar_inner_1 = batch_compression_1d(padding_nodes.flatten(), tree_rect_inner, 128)
        new_param_1 = decompression_1d(output_idx_node_1.reshape(-1, 1), cha_inner_1, size_tar_inner_1, Try_rect_inner)
        output_idx_node_1 = output_idx_node_1.flatten()

    else:
        check_inner_1 = ((padding_nodes[:, 0] >= x_lu_inner) & (padding_nodes[:, 0] <= x_ru_inner) &
                       (padding_nodes[:, 1] >= y_ld_inner) & (padding_nodes[:, 1] <= y_lu_inner))
        check_inner_1_index = np.where(check_inner_1==True)[0]


        center_node_tile = np.tile(center_node, (padding_nodes.shape[0], 1))
        each_dis = (farthest_dis - (rect_l1/2)) / best_class
        dis_list = np.linspace(0, best_class, best_class+1, endpoint=True) * each_dis + (rect_l1/2)
        dis_list[-1] = farthest_dis

        factor_list = (rect_l1 / 2) / dis_list

        dis_tile = np.linalg.norm(padding_nodes - center_node_tile, axis=1).astype(np.float64).reshape(-1,1)
        check_class_1 = np.argmax(dis_tile <= np.tile(dis_list, (dis_tile.shape[0],1)), axis=1)
        check_class_1[check_inner_1_index] = 0

        node_factor_1 = factor_list[check_class_1].reshape(-1, 1)
        OC_1 = padding_nodes - np.tile(center_node, (padding_nodes.shape[0], 1))
        OB_1 = OC_1 * node_factor_1
        B_1 = OB_1 + np.tile(center_node, (padding_nodes.shape[0], 1))
        B_star_1, index_1 = decompress_by_KDTree(Try_rect_inner, tree_rect_inner, B_1.flatten())
        B_star_1 = B_star_1.reshape(-1, 2)
        OB_star_1 = B_star_1 - np.tile(center_node, (padding_nodes.shape[0], 1))
        OC_star_1 = OB_star_1 / node_factor_1
        C_star_1 = OC_star_1 + np.tile(center_node, (padding_nodes.shape[0], 1))
        new_param_1 = C_star_1.flatten()
        output_idx_node_1 = index_1.flatten() + (num_inner + 1)*check_class_1
        output_idx_node_1 = output_idx_node_1.flatten()


    if pad_size != 0:
        new_param_1 = new_param_1[:-pad_size]


    new_param_1 = new_param_1.reshape(ori_shape)
    best_MAE_tensor_loss_list = np.abs(new_param_1.flatten() - origin_inputx.flatten())



    return (best_MAE_tensor_loss_list,
            best_class,
            new_param_1,
            output_idx_node_1,
            num_inner,
            inputx_rect_inner.shape[0],
            inputx_rect_outer.shape[0],
            pad_size,
            num_inner_list.index(num_inner),
            center_node,
            farthest_node)
    


def get_new_params(node, center_node, rect_l, tree_circle_inner, Try_circle_inner, l_loop,n_c_outer, farest_dis,
                                                         tree_circle_outer_scaled, Try_circle_outer_scaled):

    if np.linalg.norm(np.abs(node - center_node)) <= rect_l:  # 直接用inner中的点替换
        output_idx_node, cha_inner, size_tar_inner = batch_compression_1d(node, tree_circle_inner, 128)
        new_param = decompression_1d(output_idx_node, cha_inner, size_tar_inner, Try_circle_inner)

    else:  # scale后再rescale回去
        scaled_circle_outer, factor = scale_circle_outer(np.array([node]), center_node,
                                                         rect_l + l_loop * (1 / n_c_outer), l_loop, farest_dis)
        output_idx_outer_scaled, cha_outer_scaled, size_tar_outer_scaled = batch_compression_1d(
            scaled_circle_outer.flatten(), tree_circle_outer_scaled, 128)

        # param1 : 将整数index结果decode为新的float参数
        param1_outer_scaled = decompression_1d(output_idx_outer_scaled, cha_outer_scaled, size_tar_outer_scaled,
                                               Try_circle_outer_scaled)

        # 再rescale回去
        new_param = rescale_circle_outer(param1_outer_scaled.reshape(-1, 2), factor, center_node,
                                                     rect_l, l_loop)


    return new_param


def batch_compression_3d(TargetX, tree, Ba): ## reshape to 2d, and then reshape back to 1d
    # Reshape the 3D array to 2D
    size3d = TargetX.shape
    TargetX = TargetX.reshape(-1, TargetX.shape[-1])

    ## compression
    output_idx, size2d = batch_compression(TargetX, Tree=tree, Ba=128)

    return output_idx, size3d, size2d


def decompression_3d(output_idx,size3d,size2d,Try):
    outputx = decompression(output_idx,size2d,diction=Try)
    outputx = outputx.reshape(size3d)
    return outputx


def compress_decompre(inputx): ## input array for testing and recover the same parameter, numpy array type
    """
    input: inputx, 1d,2d,3d array
    output:, param1, array with same shape
    """
    ## scale the data to 0 - 1
    max_value = inputx.max()
    min_value = inputx.min()
    inputx = (inputx - min_value) / (max_value - min_value)

    #print(inputx.ndim)
    if inputx.ndim == 1:
        output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree, Ba=128)
        #print(output_idx.shape)
        param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    elif inputx.ndim == 2:
        output_idx,size_tar = batch_compression(inputx, Tree=tree, Ba=128)
        #print(output_idx.shape)
        param1 = decompression(output_idx,size_tar,diction=Try)
    elif inputx.ndim == 3:
        output_idx, size3d, size2d = batch_compression_3d(inputx, Tree=tree, Ba=128)
        #print(output_idx.shape)
        param1 = decompression_3d(output_idx,size3d,size2d,diction=Try)
    elif inputx.ndim == 4:
        b,c,h,w = inputx.shape
        inputx3 = inputx.reshape((b,c,-1))
        ## three dims
        output_idx, size3d, size2d = batch_compression_3d(inputx3, Tree=tree, Ba=128)
        #output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        #output_idx = output_idx.T
        param11 = decompression_3d(output_idx,size3d,size2d,diction=Try)
        param1 = param11.reshape((b,c,h,w))
    else:
        param1 = 0
        print('error dimension!')
        
    ## scale value back
    param1 = param1 * (max_value - min_value) + min_value
    return param1

def compress_decompre_scale_deep(inputx): ## input array for testing and recover the same parameter, numpy array type
    """
    input: inputx, 1d,2d,3d array
    output:, param1, array with same shape
    """
    ## scale the data to 0 - 1, change the idx value mapping since it's very big
    max_value = inputx.max() # +10000
    min_value = inputx.min() # -10000
    inputx = (inputx - min_value) / (max_value - min_value)

    print(inputx.ndim)
    if inputx.ndim == 1:
        output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree_l2, Ba=128)
        print(output_idx.shape)
        param1 = decompression_1d(output_idx,cha,size_tar,diction=Try_l2)
    if inputx.ndim == 2:
        output_idx,size_tar = batch_compression(inputx, Tree=tree_l2, Ba=128)
        print(output_idx.shape)
        param1 = decompression(output_idx,size_tar,diction=Try_l2)
    if inputx.ndim == 3:
        output_idx, size3d, size2d = batch_compression_3d(inputx, Tree=tree_l2, Ba=128)
        print(output_idx.shape)
        param1 = decompression_3d(output_idx,size3d,size2d,diction=Try_l2)

    ## scale value back
    param1 = param1 * (max_value - min_value) + min_value
    return param1

def compress_decompre_l2(inputx): ## input array for testing and recover the same parameter, numpy array type
    """
    input: inputx, 1d,2d,3d array
    output:, param1, array with same shape
    """
    ## scale the data to 0 - 1
    max_value = inputx.max()
    min_value = inputx.min()
    inputx = (inputx - min_value) / (max_value - min_value)

    ## scale to 0.2-0.8 to avoid jump
    gap = 0.1
    inputx = gap + inputx * (1 - 2*gap) ## scale from 0-1

    if inputx.ndim == 1:
        output_idx, cha, size_tar = batch_compression_1d(inputx, Tree=tree, Ba=128)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression, Transpose, to compression on another dimension
        output_idx = output_idx.T
        param1 = decompression_1d(output_idx,cha,size_tar,diction=Try)
    elif inputx.ndim == 2:
        output_idx,size_tar = batch_compression(inputx, Tree=tree, Ba=128)
        print(output_idx)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        output_idx = output_idx.T
        print(output_idx)
        param1 = decompression(output_idx,size_tar,diction=Try)
    elif inputx.ndim == 3:
        output_idx, size3d, size2d = batch_compression_3d(inputx, Tree=tree, Ba=128)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        output_idx = output_idx.T
        param1 = decompression_3d(output_idx,size3d,size2d,diction=Try)
    elif inputx.ndim == 4:
        b,c,h,w = inputx.shape
        inputx3 = inputx.reshape((b,c,-1))
        ## three dims
        output_idx, size3d, size2d = batch_compression_3d(inputx3, Tree=tree, Ba=128)
        output_idx = compress_decompre_scale_deep_2(output_idx.T) ## level 2 compression
        output_idx = output_idx.T
        param11 = decompression_3d(output_idx,size3d,size2d,diction=Try)
        param1 = param11.reshape((b,c,h,w))
    else:
        param1 = 0
        print('error dimension!')
    param1 = (param1 - gap) / (1 - 2*gap) ## from 0.2-0.8 to 0-1
    param1 = param1 * (max_value - min_value) + min_value
    return param1

# bat =
# TargetX = np.random.rand(4096, 500)

# TargetX = np.random.rand(999, 106)


# start_time = time.time()
# output_idx,size_tar = batch_compression(TargetX, Tree=tree, Ba=512)
# print(output_idx.shape)

# end_time = time.time()
# # Calculate the time elapsed
# elapsed_time = end_time - start_time

# # Report the time consumption
# print(f"Time consumption: {elapsed_time:.4f} seconds")
# ## output array is (500,np.ceil(100//5))

# ## Decomnpression step

# outputx = decompression(output_idx,size_tar,diction=Try)
# print('Reconstructed shape:',outputx.shape)

# end_time2 = time.time()
# elapsed_time2 = end_time2 - end_time
# print(f"Reconstruction Time consumption: {elapsed_time2:.4f} seconds")


# ## show reconstruction results
# import matplotlib.pyplot as plt

# # Assuming you have already defined xx, decimal, closest_row_indices, Try, and Target

# # Create a 4x4 grid of subplots
# fig, axs = plt.subplots(4, 4, figsize=(12, 12))

# # Flatten the axs array to make it easier to iterate
# axs_flat = axs.flatten()

# # Plot each subplot
# for i in range(16):
#     idx = i+100  # Change this line to set the index for each subplot
#     axs_flat[i].set_title(f"Line graph - Index {idx}")
#     axs_flat[i].set_xlabel("X axis")
#     axs_flat[i].set_ylabel("Y axis")
#     axs_flat[i].plot(outputx[idx], color="red", label="Try")
#     axs_flat[i].plot(TargetX[idx], color="black", label="Target")
#     axs_flat[i].legend()

# # Adjust layout to prevent overlapping
# plt.tight_layout()

# # Show the plot
# plt.show()

if __name__=="__main__":
    check_comp([0,1])