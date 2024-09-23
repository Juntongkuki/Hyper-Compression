import numpy as np
import time
from tqdm import tqdm
import shutil
import multiprocessing
from .param_compress_new_uint_i import decompress_v3, compress_decom_v3, Save_Num_Zero, Decode_Params
import torch
import os


def dummy_task():
    while True:
        pass

def save_uintn_list_to_bin(int_list, n, file_path):
    num_bytes = (n + 7) // 8
    with open(file_path, 'wb') as f:
        for number in int_list:
            masked_number = int(number) & ((1 << n) - 1)
            for i in range(num_bytes):
                byte = (masked_number >> (8 * (num_bytes - 1 - i))) & 0xFF
                f.write(byte.to_bytes(1, byteorder='big'))


def trans2byte(bitstring):
    output = int(bitstring, 2)
    return output




def save_uintn_list_to_bits(Result, uint_i, save_path):
    bitstring = ''.join(format(number, f'0{uint_i}b') for number in Result)
    if len(bitstring) % 8 == 0:
        n_zero = 0
    else:
        n_zero = (8 - len(bitstring) % 8) # 前面
        bitstring = '0'*n_zero + bitstring # 前面补n_zero个0


    trans_uint_list = []
    for i in [bitstring[i:i+8] for i in range(0, len(bitstring), 8)]:
        trans_uint_list.append(trans2byte(i))

    np.array(trans_uint_list, dtype=np.uint8).tofile(save_path+f'_{n_zero}.bin')



def save_new_param_uncompress(root_dir, ori_param, num_ori):
    """
    root_dir : “ .../Compressed_Dir/ ”
    param : param
    num_ori : i-layer
    """

    name_size = ''
    for i in ori_param.shape:
        name_size += '_' + str(i)

    save_name = f"{str(num_ori)}{name_size}_0_0.bin"
    save_path = root_dir + save_name
    ori_param_flatten = ori_param.flatten()
    ori_param_flatten.astype(np.float32).tofile(save_path)  # save as float32 array



def save_new_param_compress(root_dir, output_idx, num_ori, best_class, num_inner, ori_shape):

    name_size = ''
    for i in ori_shape:
        name_size += '_' + str(i)


    if int(np.max(output_idx))==0:
        uint_i = 1
    else:
        uint_i = int(np.max(output_idx)).bit_length()

    save_name = f"{str(num_ori)}{name_size}_{uint_i}"
    save_path = root_dir + save_name

    output_idx = output_idx.astype(np.int64)

    save_uintn_list_to_bits(output_idx, uint_i, save_path)






def multi_compress_decom_v3(param,
                            num_ori,
                            rect_l,
                            num_inner_list,
                            root_dir,
                            class_max,
                            loss_max,
                            loss_shredhold,
                            set_loss_limit):
    try:
        if param.flatten().shape[0] >= 10:
            """
            results[0] : MAE_loss list
            results[1] ：the best number of class
            results[2] ：new parameters
            results[3] : compressed results to be saved
            results[4] : the number of sample nodes
            results[5] : the number of inner nodes
            results[6] : the number of outer nodes
            results[7] : padding_size
            results[9] : centroid_node
            results[10] : farthest_node
            """

            results = compress_decom_v3(param,
                                        num_ori,
                                        rect_l,
                                        num_inner_list,
                                        class_max,
                                        loss_max,
                                        loss_shredhold)
            new_param = results[2]
            mean_MAE = np.mean(results[0])
            if set_loss_limit:
                if mean_MAE > 0.006:
                    print(1 / 0)
            max_loss = np.max(results[0])
            min_loss = np.min(results[0])
            max_index = (results[4]+1) + results[1] * (results[4]+1)
            if_padding = results[7]
            num_inner_index = results[8]
            best_class = results[1]
            center_node = results[9]
            farthest_node = results[10]
            save_new_param_compress(root_dir, results[3], num_ori, results[1], results[4], param.shape)
            print(f"The Number. {num_ori} Layer : Successful Compression")
        else:
            print(1/0)

    except:
        print(f"The Number. {num_ori} Layer : Save the Original Parameters")
        new_param = param
        mean_MAE = 0
        max_loss = 0
        min_loss = 0
        max_index = 0
        save_new_param_uncompress(root_dir, param, num_ori)
        if_padding = 0
        num_inner_index = 0
        best_class = 0
        center_node = np.array([0,0], dtype=np.float32)
        farthest_node = np.array([0,0], dtype=np.float32)

    return new_param, mean_MAE, max_index, if_padding, num_inner_index, best_class, center_node, farthest_node



def compress_params(model,
                    Save_CompressedResult_RootPath,
                    rect_l,
                    num_inner_list,
                    class_max,
                    loss_max,
                    loss_shredhold,
                    num_cores,
                    set_loss_limit):

    root_dir = Save_CompressedResult_RootPath

    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
        os.makedirs(root_dir, exist_ok=True)
    else:
        os.makedirs(root_dir, exist_ok=True)

    np.array([num_inner_list]).astype(np.uint64).tofile(root_dir+'num_inner_list.bin')

    params_list = list(model.parameters())

    with multiprocessing.Pool(processes=num_cores) as pool:
        new_params_list_multi = pool.starmap(multi_compress_decom_v3,
                                        [[params_list[num_ori].data.cpu().numpy(), num_ori, rect_l, num_inner_list,
                                          root_dir, class_max, loss_max, loss_shredhold, set_loss_limit] for num_ori in range(len(params_list))])



    new_params_list = [t[0] for t in new_params_list_multi]
    MAE = np.array([t[1] for t in new_params_list_multi])


    total_mean_MAE = np.mean(MAE)
    total_max_MAE = np.max(MAE)
    total_min_MAE = np.min(MAE[MAE != 0])
    total_max_index = np.mean(np.array([t[2] for t in new_params_list_multi]))


    if_padding = np.array([t[3] for t in new_params_list_multi])
    if_padding.astype(np.uint8).tofile(root_dir+'if_padding.bin')

    num_inner_index = np.array([t[4] for t in new_params_list_multi])
    num_inner_index.astype(np.uint8).tofile(root_dir+'num_inner_index.bin')

    class_list = np.array([t[5] for t in new_params_list_multi])
    class_list.astype(np.uint8).tofile(root_dir+'class_list.bin')

    center_node_list = np.array([t[6] for t in new_params_list_multi])
    center_node_list.tofile(root_dir+'center_node_list.bin')

    farthest_node_list = np.array([t[7] for t in new_params_list_multi])
    farthest_node_list.tofile(root_dir + 'farthest_node_list.bin')


    old_param = list(model.parameters())[0].detach().cpu().clone().numpy()
    with torch.no_grad():
        for i in tqdm(range(len(params_list))):
            ori_param = params_list[i].data
            new_param = new_params_list[i]
            params_list[i].copy_(torch.tensor(new_param).float().cuda())
    new_param = list(model.parameters())[0].detach().cpu().numpy()

    if np.array_equal(old_param, new_param):
       print("There is something wrong!")

    compressed_size = get_folder_size(Save_CompressedResult_RootPath)

    return compressed_size, model


def read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding):

    layer_param_int = []
    bit_save = np.fromfile(layer_path, dtype=np.uint8)
    results8bits = "".join(format(i, f'0{8}b') for i in bit_save)[uint_i_padding:]

    for i in range(0, len(results8bits), uint_i):
        index_bits = results8bits[i:i + uint_i]
        layer_param_int.append(trans2byte(index_bits))

    return np.array(layer_param_int)


def find_nth_occurrence(string, substring, n):
    start = 0
    for i in range(n):
        start = string.index(substring, start) + 1
    return start - 1

def decompress_params(Decode_Param_Path,
                      rect_l,
                      num_inner_list):

    """ load some auxiliary data for decompression """
    if_padding = np.fromfile(Decode_Param_Path + 'if_padding.bin', dtype=np.uint8)
    num_inner_index = np.fromfile(Decode_Param_Path + 'num_inner_index.bin', dtype=np.uint8)
    class_list = np.fromfile(Decode_Param_Path + 'class_list.bin', dtype=np.uint8)
    center_node_list = np.fromfile(Decode_Param_Path + 'center_node_list.bin', dtype=np.float32)
    center_node_list = center_node_list.reshape(-1,2)
    farthest_node_list = np.fromfile(Decode_Param_Path + 'farthest_node_list.bin', dtype=np.float32)
    farthest_node_list = farthest_node_list.reshape(-1,2)

    file_names = os.listdir(Decode_Param_Path)

    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')])
            split_str = file_name[file_name.index('_') + 1: -4].split('_')
            shape = []
            for j in split_str[:-2]:
                shape.append(int(j))
            shape = np.array(shape)
            uint_i = int(split_str[-2])
            uint_i_padding = int(split_str[-1])
            dict_layer_files[i_layer] = [Decode_Param_Path + file_name, shape, uint_i, uint_i_padding]
        except:
            continue

    num_layers = len(list(dict_layer_files.values()))

    decode_params_list = []
    for i in tqdm(range(num_layers)):

        layer_path, layer_shape, uint_i, uint_i_padding = dict_layer_files[i]

        if uint_i == 0:
            param1 = np.fromfile(layer_path, dtype=np.float32)
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
            decode_params_list.append(param1)

        else:
            if_padding_i = if_padding[i]
            num_inner_index_i = num_inner_index[i]
            class_list_i = class_list[i]
            center_node_i = center_node_list[i]
            farthest_node_i = farthest_node_list[i]

            index_array = read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding)

            param1 = decompress_v3(index_array, num_inner_index_i, class_list_i, if_padding_i, center_node_i, farthest_node_i,
                                   rect_l, num_inner_list)

            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)

            decode_params_list.append(param1)

    return decode_params_list


def decompress_params_layer(Decode_Param_Path, rect_l, num_inner_list, layer_index):
    if_padding = np.fromfile(Decode_Param_Path + 'if_padding.bin', dtype=np.uint8)
    num_inner_index = np.fromfile(Decode_Param_Path + 'num_inner_index.bin', dtype=np.uint8)
    class_list = np.fromfile(Decode_Param_Path + 'class_list.bin', dtype=np.uint8)
    center_node_list = np.fromfile(Decode_Param_Path + 'center_node_list.bin', dtype=np.float32)
    center_node_list = center_node_list.reshape(-1,2)
    farthest_node_list = np.fromfile(Decode_Param_Path + 'farthest_node_list.bin', dtype=np.float32)
    farthest_node_list = farthest_node_list.reshape(-1,2)

    # Get all file names in the root folder that do not contain "_"
    start_index = layer_index * 8
    if layer_index != 9:
        num_files = 8
    elif layer_index == 9:
        num_files = 2
    else:
        print("layer_index is out of range when decoding!")

    file_names = [name for name in os.listdir(Decode_Param_Path) if name[0].isdigit()]
    file_names = sorted(file_names, key=lambda name: int(name.split('_')[0]))
    file_names = file_names[start_index:start_index + num_files]

    # Construct a dictionary mapping layer_i to the file paths
    dict_layer_files = {}
    for file_name in file_names:
        try:
            i_layer = int(file_name[:file_name.index('_')])
            split_str = file_name[file_name.index('_') + 1: -4].split('_')
            shape = []
            for j in split_str[:-2]:
                shape.append(int(j))
            shape = np.array(shape)
            uint_i = int(split_str[-2])
            uint_i_padding = int(split_str[-1])
            dict_layer_files[i_layer] = [Decode_Param_Path + file_name, shape, uint_i, uint_i_padding]
        except:
            # Exception means it's not a "layer params" file
            continue

    num_layers = len(list(dict_layer_files.values()))

    decode_params_list = []

    for i in range(num_layers):
        # real_i
        real_i = i + start_index
        layer_path, layer_shape, uint_i, uint_i_padding = dict_layer_files[real_i]

        if uint_i == 0:  # If compression failed, directly load the file
            param1 = np.fromfile(layer_path, dtype=np.float32)
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)
            decode_params_list.append(param1)

        else:
            if_padding_i = if_padding[real_i]
            num_inner_index_i = num_inner_index[real_i]
            class_list_i = class_list[real_i]
            center_node_i = center_node_list[real_i]
            farthest_node_i = farthest_node_list[real_i]

            index_array = read_uintn_list_from_bin(layer_path, uint_i, uint_i_padding)

            param1 = decompress_v3(index_array, num_inner_index_i, class_list_i, if_padding_i, center_node_i, farthest_node_i,
                                   rect_l, num_inner_list)
            # param1: Decode integer index results into new float parameters
            param1 = np.array(param1).astype(np.float32).reshape(layer_shape)

            decode_params_list.append(param1)

    return decode_params_list




def param_split(param):
    param_flatten = param.flatten()

    param_decimal = np.array([])
    param_int = np.array([])


    t2_start = time.perf_counter()

    with multiprocessing.Pool(processes=8) as pool:

        args_list = [(index, value) for index, value in enumerate(param_flatten)]

        param_int = pool.map(find_nonzero_indices, args_list)
        param_int = [index for index in param_int if index is not None]

        param_decimal = pool.map(find_nonzero_value, args_list)
        param_decimal = [value for value in param_decimal if value is not None]

    t2_end = time.perf_counter()
    t2 = t2_end - t2_start

    with multiprocessing.Pool(processes=8) as pool:
        param_dis = pool.starmap(get_param_dis, [[param_int[ele-1], param_int[ele]] for ele in range(1, len(param_int))])
    param_dis = np.append(param_int[0], param_dis)

    return param_decimal, param_dis


def find_nonzero_value(args):
    index, value = args
    if value != 0:
        return value
    else:
        return None



def get_param_dis(param_int_l, param_int_r):
    return param_int_r - param_int_l



def find_nonzero_indices(args):
    index, value = args
    if value != 0:
        return index
    else:
        return None



def flatten_restore(new_param_flatten, param_dis, param_layer_shape):
    param_index = [param_dis[0]]
    for i in range(1, len(param_dis)):
        param_index.append(param_index[i-1]+param_dis[i])

    new_param_back_flatten = np.zeros(param_layer_shape).flatten()

    for index in range(len(param_index)): # param_index : new_param_flatten对应的index
        new_param_back_flatten[int(param_index[index])] = new_param_flatten[index]

    new_param = new_param_back_flatten.reshape(param_layer_shape)

    return new_param




def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Skip if it is a symbolic link
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size