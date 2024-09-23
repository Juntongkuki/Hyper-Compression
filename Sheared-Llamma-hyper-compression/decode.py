import numpy as np
import torch
from lib.Compress_Params_Standard import *
import time
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM



def get_args():
    parser = argparse.ArgumentParser(description='Decompress the parameters of Sheared-LlaMA from the dictory')
    parser.add_argument('--model_name', type=str, default='Sheared-LlaMA-1.3B', help='The original model to be compressed')
    parser.add_argument('--decode_path', type=str, default='./compressed_result/Sheared-LlaMA-1.3B_l_01', help='The dictory where the compressed parameters are saved')
    parser.add_argument('--gpu', type=bool, default=False, help='use cuda')

    return parser.parse_args()



if __name__=='__main__':

    args = get_args()

    Decode_Param_Path = args.decode_path
    model = AutoModelForCausalLM.from_pretrained("./Sheared-LLaMA-1.3B-Pruned")

    print("\n")
    print("#" * 10 + " Let's decompress the encoded parameters " + "#" * 10)
    t_start = time.perf_counter()
    num_inner_list = np.fromfile(Decode_Param_Path + '/Compressed_Dir/num_inner_list.bin', dtype=np.uint64)
    rect_l_str = Decode_Param_Path[find_nth_occurrence(Decode_Param_Path, "/", 2):].split('_')[2]
    rect_l = float(rect_l_str[:1] + '.' + rect_l_str[1:])
    decode_params_list = decompress_params(Decode_Param_Path + '/Compressed_Dir/', rect_l, num_inner_list)
    t_end = time.perf_counter()
    print(f"The times of decompression : {np.round(t_end - t_start,2)} Seconds")

    print("\n")
    print("#"*10 + " Let's Evaluate the model's performance with lossy parameters " + "#"*10)

    model_params = list(model.parameters())
    with torch.no_grad():
        for i in range(len(decode_params_list)):
            model_params[i].copy_(torch.tensor(decode_params_list[i]).float().cuda())

