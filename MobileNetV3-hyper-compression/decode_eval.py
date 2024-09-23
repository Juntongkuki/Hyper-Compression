from lib.Compress_Params_Standard import *
import time
import argparse
from models import build_mobilenet_v3, prepare_dataloader, eval_mobilenetv3_cifar10


def get_args():
    parser = argparse.ArgumentParser(description='Decompress the parameters of MobileNetV3 from the dictory')
    parser.add_argument('--model_name', type=str, default='MobileNetV3', help='The original model to be compressed')
    parser.add_argument('--decode_path', type=str, default='./compressed_result/MobileNetV3_l_08', help='The dictory where the compressed parameters are saved')
    parser.add_argument('--gpu', type=bool, default=False, help='use cuda')
    return parser.parse_args()



if __name__=='__main__':

    args = get_args()

    Decode_Param_Path = args.decode_path

    model = build_mobilenet_v3()
    model.load_state_dict(torch.load(Decode_Param_Path + '/Back_Params.pth'))

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

    if args.gpu:
        model.cuda()

    _, test_loader = prepare_dataloader()
    accuracy = eval_mobilenetv3_cifar10(model, test_loader)
    print(f"Accuracy : {accuracy}")