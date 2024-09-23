from lib.unet import UNet
from lib.Compress_Params_Standard import *
import shutil
import time
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Compress the parameters of UNet to a dictory')
    parser.add_argument('--model_name', type=str, default='UNet', help='The original model to be compressed')
    parser.add_argument('--rect_l', type=float, default=0.1, help='The length of sides of the inner square')
    parser.add_argument('--num_inner_list', type=list, default=[225], help='The number of the sample nodes in the inner square')
    parser.add_argument('--class_max', type=int, default=3, help='The maximum number of categories of points outside the inner square')
    parser.add_argument('--loss_max', type=float, default=0.001, help='The maximum loss per layer')
    parser.add_argument('--loss_threshold', type=float, default=0.001, help='The expected loss per layer')
    parser.add_argument('--num_cores', type=int, default=1, help='The number of multithreaded threads')
    parser.add_argument('--set_loss_limit', type=int, default=False,
                        help='If the loss of a layer exceeds this limit, the original parameters will be saved directly to prevent significant impact on model performance')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    flops_i = 'flops-00' # Original UNet model
    model = UNet(n_channels=3, n_classes=1,
                 f_channels='../lib/UNet_Channels_TXT/{}/original_model_channels.txt'.format(flops_i))
    model.load_state_dict(torch.load('../lib/UNet_Channels_TXT/{}/original_MODEL.pth'.format(flops_i)))

    # root path to save results
    Save_Param_Path = f'./compressed_result/{args.model_name}_l_{str(args.rect_l)[0] + str(args.rect_l)[2:]}/'

    if os.path.exists(Save_Param_Path):
        shutil.rmtree(Save_Param_Path)
        os.makedirs(Save_Param_Path, exist_ok=True)
    else:
        os.makedirs(Save_Param_Path, exist_ok=True)

    Save_CompressedResult_RootPath = Save_Param_Path + 'Compressed_Dir/'
    Save_BackParam_Path = Save_Param_Path + 'Back_Params.pth'
    Save_BackParam_Path_bin = Save_Param_Path + 'pytorch_model_back.bin'

    t1_start = time.perf_counter()

    size_result, model = compress_params(model, Save_CompressedResult_RootPath, args.rect_l, args.num_inner_list, args.class_max,
                                         args.loss_max, args.loss_threshold, args.num_cores, args.set_loss_limit)
    t1_end = time.perf_counter()


    torch.save(model.state_dict(), Save_BackParam_Path)
    torch.save(model.state_dict(), Save_BackParam_Path_bin)
    size_origin = os.path.getsize(Save_Param_Path + 'Back_Params.pth')
    print(f"The bytes of the compressed dictory : {size_result} bytes")
    print(f"Compression Ratio : {np.round(size_origin / size_result, 2)}")
    print(f"Compression Time : {int(t1_end - t1_start)} Seconds = {np.round((t1_end - t1_start) / 60, 2)}Minutes")