
from lib.unet import UNet
from lib.Compress_Params_Standard import *
import time
import argparse
from lib.unet_load import *
from lib.unet_utils import *
from lib.unet_eval import *


def eval_net_carvana(net,
              val_percent=0.05,
              gpu=False,
              img_scale=0.5):

    dir_img = '../data/carvana-image-masking-challenge/train/'
    dir_mask = '../data/carvana-image-masking-challenge/train_masks/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
    val_dice = eval_net(net, val, gpu)
    return val_dice


def get_args():
    parser = argparse.ArgumentParser(description='Decompress the parameters of UNet from the dictory')
    parser.add_argument('--model_name', type=str, default='UNet', help='The original model to be compressed')
    parser.add_argument('--decode_path', type=str, default='./compressed_result/UNet_l_01', help='The dictory where the compressed parameters are saved')
    parser.add_argument('--gpu', type=bool, default=False, help='use cuda')
    parser.add_argument('--scale', type=float, default=0.5, help='downscaling factor of the images')
    parser.add_argument('--val_percent', type=float, default=0.2, help='downscaling factor of the images')
    return parser.parse_args()




if __name__=='__main__':

    args = get_args()

    Decode_Param_Path = args.decode_path

    flops_i = 'flops-00'  # original unet
    model = UNet(n_channels=3, n_classes=1,
                 f_channels='../lib/UNet_Channels_TXT/{}/original_model_channels.txt'.format(flops_i))

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
            # print(list(model.parameters())[0][0][0][0][0])
            model_params[i].copy_(torch.tensor(decode_params_list[i]).float().cuda())
            # print(list(model.parameters())[0][0][0][0][0])

    if args.gpu:
        model.cuda()

    dice = eval_net_carvana(net=model,
                            val_percent=args.val_percent,
                            gpu=args.gpu,
                            img_scale=args.scale)
    print(f"Dice : {dice}")