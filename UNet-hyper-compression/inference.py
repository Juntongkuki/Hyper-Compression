from optparse import OptionParser
from lib.unet_utils import *
from lib.unet_eval import *
from lib.unet_load import *
from lib.unet.unet_parts import *
from lib.Compress_Params_Standard import *
from lib.unet import UNet
import gc
import argparse
import torch.nn as nn

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
    print(val_dice)



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-v', '--val_percent', dest='val_percent', type='float',
                      default=0.2, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options



if __name__ == '__main__':


    model_name = 'UNet'

    flops_i = 'flops-00' # original unet

    # Decoding files path
    Decode_Param_Path = f'./compressed_result/{model_name}_l_01'



    n_channels = 3
    n_classes = 1
    f_channels = f'../lib/UNet_Channels_TXT/{flops_i}/original_model_channels.txt'
    with open(f_channels, 'r') as f:
        channels = f.readlines()
    channels = [int(c.strip()) for c in channels]


    def inference_layer(layer_index, params, input_tensor, skip_activation):

        if layer_index == 0:
            layer = inconv(n_channels, channels[0], channels[1])

        elif layer_index == 1:
            layer = down(channels[1], channels[2], channels[3])

        elif layer_index == 2:
            layer = down(channels[3], channels[4], channels[5])

        elif layer_index == 3:
            layer = down(channels[5], channels[6], channels[7])

        elif layer_index == 4:
            layer = down(channels[7], channels[8], channels[9])

        elif layer_index == 5:
            layer = up(channels[10], channels[11], channels[12])

        elif layer_index == 6:
            layer = up(channels[13], channels[14], channels[15])

        elif layer_index == 7:
            layer = up(channels[16], channels[17], channels[18])

        elif layer_index == 8:
            layer = up(channels[19], channels[20], channels[21])

        elif layer_index == 9:
            layer = outconv(channels[21], n_classes)


        params_list = list(layer.parameters())
        with torch.no_grad():
            for i in range(len(params_list)):
                params_list[i].copy_(torch.tensor(params[i]).float())

        # layer.eval()
        layer.cuda()

        if layer_index in {5, 6, 7, 8}:
            with torch.no_grad():
                output_tensor = layer(input_tensor, skip_activation)
        else:
            with torch.no_grad():
                output_tensor = layer(input_tensor)
        return output_tensor


    def eval_net_carvana_multi_stream(net,
                         val_percent=0.05,
                         gpu=False,
                         img_scale=0.5):

        dir_img = '../data/carvana-image-masking-challenge/train/'
        dir_mask = '../data/carvana-image-masking-challenge/train_masks/'

        ids = get_ids(dir_img)
        ids = split_ids(ids)

        iddataset = split_train_val(ids, val_percent)

        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
        val_dice = eval_net_multi_stream(net, val, gpu)
        print(val_dice)

    class Block(nn.Module):
        def __init__(self, layer_id, Decode_Param_Path):
            super().__init__()
            self.layer_id = layer_id
            self.Decode_Param_Path = Decode_Param_Path

        def decode_params(self):
            Decode_Param_Path = self.Decode_Param_Path
            num_inner_list = np.fromfile(Decode_Param_Path + '/Compressed_Dir/num_inner_list.bin', dtype=np.uint64)
            rect_l_str = Decode_Param_Path[find_nth_occurrence(Decode_Param_Path, "/", 2):].split('_')[2]
            rect_l = float(rect_l_str[:1] + '.' + rect_l_str[1:])
            params = decompress_params_layer(Decode_Param_Path + '/Compressed_Dir/', rect_l, num_inner_list,
                                             layer_index=self.layer_id)
            return [torch.from_numpy(param).to('cuda:0') for param in params]

        def forward(self, previous_output, params, skip_activation):
            return inference_layer(self.layer_id, params, previous_output, skip_activation)


    class MultiStreamStreansformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.default_stream = torch.cuda.current_stream()
            self.custom_stream = torch.cuda.Stream()

            self.num_layers = 10
            self.layers = nn.ModuleList()
            for layer_id in range(self.num_layers):
                self.layers.append(Block(layer_id, Decode_Param_Path))

            self.to('cuda:0')

        def forward(self, x: torch.Tensor, multi_stream: bool):
            x = x.to('cuda:0')
            previous_output = x
            skip_0 = None
            skip_1 = None
            skip_2 = None
            skip_3 = None

            if not multi_stream:
                for i, layer in enumerate(self.layers):
                    params = layer.decode_params()
                    if i == 8:
                        skip_activation = skip_0
                    elif i == 7:
                        skip_activation = skip_1
                    elif i == 6:
                        skip_activation = skip_2
                    elif i == 5:
                        skip_activation = skip_3
                    else:
                        skip_activation = None
                    previous_output = layer.forward(previous_output, params, skip_activation)
                    if i == 0:
                        skip_0 = previous_output
                    if i == 1:
                        skip_1 = previous_output
                    if i == 2:
                        skip_2 = previous_output
                    if i == 3:
                        skip_3 = previous_output
                final_output = previous_output
            else:
                # first layer
                layer_0 = self.layers[0]
                with torch.cuda.stream(self.default_stream):
                    params = layer_0.decode_params()

                for i in range(1, self.num_layers):
                    if i == 9:
                        skip_activation = skip_0
                    elif i == 8:
                        skip_activation = skip_1
                    elif i == 7:
                        skip_activation = skip_2
                    elif i == 6:
                        skip_activation = skip_3
                    else:
                        skip_activation = None

                    with torch.cuda.stream(self.default_stream):
                        previous_output = self.layers[i - 1].forward(previous_output, params, skip_activation)
                    if i == 1:
                        skip_0 = previous_output
                    if i == 2:
                        skip_1 = previous_output
                    if i == 3:
                        skip_2 = previous_output
                    if i == 4:
                        skip_3 = previous_output

                    with torch.cuda.stream(self.custom_stream):
                        params_next = self.layers[i].decode_params()
                    torch.cuda.synchronize()
                    params = params_next
                # last layer
                with torch.cuda.stream(self.default_stream):
                    final_output = self.layers[-1].forward(previous_output, params, None)
            return final_output


    loops = 1
    args = get_args()

    model = MultiStreamStreansformer()
    if args.gpu:
        model.cuda()

    for i in range(loops):
        dice = eval_net_carvana_multi_stream(net=model,
                                val_percent=args.val_percent,
                                gpu=args.gpu,
                                img_scale=args.scale)

        torch.cuda.empty_cache()
        gc.collect()

