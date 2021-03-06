import os, time
import scipy.io as sio
import scipy
import numpy as np
import rawpy
import glob
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from EDSR import EDSR, EDSR_
import argparse
import skimage.measure as skm
import Wavelet

# Stage 1
parser1 = argparse.ArgumentParser(description='EDSR 1')
parser1.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser1.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser1.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser1.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser1.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser1.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args1 = parser1.parse_args()

# Stage 2
parser2 = argparse.ArgumentParser(description='EDSR 2')
parser2.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser2.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser2.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser2.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser2.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser2.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args2 = parser2.parse_args()

# Stage 3
parser3 = argparse.ArgumentParser(description='EDSR 3')
parser3.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser3.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser3.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser3.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser3.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser3.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args3 = parser3.parse_args()

# Stage 4
parser4 = argparse.ArgumentParser(description='EDSR 4')
parser4.add_argument('--n_resblocks', type=int, default=4,
                    help='number of residual blocks')
parser4.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser4.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser4.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser4.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser4.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args4 = parser4.parse_args()

# Stage 5
parser5 = argparse.ArgumentParser(description='EDSR 5')
parser5.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser5.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser5.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser5.add_argument('--scale', type=str, default=2,
                    help='super resolution scale')
parser5.add_argument('--n_colors', type=int, default=16,
                    help='number of input color channels to use')
parser5.add_argument('--o_colors', type=int, default=3,
                    help='number of output color channels to use')
args5 = parser5.parse_args()

# Directories
input_dir  = '../Dataset/Sony/Sony/short/'
gt_dir     = '../Dataset/Sony/Sony/long/'
result_dir = 'Result/final/'
model_dir  = 'ckpt/'
test_name  = ''
filter     = 'haar'
cluster    = False

# Parameters
ps            = 512
save_freq     = 25
learning_rate = 1e-4

# Check if cuda available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


g_loss = np.zeros((5000, 1))

if cluster:
    input_images = {}
    input_images['300'] = [None]*len(train_ids)
    input_images['250'] = [None]*len(train_ids)
    input_images['100'] = [None]*len(train_ids)
    gt_images = [None] * 6000

#LL
model_ll = EDSR(args1)
model_ll.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_raw_ll_sony_e2000.pth'))
model_ll.cuda()
#LH
model_lh = EDSR(args2)
model_lh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_raw_lh_sony_e0065.pth'))
model_lh.cuda()
#HL
model_hl = EDSR(args3)
model_hl.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_raw_hl_sony_e0065.pth'))
model_hl.cuda()
#HH
model_hh = EDSR(args4)
model_hh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_raw_hh_sony_e0065.pth'))
model_hh.cuda()

# combine network
model = EDSR(args5)
model.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_enhancement_sony_e0000.pth'))
model.cuda()
opt = optim.Adam(model.parameters(), lr=learning_rate)

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
psnr = []
ssim = []
with torch.no_grad():
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
            # input_full = input_full[:, :512, :512, :]
            input_full = np.minimum(input_full, 1.0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            #im = im[:1024, :1024]
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).cuda()
            model.eval()
            out_img_ll = model_ll(in_img)
            model.eval()
            out_img_lh = model_lh(in_img)
            model.eval()
            out_img_hl = model_hl(in_img)
            model.eval()
            out_img_hh = model_hh(in_img)

            in_img = torch.cat((out_img_ll, out_img_lh, out_img_hl, out_img_hh, in_img), 1)

            model.eval()
            out_img = model(in_img)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]

            psnr.append(skm.compare_psnr(gt_full[:, :, :], output[:, :, :]))
            ssim.append(skm.compare_ssim(gt_full[:, :, :], output[:, :, :], multichannel=True))
            print('psnr: ', psnr[-1], 'ssim: ', ssim[-1])

            temp = np.concatenate((gt_full, output), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%05d_00_train_%d.jpg' % (test_id, ratio))
            torch.cuda.empty_cache()
        print('mean psnr: ', np.mean(psnr))
        print('mean ssim: ', np.mean(ssim))

print("Done...")
