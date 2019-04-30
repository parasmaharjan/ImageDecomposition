import os, time
import scipy.io as sio
import numpy as np
import rawpy
import glob
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from EDSR import EDSR
import argparse
import skimage.measure as skm
import Wavelet

# Stage 1
parser1 = argparse.ArgumentParser(description='EDSR 1')
parser1.add_argument('--n_resblocks', type=int, default=8,
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

# Directories
input_dir  = '../dataset/Sony/short/'
gt_dir     = '../dataset/Sony/long/'
result_dir = 'result_Sony/'
model_dir  = 'ckpt/'
test_name  = 'Stage1/'
filter     = 'haar'
cluster    = True
lf         = True  # 0: ll, 1: lh, 2: hl, 3: hh

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

# validation
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


g_loss_ll = np.zeros((5000, 1))
g_loss_lh = np.zeros((5000, 1))
g_loss_hl = np.zeros((5000, 1))
g_loss_hh = np.zeros((5000, 1))

if cluster:
    input_images = {}
    input_images['300'] = [None]*len(train_ids)
    input_images['250'] = [None]*len(train_ids)
    input_images['100'] = [None]*len(train_ids)
    gt_images = [None] * 6000

if lf:
    #LL
    model_ll = EDSR(args1)
    #model_ll.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_ll_sony_e0059.pth'))
    model_ll.cuda()
    opt_ll = optim.Adam(model_ll.parameters(), lr=learning_rate)
else:
    #LH
    model_lh = EDSR(args2)
    #model_lh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_lh_sony_e0059.pth'))
    model_lh.cuda()
    opt_lh = optim.Adam(model_lh.parameters(), lr=learning_rate)
    #HL
    model_hl = EDSR(args3)
    #model_hl.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_hl_sony_e0059.pth'))
    model_hl.cuda()
    opt_hl = optim.Adam(model_hl.parameters(), lr=learning_rate)
    #HH
    model_hh = EDSR(args4)
    #model_hh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_hh_sony_e0059.pth'))
    model_hh.cuda()
    opt_hh = optim.Adam(model_hh.parameters(), lr=learning_rate)

for epoch in range(0, 2001):
    cnt = 0

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if cluster:
            if input_images[str(ratio)[0:3]][ind] is None:
                raw = rawpy.imread(in_path)
                input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)
            # crop
            H = input_images[str(ratio)[0:3]][ind].shape[1]
            W = input_images[str(ratio)[0:3]][ind].shape[2]

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)
            input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
        else:
            raw = rawpy.imread(in_path)
            input_images = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images = np.expand_dims(np.float32(im / 65535.0), axis=0)

            # crop
            H = input_images.shape[1]
            W = input_images.shape[2]

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)
            input_patch = input_images[:, yy:yy + ps, xx:xx + ps, :]
            gt_patch = gt_images[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]


        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        gt_patch = np.maximum(gt_patch, 0.0)

        #(in_ll, in_lh, in_hl, in_hh) = Wavelet.decompose4ch(input_patch[0, :, :, :], filter)
        (gt_ll, gt_lh, gt_hl, gt_hh) = Wavelet.decompose3ch(gt_patch[0, :, :, :], filter)

        in_img = torch.from_numpy(input_patch).permute(0, 3, 1, 2).cuda()
        # in_img_ll = torch.from_numpy(np.expand_dims(in_ll, axis=0)).permute(0, 3, 1, 2).cuda()
        gt_img_ll = torch.from_numpy(np.expand_dims(gt_ll, axis=0)).permute(0, 3, 1, 2).cuda()
        if lf:
            model_ll.train()
            model_ll.zero_grad()
        else:
            # in_img_lh = torch.from_numpy(np.expand_dims(in_lh, axis=0)).permute(0, 3, 1, 2).cuda()
            # in_img_hl = torch.from_numpy(np.expand_dims(in_hl, axis=0)).permute(0, 3, 1, 2).cuda()
            # in_img_hh = torch.from_numpy(np.expand_dims(in_hh, axis=0)).permute(0, 3, 1, 2).cuda()

            gt_img_lh = torch.from_numpy(np.expand_dims(gt_lh, axis=0)).permute(0, 3, 1, 2).cuda()
            gt_img_hl = torch.from_numpy(np.expand_dims(gt_hl, axis=0)).permute(0, 3, 1, 2).cuda()
            gt_img_hh = torch.from_numpy(np.expand_dims(gt_hh, axis=0)).permute(0, 3, 1, 2).cuda()

            model_lh.train()
            model_hl.train()
            model_hh.train()
            model_lh.zero_grad()
            model_hl.zero_grad()
            model_hh.zero_grad()

        if lf:
            out_img_ll = model_ll(in_img)
        else:
            out_img_lh = model_lh(in_img)
            out_img_hl = model_hl(in_img)
            out_img_hh = model_hh(in_img)

        if lf:
            loss_ll = reduce_mean(out_img_ll, gt_img_ll)
            loss_ll.backward()
            opt_ll.step()
            g_loss_ll[ind] = loss_ll.cpu().data
            print("%d %d Loss_LL=%.3f Time=%.3f" % (
                epoch, cnt, np.mean(g_loss_ll[np.where(g_loss_ll)]), time.time() - st))
        else:
            loss_lh = reduce_mean(out_img_lh, gt_img_lh)
            loss_lh.backward()
            opt_lh.step()
            loss_hl = reduce_mean(out_img_hl, gt_img_hl)
            loss_hl.backward()
            opt_hl.step()
            loss_hh = reduce_mean(out_img_hh, gt_img_hh)
            loss_hh.backward()
            opt_hh.step()

            g_loss_lh[ind] = loss_lh.cpu().data
            g_loss_hl[ind] = loss_hl.cpu().data
            g_loss_hh[ind] = loss_hh.cpu().data

            print("%d %d Loss_LL=%.3f Loss_LH=%.3f Loss_HL=%.3f Loss_HH=%.3f Time=%.3f" % (
                epoch, cnt, np.mean(g_loss_ll[np.where(g_loss_ll)]),
                np.mean(g_loss_lh[np.where(g_loss_lh)]), np.mean(g_loss_hl[np.where(g_loss_hl)]),
                np.mean(g_loss_hh[np.where(g_loss_hh)]), time.time() - st))

        if epoch % save_freq == 0:
            if not os.path.isdir(model_dir):
               os.makedirs(model_dir)
            # save model
            if lf:
                torch.save(model_ll.state_dict(), model_dir + 'Wavelet_512ll8_sony_e%04d.pth' % epoch)
            else:
                torch.save(model_lh.state_dict(), model_dir + 'Wavelet_lh_sony_e%04d.pth' % epoch)
                torch.save(model_hl.state_dict(), model_dir + 'Wavelet_hl_sony_e%04d.pth' % epoch)
                torch.save(model_hh.state_dict(), model_dir + 'Wavelet_hh_sony_e%04d.pth' % epoch)
print('done training...')

