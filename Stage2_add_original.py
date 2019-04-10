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
parser2.add_argument('--scale', type=str, default=2,
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
parser3.add_argument('--scale', type=str, default=2,
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
parser4.add_argument('--scale', type=str, default=2,
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
input_dir = '../LearningToSeeInDark/dataset/Sony/short/'
gt_dir = '../LearningToSeeInDark/dataset/Sony/long/'
result_dir = 'Result/'
model_dir = 'ckpt/'
test_name = ''
filter = 'haar' #'bior1.3'

# Parameters
ps = 256
save_freq = 15
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

allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

#LL
model_ll = EDSR(args1)
model_ll.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_ll_add_original_sony_e0640.pth'))
model_ll.cuda()
#LH
model_lh = EDSR(args2)
model_lh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_lh_sony_e0074.pth'))
model_lh.cuda()
#HL
model_hl = EDSR(args3)
model_hl.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_hl_sony_e0074.pth'))
model_hl.cuda()
#HH
model_hh = EDSR(args4)
model_hh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_hh_sony_e0074.pth'))
model_hh.cuda()

# combine network
model = EDSR(args5)
#model_hh.load_state_dict(torch.load(model_dir + test_name + 'Wavelet_combine_sony_e0079.pth'))
model.cuda()
opt = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(lastepoch, 101):
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

        (in_ll, in_lh, in_hl, in_hh) = Wavelet.decompose4ch(input_patch[0, :, :, :], filter)
        (gt_ll, gt_lh, gt_hl, gt_hh) = Wavelet.decompose3ch(gt_patch[0, :, :, :], filter)

        in_img_ll = torch.from_numpy(input_patch).permute(0, 3, 1, 2).cuda()
        in_img_lh = torch.from_numpy(np.expand_dims(in_lh, axis=0)).permute(0, 3, 1, 2).cuda()
        in_img_hl = torch.from_numpy(np.expand_dims(in_hl, axis=0)).permute(0, 3, 1, 2).cuda()
        in_img_hh = torch.from_numpy(np.expand_dims(in_hh, axis=0)).permute(0, 3, 1, 2).cuda()
        in_img_ori = torch.from_numpy(input_patch).permute(0, 3, 1, 2).cuda()

        gt_img_ll = torch.from_numpy(np.expand_dims(gt_ll, axis=0)).permute(0, 3, 1, 2).cuda()
        gt_img_lh = torch.from_numpy(np.expand_dims(gt_lh, axis=0)).permute(0, 3, 1, 2).cuda()
        gt_img_hl = torch.from_numpy(np.expand_dims(gt_hl, axis=0)).permute(0, 3, 1, 2).cuda()
        gt_img_hh = torch.from_numpy(np.expand_dims(gt_hh, axis=0)).permute(0, 3, 1, 2).cuda()
        gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).cuda()

        # Stage 1
        with torch.no_grad():
            model_ll.eval()
            out_img_ll = model_ll(in_img_ll)
            model_lh.eval()
            out_img_lh = model_lh(in_img_lh)
            model_hl.eval()
            out_img_hl = model_hl(in_img_hl)
            model_hh.eval()
            out_img_hh = model_hh(in_img_hh)

        LL, LH, HL, HH = out_img_ll.permute(0, 2, 3, 1).cpu().data.numpy(), \
                         out_img_lh.permute(0, 2, 3, 1).cpu().data.numpy(), \
                         out_img_hl.permute(0, 2, 3, 1).cpu().data.numpy(), \
                         out_img_hh.permute(0, 2, 3, 1).cpu().data.numpy()
        LL = torch.from_numpy(np.minimum(np.maximum(LL, 0), 1)).permute(0, 3, 1, 2).cuda()
        LH = torch.from_numpy(np.minimum(np.maximum(LH, 0), 1)).permute(0, 3, 1, 2).cuda()
        HL = torch.from_numpy(np.minimum(np.maximum(HL, 0), 1)).permute(0, 3, 1, 2).cuda()
        HH = torch.from_numpy(np.minimum(np.maximum(HH, 0), 1)).permute(0, 3, 1, 2).cuda()
        #in_img = Wavelet.combine3ch(LL, LH, HL, HH)
        in_img = torch.cat((LL, LH, HL, HH, in_img_ori), 1)
        model.train()
        model.zero_grad()
        #in_img = torch.from_numpy(np.expand_dims(in_img, axis=0)).permute(0, 3, 1, 2).cuda()
        out_img = model(in_img)
        loss = reduce_mean(out_img, gt_img)
        loss.backward()
        opt.step()

        g_loss[ind] = loss.cpu().data

        print("%d %d Loss=%.3f Time=%.3f" % (
                epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st))


    if epoch % save_freq == 0:
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
                    input_full = input_full[:, :512, :512, :]

                    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                    im = im[:1024,:1024]
                    scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
                    # scale_full = np.minimum(scale_full, 1.0)

                    gt_raw = rawpy.imread(gt_path)
                    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                    im = im[:1024, :1024]
                    gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

                    input_full = np.minimum(input_full, 1.0)

                    #in_img = torch.from_numpy(input_full).permute(0, 3, 1, 2).to(device)

                    (in_ll, in_lh, in_hl, in_hh) = Wavelet.decompose4ch(input_full[0, :, :, :], filter)
                    (gt_ll, gt_lh, gt_hl, gt_hh) = Wavelet.decompose3ch(gt_full[0, :, :, :], filter)

                    in_img_ll = torch.from_numpy(input_full).permute(0, 3, 1, 2).cuda()
                    in_img_lh = torch.from_numpy(np.expand_dims(in_lh, axis=0)).permute(0, 3, 1, 2).cuda()
                    in_img_hl = torch.from_numpy(np.expand_dims(in_hl, axis=0)).permute(0, 3, 1, 2).cuda()
                    in_img_hh = torch.from_numpy(np.expand_dims(in_hh, axis=0)).permute(0, 3, 1, 2).cuda()
                    in_img_ori = torch.from_numpy(input_full).permute(0, 3, 1, 2).cuda()
                    out_img_ll = model_ll(in_img_ll)
                    out_img_lh = model_lh(in_img_lh)
                    out_img_hl = model_hl(in_img_hl)
                    out_img_hh = model_hh(in_img_hh)

                    LL, LH, HL, HH = out_img_ll.permute(0, 2, 3, 1).cpu().data.numpy(), \
                                     out_img_lh.permute(0, 2, 3, 1).cpu().data.numpy(), \
                                     out_img_hl.permute(0, 2, 3, 1).cpu().data.numpy(), \
                                     out_img_hh.permute(0, 2, 3, 1).cpu().data.numpy()
                    LL = torch.from_numpy(np.minimum(np.maximum(LL, 0), 1)).permute(0, 3, 1, 2).cuda()
                    LH = torch.from_numpy(np.minimum(np.maximum(LH, 0), 1)).permute(0, 3, 1, 2).cuda()
                    HL = torch.from_numpy(np.minimum(np.maximum(HL, 0), 1)).permute(0, 3, 1, 2).cuda()
                    HH = torch.from_numpy(np.minimum(np.maximum(HH, 0), 1)).permute(0, 3, 1, 2).cuda()
                    in_img = torch.cat((LL, LH, HL, HH, in_img_ori), 1)
                    model.eval()
                    out_img = model(in_img)
                    output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                    output = np.minimum(np.maximum(output, 0), 1)

                    output = output[0, :, :, :]
                    gt_full = gt_full[0, :, :, :]
                    scale_full = scale_full[0, :, :, :]
                    origin_full = scale_full
                    scale_full = scale_full * np.mean(gt_full) / np.mean(scale_full)

                    psnr.append(skm.compare_psnr(gt_full[:, :, :], output[:, :, :]))
                    ssim.append(skm.compare_ssim(gt_full[:, :, :], output[:, :, :], multichannel=True))
                    print('psnr: ', psnr[-1], 'ssim: ', ssim[-1])

                    temp = np.concatenate((gt_full, output), axis=1)
                    scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                        result_dir + '%05d_00_train_%d.jpg' % (test_id, ratio))
        print('mean psnr: ', np.mean(psnr))
        print('mean ssim: ', np.mean(ssim))
        torch.save(model.state_dict(), model_dir + 'Wavelet_add_original_sony_e%04d.pth' % epoch)

print("Done...")

output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
output = np.minimum(np.maximum(output, 0), 1)
plt.figure(1)
plt.imshow(output[0,:,:,:])
plt.figure(2)
plt.imshow(gt_patch[0,:,:,:])


print('mean psnr: ', np.mean(psnr))
print('mean ssim: ', np.mean(ssim))