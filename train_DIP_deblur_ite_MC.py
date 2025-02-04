import argparse
import time
import cv2
import os
import glob
import pickle
import random
from skimage.metrics import structural_similarity as ssim
from scipy.fft import dctn, idctn
import torch
import torch.nn.functional as F
import numpy as np

from utils import gen_latent_code_patch, PSNR
from function_grad import nll_grad_operator_MC_CGD
from decoder import autoencodernet

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--dataset', type=str, default='Set11', help='test dataset')
parser.add_argument('--seed', type=int, default=312, help='name of test set')
parser.add_argument('--kernel_size', type=int, default=1, help='kernel size in DIP')
parser.add_argument('--mask_const_en', type=float, default=0, help='epsilon in aperture.')
parser.add_argument('--mask_rate', type=float, default=0.8, help='aperture percentage')
parser.add_argument('--x_init', type=str, default='constant', help='forward operator')
parser.add_argument('--decodetype', type=str, default='upsample', help='upsample, transposeconv')
parser.add_argument('--num_look', type=int, default=1, help='number of looks')
parser.add_argument('--lr_NN', type=float, default=1e-3, help='DIP learning rate.')
parser.add_argument('--lr_GD', type=float, default=1e-2, help='PGD step size.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for NN training.')
parser.add_argument('--outer_ite', type=int, default=100, help='PGD iterations')
parser.add_argument('--inner_ite', type=int, default=200, help='DIP iterations')
parser.add_argument('--num_ite_MC', type=int, default=10, help='num ite in MC')
parser.add_argument('--out_nonlinear', type=bool, default=True, help='DIP output layer')
parser.add_argument('--add_std', type=float, default=0.2, help='additive noise standard deviation.')
parser.add_argument('--MC', type=bool, default=True, help='use Monte-Carlo + conjugate gradient to avoid matrix inversion')
args = parser.parse_args()
print(args)

def create_mask_en(n, frac):
    mask = torch.zeros(n, n, dtype=torch.float64) + args.mask_const_en
    mask_idx = int(n * np.sqrt(frac))
    mask[:mask_idx, :mask_idx] = 1
    print('mask idx', mask_idx)
    return mask

def train(out_path, filepaths, channel_list, dtype, device):

    img_te_num = len(filepaths)
    ########## Save the running logs ##########
    PSNR_GD_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    PSNR_NN_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    SSIM_GD_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)
    SSIM_NN_All = np.zeros([args.outer_ite+1, img_te_num], dtype=np.float64)

    ########## Loop over every test image ##########
    for img_no in range(img_te_num):
        imgName = filepaths[img_no]
        single_imgName_ = imgName.split(".")[0]
        single_imgName = single_imgName_.split("/")[-1]
        print('image name:', imgName)
        print('channel list', channel_list)

        ########## Prepare the image ##########
        Img = cv2.imread(imgName, 1)
        patch_size = np.shape(Img)[0]
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb) / 255.0
        img_gt_ = torch.from_numpy(Img_yuv[:, :, 0]).type(dtype).to(device)
        img_gt = img_gt_.detach().cpu().numpy()
        cv2.imwrite(os.path.join(out_path, "%s_raw.png" % (single_imgName)), np.clip(img_gt, 0, 1)*255.0)


        ########## generate the blurred (multi-look) measurements ##########
        img_blur = np.zeros((args.num_look, patch_size, patch_size))
        mask_matrix_en = create_mask_en(patch_size, args.mask_rate).detach().cpu().numpy()
        for look_idx in range(args.num_look):
            w_noise = torch.randn(img_gt.shape).type(dtype)
            xw = torch.mul(img_gt_, w_noise)
            xw_arr = xw.detach().cpu().numpy()

            img_dct = dctn(xw_arr, norm="ortho")
            img_dct_mask = img_dct * mask_matrix_en
            img_blur_idx = idctn(img_dct_mask, norm="ortho")
            add_noise = np.random.normal(loc=0.0, scale=args.add_std, size=(patch_size, patch_size))
            img_blur_idx += add_noise
            Axw_z_arr = img_blur_idx
            img_blur[look_idx] = Axw_z_arr

            x_init = img_blur[look_idx]
            psnr_x_init = PSNR(np.clip(x_init, 0, 1) * 255.0, img_gt * 255.0)
            ssim_x_init = ssim(np.clip(x_init, 0, 1) * 255.0, img_gt * 255.0, data_range=255)
            cv2.imwrite(os.path.join(out_path, "%s_y_blur_PSNR_%.2f_SSIM_%.4f.png" % (single_imgName, psnr_x_init, ssim_x_init)), np.clip(x_init, 0, 1) * 255.0)
            print('psnr init', psnr_x_init, 'ssim init', ssim_x_init)
            PSNR_NN_All[0, img_no] = psnr_x_init
            SSIM_NN_All[0, img_no] = ssim_x_init

            x_init_abs = np.abs(img_blur[look_idx])
            psnr_x_init_abs = PSNR(np.clip(x_init_abs, 0, 1) * 255.0, img_gt * 255.0)
            ssim_x_init_abs = ssim(np.clip(x_init_abs, 0, 1) * 255.0, img_gt * 255.0, data_range=255)
            cv2.imwrite(os.path.join(out_path, "%s_y_abs_blur_PSNR_%.2f_SSIM_%.4f.png" % (single_imgName, psnr_x_init_abs, ssim_x_init_abs)), np.clip(x_init_abs, 0, 1) * 255.0)


        ########## Init the GD input ##########
        if args.x_init == 'constant':
            x_init = np.ones((patch_size, patch_size)) * 0.5
        elif args.x_init == 'measurement_single':
            x_init = img_blur[0]
        x_new = x_init
        y = img_blur

        total_start_time = time.time()
        ########## iterative PGD ##########
        for outer_idx in range(args.outer_ite):
            print('outer ite:', outer_idx + 1)
            GD_start_time = time.time()
            if args.MC:
                grad_matrix = nll_grad_operator_MC_CGD(x_new, y, mask_matrix_en, args.add_std, args.num_ite_MC)
            x_G = x_new - args.lr_GD * grad_matrix
            x_G_save = np.clip(x_G, 0, 1) * 255.0
            psnr_GD = PSNR(x_G_save, img_gt * 255.0)
            ssim_GD = ssim(x_G_save, img_gt * 255.0, data_range=255)
            print('psnr GD', psnr_GD, 'ssim GD', ssim_GD)
            GD_end_time = time.time()

            ######### projection step: train DIP/Deep Decoder ##########
            projection_start = time.time()
            x_raw = torch.from_numpy(x_G).type(dtype).to(device)
            output_depth = 1 # number of output channels (gray scale image)
            DIP_patch_size = patch_size
            net = autoencodernet(num_output_channels=output_depth, num_channels_up=channel_list,
                                   need_sigmoid=args.out_nonlinear, decodetype=args.decodetype,
                                   kernel_size=args.kernel_size).type(dtype).to(device)
            latent_code = gen_latent_code_patch(1, DIP_patch_size, channel_list, 1).type(dtype).to(device)
            params = [x for x in net.decoder.parameters()]
            optimizer = torch.optim.Adam(params, lr=args.lr_NN, weight_decay=args.weight_decay)
            for ee in range(args.inner_ite):
                net.train()
                optimizer.zero_grad()
                x_gen_tensor = net(latent_code).squeeze(0).squeeze(0)
                loss_train = F.mse_loss(x_gen_tensor, x_raw.detach())
                loss_train.backward()
                optimizer.step()
            with torch.no_grad():
                x_gen = net(latent_code).squeeze(0).squeeze(0).detach()

            projection_end = time.time()

            with torch.no_grad():
                # net.eval()
                x_gen = x_gen.detach().cpu().numpy()
                x_new = x_gen

                x_gen_save = np.clip(x_gen, 0, 1) * 255.0
                psnr_NN = PSNR(x_gen_save, img_gt * 255.0)
                ssim_NN = ssim(x_gen_save, img_gt * 255.0, data_range=255)
                print('psnr NN', psnr_NN, 'ssim NN', ssim_NN)
                print('GD time', GD_end_time - GD_start_time, 'proj time', projection_end - projection_start)

                # Save the results and reconstructed images
                PSNR_GD_All[outer_idx+1, img_no] = psnr_GD
                SSIM_GD_All[outer_idx+1, img_no] = ssim_GD
                PSNR_NN_All[outer_idx+1, img_no] = psnr_NN
                SSIM_NN_All[outer_idx+1, img_no] = ssim_NN
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_GD_PSNR_%.2f_SSIM_%.3f.png" % (single_imgName, outer_idx, psnr_GD, ssim_GD)), x_G_save)
                cv2.imwrite(os.path.join(out_path, "%s_ite_%d_NN_PSNR_%.2f_SSIM_%.3f.png" % (single_imgName, outer_idx, psnr_NN, ssim_NN)), x_gen_save)

        total_end_time = time.time()
        print('total running time:', total_end_time - total_start_time)

    return PSNR_GD_All, SSIM_GD_All, PSNR_NN_All, SSIM_NN_All

if __name__ == '__main__':
    ############# Initialize the random seed ##############
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dtype = torch.float64
    channel_list = [100,50,25,10]

    ############# testing data and saving path #############
    out_path = os.path.join('./results', "_".join(map(str, [args.dataset,
                                                                args.mask_const_en, args.weight_decay,
                                                                args.outer_ite, args.inner_ite,
                                                                args.lr_NN, args.lr_GD,
                                                                args.x_init, args.num_look,
                                                                args.mask_rate, args.kernel_size,
                                                                channel_list, args.add_std,
                                                                args.MC, args.num_ite_MC])))
    os.makedirs(out_path, exist_ok=True)
    filepaths = glob.glob(os.path.join(args.data_dir, args.dataset) + '/*.png')

    ############# training function #############
    PSNR_GD_All, SSIM_GD_All, PSNR_NN_All, SSIM_NN_All = train(out_path, filepaths, channel_list, dtype, device)

    with open(out_path + '/' + 'PSNR_GD' + '.pkl', 'wb') as psnr_GD_file:
        pickle.dump(PSNR_GD_All, psnr_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_GD' + '.pkl', 'wb') as ssim_GD_file:
        pickle.dump(SSIM_GD_All, ssim_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'PSNR_NN' + '.pkl', 'wb') as psnr_NN_file:
        pickle.dump(PSNR_NN_All, psnr_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_NN' + '.pkl', 'wb') as ssim_NN_file:
        pickle.dump(SSIM_NN_All, ssim_NN_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

