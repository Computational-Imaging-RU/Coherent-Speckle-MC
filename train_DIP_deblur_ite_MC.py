import argparse
import torch
import numpy as np
import time
import pickle
import random
from model import *
from dataio import *
import utils
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from function_grad import *
from decoder import autoencodernet
from scipy.fft import dct, idct, dctn, idctn

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--dataset', type=str, default='Set11', help='test dataset')
parser.add_argument('--seed', type=int, default=312, help='name of test set')
parser.add_argument('--kernel_size1', type=int, default=3, help='kernel size in DIP')
parser.add_argument('--kernel_size2', type=int, default=3, help='kernel size in DIP')
parser.add_argument('--kernel_size3', type=int, default=1, help='kernel size in DIP')
parser.add_argument('--degradation', type=str, default='blur', help='forward operator')
parser.add_argument('--mask_const_en', type=float, default=0, help='Initial learning rate.')
parser.add_argument('--mask_rate', type=float, default=0.8, help='name of test set')
parser.add_argument('--x_init', type=str, default='constant', help='forward operator')
parser.add_argument('--decodetype', type=str, default='upsample', help='upsample, transposeconv')
parser.add_argument('--num_look', type=int, default=1, help='number of looks')
parser.add_argument('--crop', type=bool, default=True, help='crop the raw image')
parser.add_argument('--lr_NN', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--lr_GD', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for NN training.')
parser.add_argument('--outer_ite', type=int, default=100, help='name of test set')
parser.add_argument('--inner_ite', type=int, default=200, help='name of test set')
parser.add_argument('--num_ite_MC', type=int, default=2, help='num ite in MC')
parser.add_argument('--out_nonlinear', type=bool, default=True, help='crop the raw image')
parser.add_argument('--add_std', type=float, default=0.2, help='std of additive noise.')
parser.add_argument('--MC', type=bool, default=True, help='use Newton Schulz to approximate matrix inverse')
args = parser.parse_args()
print(args)

def create_mask_en(n, frac):
    mask = torch.zeros(n, n, dtype=torch.float64) + args.mask_const_en
    mask_idx = int(n * np.sqrt(frac))
    mask[:mask_idx, :mask_idx] = 1
    print('mask idx', mask_idx)
    return mask

def train(out_path, filepaths, inner_ite_list_1, channel_list, dtype, device):

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
        inner_ite_list = inner_ite_list_1
        print('image name:', imgName)
        print('inner ites:', inner_ite_list)
        print('channel list', channel_list)

        ########## Crop the image into patches ##########
        Img = cv2.imread(imgName, 1)
        patch_size = np.shape(Img)[0]
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb) / 255.0
        Iorg_y = np.expand_dims(Img_yuv[:, :, 0], axis=0)
        vid_dataset = Video_bee(Iorg_y)

        ########## init the INR model ##########
        coord_dataset = Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=1.0, frame_num=1, device=device, dtype=dtype)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
        for step, (model_input, gt) in enumerate(dataloader):
            model_input = {key: value.type(dtype).to(device) for key, value in model_input.items()}
            gt = {key: value.type(dtype).to(device) for key, value in gt.items()}

        img_gt_ = (gt['img'] / 2 + 0.5)
        img_gt = img_gt_.view(patch_size, patch_size)
        img_gt = img_gt.detach().cpu().numpy()
        cv2.imwrite(os.path.join(out_path, "%s_raw.png" % (single_imgName)), np.clip(img_gt, 0, 1)*255.0)

        ########## generate the blurred (multi-look) measurements ##########
        img_blur = np.zeros((args.num_look, patch_size, patch_size))
        mask_matrix_en = create_mask_en(patch_size, args.mask_rate).detach().cpu().numpy()
        for look_idx in range(args.num_look):
            w_noise = torch.randn(gt['img'].size(dim=0), gt['img'].size(dim=1), gt['img'].size(dim=2)).type(dtype).to(device)
            xw = torch.mul((gt['img'] / 2 + 0.5), w_noise).squeeze(0)
            # xw = (gt['img'] / 2 + 0.5).squeeze(0)# (L,n) # no speckle
            xw = xw.view(patch_size, patch_size)
            xw_arr = xw.detach().cpu().numpy()
            if args.degradation == 'blur':
                img_dct = dctn(xw_arr, norm="ortho")
                img_dct_mask = img_dct * mask_matrix_en
                img_blur_idx = idctn(img_dct_mask, norm="ortho")
                add_noise = np.random.normal(loc=0.0, scale=args.add_std, size=(patch_size, patch_size))
                img_blur_idx += add_noise
                Axw_z_arr = img_blur_idx
                img_blur[look_idx] = Axw_z_arr

                x_init = img_blur[look_idx]
                psnr_x_init = utils.PSNR(np.clip(x_init, 0, 1) * 255.0, img_gt * 255.0)
                ssim_x_init = ssim(np.clip(x_init, 0, 1) * 255.0, img_gt * 255.0, data_range=255)
                cv2.imwrite(os.path.join(out_path, "%s_y_blur_PSNR_%.2f_SSIM_%.4f.png" % (single_imgName, psnr_x_init, ssim_x_init)), np.clip(x_init, 0, 1) * 255.0)
                print('psnr init', psnr_x_init, 'ssim init', ssim_x_init)
                PSNR_NN_All[0, img_no] = psnr_x_init
                SSIM_NN_All[0, img_no] = ssim_x_init

                x_init_abs = np.abs(img_blur[look_idx])
                psnr_x_init_abs = utils.PSNR(np.clip(x_init_abs, 0, 1) * 255.0, img_gt * 255.0)
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
            psnr_GD = utils.PSNR(x_G_save, img_gt * 255.0)
            ssim_GD = ssim(x_G_save, img_gt * 255.0, data_range=255)
            print('psnr GD', psnr_GD, 'ssim GD', ssim_GD)
            GD_end_time = time.time()

            ######### projection step: train DIP/Deep Decoder ##########
            projection_start = time.time()
            x_raw = torch.from_numpy(x_G).type(dtype).to(device)
            x_raw_ = x_raw.view(patch_size, patch_size)
            x_raw_np = x_raw_.detach().cpu().numpy().copy()
            DIP_patch_size_list = [64,128,patch_size]

            # DIP_patch_size = DIP_patch_size_list[0]
            # num_patch = int((args.patch_size / DIP_patch_size) ** 2)
            # [Iorg_raw, row_raw, col_raw, Ipad_raw, row_new_raw, col_new_raw] = utils.imread_CS_py(x_raw_np, DIP_patch_size)
            # x_raw_np_patch = utils.img2col_py(Ipad_raw, DIP_patch_size).transpose()
            # x_raw_tensor_patch = torch.from_numpy(x_raw_np_patch).type(dtype).to(device)
            # x_raw_patch = x_raw_tensor_patch.view(-1, 1, DIP_patch_size, DIP_patch_size)
            # x_gen_patch = torch.zeros_like(x_raw_tensor_patch)
            # for DIP_i in range(num_patch):
            #     output_depth = 1  # number of output channels (gray scale image)
            #     net_1 = autoencodernet(num_output_channels=output_depth, num_channels_up=channel_list, need_sigmoid=args.out_nonlinear, decodetype=args.decodetype, kernel_size=args.kernel_size1).type(dtype).to(device)
            #     latent_code_1 = utils.gen_latent_code_patch(1, DIP_patch_size, channel_list, 1).type(dtype).to(device)
            #     params_1 = [x for x in net_1.decoder.parameters()]
            #     optimizer_1 = torch.optim.Adam(params_1, lr=args.lr_NN, weight_decay=args.weight_decay)
            #     x_raw_i = x_raw_patch[DIP_i]
            #     x_raw_i_flatten = x_raw_i.view(-1, DIP_patch_size ** 2)
            #     for ee in range(inner_ite_list[0]):
            #     # for ee in range(inner_ite):
            #         net_1.train()
            #         optimizer_1.zero_grad()
            #         x_gen_tensor_i_ = net_1(latent_code_1)
            #         x_gen_tensor_i = x_gen_tensor_i_.view(-1, DIP_patch_size ** 2)
            #         loss_train = F.mse_loss(x_gen_tensor_i, x_raw_i_flatten)
            #         loss_train.backward()
            #         optimizer_1.step()
            #     with torch.no_grad():
            #         x_gen_img_i_final = net_1(latent_code_1).detach()
            #         x_gen_patch[DIP_i] = x_gen_img_i_final.view(-1, DIP_patch_size ** 2)
            # x_gen_np = x_gen_patch.detach().cpu().numpy()
            # x_gen_np_1 = utils.col2im_CS_py(x_gen_np.transpose(), row_raw, col_raw, row_new_raw, col_new_raw, DIP_patch_size)
            # x_gen_np_clip_1 = np.clip(x_gen_np_1, 0, 1)
            # x_gen_psnr_1 = utils.PSNR(x_gen_np_clip_1 * 255.0, img_gt.astype(np.float64) * 255.0)
            # print('psnr estimate 1:', x_gen_psnr_1)
            # x_gen_1 = torch.from_numpy(x_gen_np_1).type(dtype).to(device)
            # x_gen_1 = x_gen_1.view(args.patch_size, args.patch_size)
            #
            # DIP_patch_size = DIP_patch_size_list[1]
            # num_patch = int((args.patch_size / DIP_patch_size) ** 2)
            # [Iorg_raw, row_raw, col_raw, Ipad_raw, row_new_raw, col_new_raw] = utils.imread_CS_py(x_raw_np, DIP_patch_size)
            # x_raw_np_patch = utils.img2col_py(Ipad_raw, DIP_patch_size).transpose()
            # x_raw_tensor_patch = torch.from_numpy(x_raw_np_patch).type(dtype).to(device)
            # x_raw_patch = x_raw_tensor_patch.view(-1, 1, DIP_patch_size, DIP_patch_size)
            # x_gen_patch = torch.zeros_like(x_raw_tensor_patch)
            # for DIP_i in range(num_patch):
            #     output_depth = 1  # number of output channels (gray scale image)
            #     net_2 = autoencodernet(num_output_channels=output_depth, num_channels_up=channel_list, need_sigmoid=args.out_nonlinear, decodetype=args.decodetype, kernel_size=args.kernel_size2).type(dtype).to(device)
            #     latent_code_2 = utils.gen_latent_code_patch(1, DIP_patch_size,channel_list, 1).type(dtype).to(device)
            #     params_2 = [x for x in net_2.decoder.parameters()]
            #     optimizer_2 = torch.optim.Adam(params_2, lr=args.lr_NN, weight_decay=args.weight_decay)
            #     x_raw_i = x_raw_patch[DIP_i]
            #     x_raw_i_flatten = x_raw_i.view(-1, DIP_patch_size ** 2)
            #     for ee in range(inner_ite_list[1]):
            #     # for ee in range(inner_ite):
            #         net_2.train()
            #         optimizer_2.zero_grad()
            #         x_gen_tensor_i_ = net_2(latent_code_2)
            #         x_gen_tensor_i = x_gen_tensor_i_.view(-1, DIP_patch_size ** 2)
            #         loss_train = F.mse_loss(x_gen_tensor_i, x_raw_i_flatten)
            #         loss_train.backward()
            #         optimizer_2.step()
            #     with torch.no_grad():
            #         x_gen_img_i_final = net_2(latent_code_2).detach()
            #         x_gen_patch[DIP_i] = x_gen_img_i_final.view(-1, DIP_patch_size ** 2)
            # x_gen_np = x_gen_patch.detach().cpu().numpy()
            # x_gen_np_2 = utils.col2im_CS_py(x_gen_np.transpose(), row_raw, col_raw, row_new_raw, col_new_raw, DIP_patch_size)
            # x_gen_np_clip_2 = np.clip(x_gen_np_2, 0, 1)
            # x_gen_psnr_2 = utils.PSNR(x_gen_np_clip_2 * 255.0, img_gt.astype(np.float64) * 255.0)
            # print('psnr estimate 2:', x_gen_psnr_2)
            # x_gen_2 = torch.from_numpy(x_gen_np_2).type(dtype).to(device)
            # x_gen_2 = x_gen_2.view(args.patch_size, args.patch_size)

            DIP_patch_size = DIP_patch_size_list[2]
            num_patch = int((patch_size / DIP_patch_size) ** 2)
            [Iorg_raw, row_raw, col_raw, Ipad_raw, row_new_raw, col_new_raw] = utils.imread_CS_py(x_raw_np, DIP_patch_size)
            x_raw_np_patch = utils.img2col_py(Ipad_raw, DIP_patch_size).transpose()
            x_raw_tensor_patch = torch.from_numpy(x_raw_np_patch).type(dtype).to(device)
            x_raw_patch = x_raw_tensor_patch.view(-1, 1, DIP_patch_size, DIP_patch_size)
            x_gen_patch = torch.zeros_like(x_raw_tensor_patch)
            for DIP_i in range(num_patch):
                output_depth = 1  # number of output channels (gray scale image)
                net_3 = autoencodernet(num_output_channels=output_depth, num_channels_up=channel_list, need_sigmoid=args.out_nonlinear, decodetype=args.decodetype, kernel_size=args.kernel_size3).type(dtype).to(device)
                latent_code_3 = utils.gen_latent_code_patch(1, DIP_patch_size, channel_list, 1).type(dtype).to(device)
                params_3 = [x for x in net_3.decoder.parameters()]
                optimizer_3 = torch.optim.Adam(params_3, lr=args.lr_NN, weight_decay=args.weight_decay)
                x_raw_i = x_raw_patch[0]
                x_raw_i_flatten = x_raw_i.view(-1, DIP_patch_size ** 2)
                for ee in range(inner_ite_list[2]):
                # for ee in range(inner_ite):
                    net_3.train()
                    optimizer_3.zero_grad()
                    x_gen_tensor_i_ = net_3(latent_code_3)
                    x_gen_tensor_i = x_gen_tensor_i_.view(-1, DIP_patch_size ** 2)
                    loss_train = F.mse_loss(x_gen_tensor_i, x_raw_i_flatten)
                    loss_train.backward()
                    optimizer_3.step()
                with torch.no_grad():
                    x_gen_img_i_final = net_3(latent_code_3).detach()
                    x_gen_patch += x_gen_img_i_final.view(-1, DIP_patch_size ** 2)
            x_gen_np = x_gen_patch.detach().cpu().numpy()
            x_gen_np_3 = utils.col2im_CS_py(x_gen_np.transpose(), row_raw, col_raw, row_new_raw, col_new_raw, DIP_patch_size)
            # x_gen_np_clip_3 = np.clip(x_gen_np_3, 0, 1)
            # x_gen_psnr_3 = utils.PSNR(x_gen_np_clip_3 * 255.0, img_gt.astype(np.float64) * 255.0)
            # print('psnr estimate 3:', x_gen_psnr_3)
            x_gen_3 = torch.from_numpy(x_gen_np_3).type(dtype).to(device)
            x_gen_3 = x_gen_3.view(patch_size, patch_size)

            # x_gen = (x_gen_1 + x_gen_2 + x_gen_3) / 3
            # x_gen = (x_gen_1 + x_gen_2) / 2
            x_gen = x_gen_3

            projection_end = time.time()

            with torch.no_grad():
                # net.eval()
                x_gen = x_gen.detach().cpu().numpy()
                x_new = x_gen

                x_gen_save = np.clip(x_gen, 0, 1) * 255.0
                psnr_NN = utils.PSNR(x_gen_save, img_gt * 255.0)
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

    inner_ite_list_1 = [args.inner_ite, args.inner_ite, args.inner_ite]
    channel_list = [100,50,25,10]
    # channel_list = [128, 128, 128, 128]

    ############# testing data and saving path #############
    out_path = os.path.join('./results_DIP_deblur_MC_CGD', "_".join(map(str, [args.dataset, args.degradation,
                                                                        args.mask_const_en, args.weight_decay,
                                                                        args.outer_ite, args.inner_ite,
                                                                        args.lr_NN, args.lr_GD,
                                                                        args.x_init, args.num_look,
                                                                        args.mask_rate, inner_ite_list_1,
                                                                        args.kernel_size1, args.kernel_size2, args.kernel_size3,
                                                                        channel_list, args.add_std,
                                                                        args.MC, args.num_ite_MC])))
    os.makedirs(out_path, exist_ok=True)
    filepaths = glob.glob(os.path.join(args.data_dir, args.dataset) + '/*.png')

    ############# training function #############
    PSNR_GD_All, SSIM_GD_All, PSNR_NN_All, SSIM_NN_All = train(out_path, filepaths, inner_ite_list_1, channel_list, dtype, device)

    with open(out_path + '/' + 'PSNR_GD' + '.pkl', 'wb') as psnr_GD_file:
        pickle.dump(PSNR_GD_All, psnr_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_GD' + '.pkl', 'wb') as ssim_GD_file:
        pickle.dump(SSIM_GD_All, ssim_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'PSNR_NN' + '.pkl', 'wb') as psnr_NN_file:
        pickle.dump(PSNR_NN_All, psnr_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_NN' + '.pkl', 'wb') as ssim_NN_file:
        pickle.dump(SSIM_NN_All, ssim_NN_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')

