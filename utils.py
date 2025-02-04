import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import dataio
import os
from torchvision.utils import make_grid
import skimage.measure
import cv2
from torch.utils.data import Dataset
import scipy.io.wavfile as wavfile


def gen_latent_code(num_frame, num_channels, patch_size, out_ch):
    # img_meas, A_sensing are all tensor
    totalupsample = 2 ** (len(num_channels) - 1)
    # if running compressive imaging
    w = np.sqrt(int(patch_size**2 / out_ch))
    width = int(w / (totalupsample))
    height = int(w / (totalupsample))
    # (1, num_channel_init, width_init, height_init)
    # shape = [num_frame, num_channels[0], width, height]
    shape = [1, num_channels[0], width, height]
    print("shape of latent code: ", shape)
    # latent_code = nn.Parameter(torch.zeros(shape))
    latent_code = torch.zeros(shape)
    latent_code.data.normal_()
    latent_code.data *= 1. / 10
    return latent_code

def PSNR(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    # print('img1', img1)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def PSNR_video(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    PSNR_frames = []
    for frame_idx in range(int(img1.shape[0])):
        img1_frame = img1[frame_idx]
        img2_frame = img2[frame_idx]
        mse = np.mean((img1_frame - img2_frame) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        PSNR_frame = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        PSNR_frames.append(PSNR_frame)
    PSNR_avg = np.mean(np.array(PSNR_frames))
    return PSNR_avg

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_result_img(experiment_name, filename, img):
    root_path = '/media/data1/sitzmann/generalization/results'
    trgt_dir = os.path.join(root_path, experiment_name)

    img = img.detach().cpu().numpy()
    np.save(os.path.join(trgt_dir, filename), img)


def densely_sample_activations(model, num_dim=1, num_steps=int(1e6)):
    input = torch.linspace(-1., 1., steps=num_steps).float()

    if num_dim == 1:
        input = input[...,None]
    else:
        input = torch.stack(torch.meshgrid(*(input for _ in num_dim)), dim=-1).view(-1, num_dim)

    input = {'coords':input[None,:].cuda()}
    with torch.no_grad():
        activations = model.forward_with_activations(input)['activations']
    return activations


def write_wave_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):

    sl = 256
    def scale_percentile(pred, min_perc=1, max_perc=99):
        min = np.percentile(pred.cpu().numpy(),1)
        max = np.percentile(pred.cpu().numpy(),99)
        pred = torch.clamp(pred, min, max)
        return (pred - min) / (max-min)

    with torch.no_grad():
        frames = [0.0, 0.05, 0.1, 0.15, 0.25]
        coords = [dataio.get_mgrid((1, sl, sl), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = f
        coords = torch.cat(coords, dim=0)

        Nslice = 10
        output = torch.zeros(coords.shape[0], coords.shape[1], 1)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    min_max_summary(prefix + 'pred', pred, writer, total_steps)
    pred = output.view(len(frames), 1, sl, sl)

    plt.switch_backend('agg')
    fig = plt.figure()
    plt.subplot(2,2,1)
    data = pred[0, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    plt.subplot(2,2,2)
    data = pred[1, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    plt.subplot(2,2,3)
    data = pred[2, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    plt.subplot(2,2,4)
    data = pred[3, :, sl//2, :].numpy().squeeze()
    plt.plot(np.linspace(-1, 1, sl), data)
    plt.ylim([-0.01, 0.02])

    writer.add_figure(prefix + 'center_slice', fig, global_step=total_steps)

    pred = torch.clamp(pred, -0.002, 0.002)
    writer.add_image(prefix + 'pred_img', make_grid(pred, scale_each=False, normalize=True),
                     global_step=total_steps)


def write_image_summary_small(image_resolution, mask, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    if mask is None:
        gt_img = dataio.lin2img(gt['img'], image_resolution)
        gt_dense = gt_img
    else:
        gt_img = dataio.lin2img(gt['img'], image_resolution) * mask
        gt_dense = gt_img

    pred_img = dataio.lin2img(model_output['model_out'], image_resolution)

    with torch.no_grad():
        img_gradient = torch.autograd.grad(model_output['model_out'], [model_output['model_in']],
                                           grad_outputs=torch.ones_like(model_output['model_out']), create_graph=True,
                                           retain_graph=True)[0]

        grad_norm = img_gradient.norm(dim=-1, keepdim=True)
        grad_norm = dataio.lin2img(grad_norm, image_resolution)
        writer.add_image(prefix + 'pred_grad_norm', make_grid(grad_norm, scale_each=False, normalize=True),
                         global_step=total_steps)

    output_vs_gt = torch.cat((gt_img, pred_img), dim=-1)
    writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    write_psnr(pred_img, gt_dense, writer, total_steps, prefix + 'img_dense_')

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_img', pred_img, writer, total_steps)
    min_max_summary(prefix + 'gt_img', gt_img, writer, total_steps)

    hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix)


def make_contour_plot(array_2d,mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode=='log'):
        num_levels = 6
        levels_pos = np.logspace(-2, 0, num=num_levels) # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels*2+1))
    elif(mode=='lin'):
        num_levels = 10
        levels = np.linspace(-.5,.5,num=num_levels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=num_levels))

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_sdf_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    slice_coords_2d = dataio.get_mgrid(512)

    with torch.no_grad():
        yz_slice_coords = torch.cat((torch.zeros_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.zeros_like(slice_coords_2d[:, :1]),
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     -0.75*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)

        min_max_summary(prefix + 'model_out_min_max', model_output['model_out'], writer, total_steps)
        min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)


def hypernet_activation_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    with torch.no_grad():
        hypo_parameters, embedding = model.get_hypo_net_weights(model_input)

        for name, param in hypo_parameters.items():
            writer.add_histogram(prefix + name, param.cpu(), global_step=total_steps)

        writer.add_histogram(prefix + 'latent_code', embedding.cpu(), global_step=total_steps)


def write_video_summary(vid_dataset, model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    resolution = vid_dataset.shape
    frames = [0, 60, 120, 200]
    Nslice = 10
    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)

        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            pred = model({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()

    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))

    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)

    output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
    writer.add_image(prefix + 'output_vs_gt', make_grid(output_vs_gt, scale_each=False, normalize=True),
                     global_step=total_steps)
    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_vid', pred_vid, writer, total_steps)
    writer.add_scalar(prefix + "psnr", psnr, total_steps)


def write_audio_summary(logging_root_path, model, model_input, gt, model_output, writer, total_steps, prefix='train'):
    gt_func = torch.squeeze(gt['func'])
    gt_rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
    gt_scale = torch.squeeze(gt['scale']).detach().cpu().numpy()
    pred_func = torch.squeeze(model_output['model_out'])
    coords = torch.squeeze(model_output['model_in'].clone()).detach().cpu().numpy()

    fig, axes = plt.subplots(3,1)

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot]
    gt_func_plot = gt_func.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_func.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    writer.add_figure(prefix + 'gt_vs_pred', fig, global_step=total_steps)

    min_max_summary(prefix + 'coords', model_input['coords'], writer, total_steps)
    min_max_summary(prefix + 'pred_func', pred_func, writer, total_steps)
    min_max_summary(prefix + 'gt_func', gt_func, writer, total_steps)

    # write audio files:
    wavfile.write(os.path.join(logging_root_path, 'gt.wav'), gt_rate, gt_func.detach().cpu().numpy())
    wavfile.write(os.path.join(logging_root_path, 'pred.wav'), gt_rate, pred_func.detach().cpu().numpy())


def min_max_summary(name, tensor, writer, total_steps):
    writer.add_scalar(name + '_min', tensor.min().detach().cpu().numpy(), total_steps)
    writer.add_scalar(name + '_max', tensor.max().detach().cpu().numpy(), total_steps)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs, ssims = list(), list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)

        trgt = (trgt / 2.) + 0.5

        ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

        psnrs.append(psnr)
        ssims.append(ssim)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)
    writer.add_scalar(prefix + "ssim", np.mean(ssims), iter)



dtype = torch.float64

def img_to_blocks(imgs, path, stride=32, filter_size=128, downsample=False):
  images_dataset = []
  for img in imgs:
    image = plt.imread(os.path.join(path,img))
    # 3 dimensions (RGB) convert to YCrCb (take only Y -> luminance)
    # downsample the image to filtersize
    if downsample:
      image = cv2.resize(image, (128, 128))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = image[:,:,0]
    # print('max', np.max(image), 'min', np.min(image))
    h,w = image.shape
    h_n = ((h - filter_size) // stride) + 1
    w_n = ((w - filter_size) // stride) + 1

    for i in range(h_n):
      for j in range(w_n):
        blocks = image[i*stride:(i*stride)+filter_size, j*stride:(j*stride)+filter_size]
        images_dataset.append(blocks)

  return np.array(images_dataset)

class TrainDataset(Dataset):
  def __init__(self, data_dir, stride_step, patch_size, num_look, device, transform=None, downsample=False, norm_noisy_img=False, noise_type='speckle', X_init='sq_avg_sqrt', variance=1.0):
    self.data_dir = os.listdir(data_dir)
    self.transform = transform
    self.image_blocks = img_to_blocks(self.data_dir, data_dir, stride_step, patch_size, downsample)
    self.patch_size = patch_size
    self.device = device
    self.norm_noisy_img = norm_noisy_img
    self.num_look = num_look
    self.noise_type = noise_type
    self.X_init = X_init
    self.variance = torch.Tensor([variance]).to(self.device)

  def __len__(self):
    return len(self.image_blocks)

  def __getitem__(self, idx):
    img_clean = self.image_blocks[idx]
    if self.transform is not None:
      img_clean = self.transform(img_clean)
    img_clean = img_clean.type(torch.float64).to(self.device)
    with torch.no_grad():

      w_noise_0 = torch.randn(1, self.patch_size, self.patch_size).type(torch.float64).to(self.device)
      if self.noise_type == 'speckle':
        img_noisy_L = torch.mul(img_clean, torch.sqrt(self.variance) * w_noise_0)
      elif self.noise_type == 'additive':
        img_noisy_L = torch.add(img_clean, torch.sqrt(self.variance) * w_noise_0)
      img_noisy_L = img_noisy_L.unsqueeze(1)

      for l in range(self.num_look-1):
        w_noise_l = torch.randn(1, self.patch_size, self.patch_size).type(torch.float64).to(self.device)
        if self.noise_type == 'speckle':
          img_noisy_l = torch.mul(img_clean, torch.sqrt(self.variance) * w_noise_l)
        elif self.noise_type == 'additive':
          img_noisy_l = torch.add(img_clean, torch.sqrt(self.variance) * w_noise_l)
        img_noisy_l = img_noisy_l.unsqueeze(1)
        img_noisy_L = torch.cat((img_noisy_L, img_noisy_l), dim=1)

      # Init the L-look input
      if self.noise_type == 'speckle':
        if self.X_init == 'abs_avg':
          img_noisy_L = torch.abs(img_noisy_L)
          img_noisy = torch.mean(img_noisy_L, dim=1)
        elif self.X_init == 'sq_avg_sqrt':
          img_noisy_L = torch.square(img_noisy_L)
          img_noisy = torch.sqrt(torch.mean(img_noisy_L, dim=1))
      elif self.noise_type == 'additive':
        img_noisy = torch.mean(img_noisy_L, dim=1)

      if self.norm_noisy_img:
        img_noisy = torch.abs(img_noisy) / torch.max(torch.abs(img_noisy))

    return img_noisy, img_clean

class TrainDataset_BNoise(Dataset):
  def __init__(self, data_dir, stride_step, patch_size, num_look, device, transform=None, downsample=False, norm_noisy_img=False, noise_type='speckle', variance=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]):
    self.data_dir = os.listdir(data_dir)
    self.transform = transform
    self.image_blocks = img_to_blocks(self.data_dir, data_dir, stride_step, patch_size, downsample)
    self.patch_size = patch_size
    self.device = device
    self.norm_noisy_img = norm_noisy_img
    self.num_look = num_look
    self.noise_type = noise_type
    self.variance = variance

  def __len__(self):
    return len(self.image_blocks)

  def __getitem__(self, idx):
    img_clean = self.image_blocks[idx]
    if self.transform is not None:
      img_clean = self.transform(img_clean)
    img_clean = img_clean.type(torch.float64).to(self.device)
    with torch.no_grad():
      w_noise_0 = torch.randn(1, self.patch_size, self.patch_size).type(torch.float64).to(self.device)
      if self.noise_type == 'speckle':
        img_noisy_L = torch.mul(img_clean, w_noise_0)
      elif self.noise_type == 'additive':
        noise_idx = int(np.random.choice(6, 1))
        variance_choice_ = self.variance[noise_idx]
        variance_choice = torch.Tensor([variance_choice_]).to(self.device)
        img_noisy_L = torch.add(img_clean, torch.sqrt(variance_choice) * w_noise_0)
      img_noisy_L = img_noisy_L.unsqueeze(1)
      for l in range(self.num_look-1):
        w_noise_l = torch.randn(1, self.patch_size, self.patch_size).type(torch.float64).to(self.device)
        if self.noise_type == 'speckle':
          img_noisy_l = torch.mul(img_clean, w_noise_l)
        elif self.noise_type == 'additive':
          img_noisy_l = torch.add(img_clean, torch.sqrt(variance_choice) * w_noise_l)
        img_noisy_l = img_noisy_l.unsqueeze(1)
        img_noisy_L = torch.cat((img_noisy_L, img_noisy_l), dim=1)
      if self.noise_type == 'speckle':
        img_noisy_L = torch.abs(img_noisy_L)
      img_noisy = torch.mean(img_noisy_L, dim=1)
      if self.norm_noisy_img:
        img_noisy = torch.abs(img_noisy) / torch.max(torch.abs(img_noisy))

    return img_noisy, img_clean

def rgb2ycbcr(rgb):
  m = np.array([[65.481, 128.553, 24.966],
                [-37.797, -74.203, 112],
                [112, -93.786, -18.214]])
  shape = rgb.shape
  if len(shape) == 3:
    rgb = rgb.reshape((shape[0] * shape[1], 3))
  ycbcr = np.dot(rgb, m.transpose() / 255.)
  ycbcr[:, 0] += 16.
  ycbcr[:, 1:] += 128.
  return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
  m = np.array([[65.481, 128.553, 24.966],
                [-37.797, -74.203, 112],
                [112, -93.786, -18.214]])
  shape = ycbcr.shape
  if len(shape) == 3:
    ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
  rgb = copy.deepcopy(ycbcr)
  rgb[:, 0] -= 16.
  rgb[:, 1:] -= 128.
  rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
  return rgb.clip(0, 255).reshape(shape)


def imread_CS_py(Iorg, block_size):
  [row, col] = Iorg.shape
  row_pad = block_size - np.mod(row, block_size)
  col_pad = block_size - np.mod(col, block_size)
  if col_pad == block_size:
    Ipad = Iorg
    if row_pad == block_size:
      Ipad = Ipad
    else:
      Ipad = np.concatenate((Ipad, np.zeros([row_pad, col])), axis=0)
  else:
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    if row_pad == block_size:
      Ipad = Ipad
    else:
      Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)

  [row_new, col_new] = Ipad.shape

  return [Iorg, row, col, Ipad, row_new, col_new]


def imread_CS_py_(Iorg, block_size):
  [row, col] = Iorg.shape
  row_pad = block_size - np.mod(row, block_size)
  col_pad = block_size - np.mod(col, block_size)
  Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
  Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
  [row_new, col_new] = Ipad.shape

  return [Iorg, row, col, Ipad, row_new, col_new]

def img2col_py(Ipad, block_size):
  [row, col] = Ipad.shape
  row_block = row / block_size
  col_block = col / block_size
  block_num = int(row_block * col_block)
  img_col = np.zeros([block_size ** 2, block_num])
  count = 0
  for x in range(0, row - block_size + 1, block_size):
    for y in range(0, col - block_size + 1, block_size):
      img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
      # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
      count = count + 1
  return img_col

def img2col_py_(Ipad, block_size):
  [row, col] = Ipad.shape
  row_block = row / block_size
  col_block = col / block_size
  block_num = int(row_block * col_block)
  img_col = np.zeros([block_size ** 2, block_num])
  count = 0
  for x in range(0, row - block_size + 1, block_size):
    for y in range(0, col - block_size + 1, block_size):
      img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
      count = count + 1
  return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new, block_size):
  X0_rec = np.zeros([row_new, col_new])
  count = 0
  for x in range(0, row_new - block_size + 1, block_size):
    for y in range(0, col_new - block_size + 1, block_size):
      X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
      # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
      count = count + 1
  X_rec = X0_rec[:row, :col]
  return X_rec

def psnr(img1, img2):
  img1.astype(np.float32)
  img2.astype(np.float32)
  mse = np.mean((img1 - img2) ** 2)
  # print('img1', img1)
  if mse == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def num_param(net):
  s = sum([np.prod(list(p.size())) for p in net.parameters()]);
  return s

def gen_latent_code_patch(batch_size, patch_size, num_channels, out_ch):
  # img_meas, A_sensing are all tensor
  totalupsample = 2 ** (len(num_channels) - 1)
  # if running as decoder/compressor
  width, height = 0, 0

  w = patch_size
  width = int(w / (totalupsample))
  height = int(w / (totalupsample))

  # (1, num_channel_init, width_init, height_init)
  shape = [batch_size, num_channels[0], width, height]
  # print("shape of latent code: ", shape)
  # latent_code = nn.Parameter(torch.zeros(shape))
  latent_code = torch.zeros(shape)
  latent_code.data.normal_()
  latent_code.data *= 1. / 10
  return latent_code

def gen_latent_code_MSE(batch_size, num_channels, A_sensing, out_ch):
  # img_meas, A_sensing are all tensor
  totalupsample = 2 ** (len(num_channels) - 1)
  # if running as decoder/compressor
  width, height = 0, 0
  w = np.sqrt(int(A_sensing.shape[1] / out_ch))
  width = int(w / (totalupsample))
  height = int(w / (totalupsample))

  # (1, num_channel_init, width_init, height_init)
  shape = [batch_size, num_channels[0], width, height]
  print("shape of latent code: ", shape)
  # latent_code = nn.Parameter(torch.zeros(shape))
  latent_code = torch.zeros(shape)
  latent_code.data.normal_()
  latent_code.data *= 1. / 10
  return latent_code

def generate_gaussian_matrix(m, n, use_complex, dtype):
  if use_complex:
    a1 = torch.rand(m, n).type(dtype)
    a = torch.complex(a1, a1)
  else:
    a = torch.rand(m, n).type(dtype)
  return a

def generate_orthogonal_matrix(m, n, use_complex, diff_A, dtype):
  if use_complex:
    a1 = torch.rand(n, m).type(dtype)
    if diff_A:
      a2 = torch.rand(n, m).type(dtype)
    else:
      a2 = a1
    a = torch.complex(a1, a2)
  else:
    a = torch.rand(n, m).type(dtype)
    # a = np.random.random(size=(n, m))
    # a = np.random.randn(n, m)
  # q, _ = np.linalg.qr(a)
  q, _ = torch.linalg.qr(a)
  return q.T
