
import argparse
import torch
import numpy as np
import os
import torchvision
from PIL import Image
from backbones.ncsnpp_generator_adagn import NCSNpp
from dataset import GetDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
        
#%% Diffusion coefficients
def var_func_affine_logistic(t, vp_max, vp_k1, vp_k2):
    vp_k1 = torch.tensor(vp_k1)
    vp_k2 = torch.tensor(vp_k2)
    log_mean_coeff = -vp_max / vp_k1 * (torch.log(torch.exp(vp_k1 * t - vp_k2) + 1.) - torch.log(torch.exp(-vp_k2) + 1))
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_sigma_schedule(device):
    eps_small = 1e-3
    t = np.arange(0, args.vp_t + 1, dtype=np.float64)
    t = t / args.vp_t
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    var = var_func_affine_logistic(t, args.vp_max, args.vp_k, args.vp_k)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, device):
        _, _, self.betas = get_sigma_schedule(device=device)

        self.betas = self.betas.type(torch.float32)[1:]
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

def sample_posterior(coefficients, x_0, x_t, t):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        prior = x_0 - mean
        unit_prior = torch.nn.functional.normalize(prior, p=2, dim=1)
        unit_noise = torch.nn.functional.normalize(torch.randn_like(x_t), p=2, dim=1)
        noise = unit_noise * args.vp_noise + unit_prior * args.vp_prior
        if args.sample_fixed:
            noise = 0.
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos

def random_position_ids(n, l=40):
    return np.sort(np.random.permutation(l)[:n])

def sample_from_model(coefficients, mapping_model, x_T, source, args):
    x_t = x_T
    with torch.no_grad():
        for t in reversed(range(args.vp_t)):
            t_module = torch.full((x_t.size(0),), args.vp_sparse * t, dtype=torch.int64).to(x_t.device)
            t_time = torch.full((x_t.size(0),), t, dtype=torch.int64).to(x_t.device)
            latent_z = torch.zeros(x_t.size(0), args.nz, device=x_t.device) if args.sample_fixed else torch.randn(x_t.size(0), args.nz, device=x_t.device)
            h_t = mapping_model(torch.cat((x_t, source),axis=1), t_module, latent_z)
            h_t = (h_t - args.AO) / args.MO
            x_td1 = sample_posterior(coefficients, h_t, x_t, t_time)
            x_t = x_td1.detach()

    return x_t

def load_checkpoint(checkpoint_dir, mapping_network, name_of_network, epoch,device = 'cuda:0'):
    checkpoint_file = checkpoint_dir.format(name_of_network, epoch)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    ckpt = checkpoint
    for key in list(ckpt.keys()):
         ckpt[key[7:]] = ckpt.pop(key)
    mapping_network.load_state_dict(ckpt)
    mapping_network.eval()

def evaluate_samples(real_data, fake_sample, input_channels):
    to_range_0_1 = lambda x: (x + 1.) / 2.
    real_data = real_data.cpu().numpy()
    fake_sample = fake_sample.cpu().numpy()
    psnr_list = []
    ssim_list = []
    mae_list = []
    for i in range(real_data.shape[0]):
        real_data_i = real_data[i]
        fake_sample_i = fake_sample[i]
        real_data_i = to_range_0_1(real_data_i) / real_data_i.max()
        fake_sample_i = to_range_0_1(fake_sample_i) / fake_sample_i.max()
        psnr_val = psnr(real_data_i, fake_sample_i, data_range=real_data_i.max() - real_data_i.min())
        mae_val = np.mean(np.abs(real_data_i - fake_sample_i))
        if input_channels == 1:
            ssim_val = ssim(real_data_i[0], fake_sample_i[0], data_range=real_data_i.max() - real_data_i.min())
        elif input_channels == 3:
            real_data_i = np.squeeze(real_data_i).transpose(1, 2, 0)
            fake_sample_i = np.squeeze(fake_sample_i).transpose(1, 2, 0)
            ssim_val = ssim(real_data_i, fake_sample_i, channel_axis=-1, data_range=real_data_i.max() - real_data_i.min())
        else:
            raise ValueError("Unsupported number of input channels")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val * 100)
        mae_list.append(mae_val)
    return psnr_list, ssim_list, mae_list

def save_image(img, save_dir, phase, iteration, input_channels):
    file_path = '{}/{}({}).png'.format(save_dir, phase, str(iteration).zfill(4))
    to_range_0_1 = lambda x: (x + 1.) / 2.
    img = to_range_0_1(img)
    if input_channels == 1:
        torchvision.utils.save_image(img, file_path)
    elif input_channels == 3:
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 127.5 + 127.5).astype(np.uint8)[..., [2, 1, 0]]
        image = Image.fromarray(img)
        image.save(file_path)


#%% MAIN FUNCTION
def sample_and_test(args):
    torch.manual_seed(42)

    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen=args.which_epoch

    test_dataset = GetDataset("test", args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mapping_model = NCSNpp(args).to(device)

    checkpoint_file = args.checkpoint_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, mapping_model,'EDDM', epoch=str(epoch_chosen), device = device)

    pos_coeff = Posterior_Coefficients(device)
         
    save_dir = args.checkpoint_path + "/generated_samples/EDDM_EPOCH({})".format(epoch_chosen)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    PSNR = []
    SSIM = []
    MAE = []
    image_iteration = 0

    for iteration, (source_data,  target_data) in enumerate(test_dataloader):
        
        target_data = target_data.to(device, non_blocking=True)
        source_data = source_data.to(device, non_blocking=True)

        if args.input_channels == 3:
            target_data = target_data.squeeze(1)
            source_data = source_data.squeeze(1)

        x_T = torch.randn_like(target_data)
        fake_sample = sample_from_model(pos_coeff, mapping_model, x_T, source_data, args)

        psnr_list, ssim_list, mae_list = evaluate_samples(target_data, fake_sample, args.input_channels)
        PSNR.extend(psnr_list)
        SSIM.extend(ssim_list)
        MAE.extend(mae_list)
        print(f"[{iteration}/{len(test_dataloader)}]," + str(psnr_list[0]) + "," + str(ssim_list[0]) + "," + str(mae_list[0]))
        for i in range(fake_sample.shape[0]):
            save_image(fake_sample[i], save_dir, args.phase, image_iteration, args.input_channels)
            image_iteration = image_iteration + 1

    print('TEST PSNR: mean:' + str(sum(PSNR) / len(PSNR)) + ' max:' + str(max(PSNR)) + ' min:' + str(min(PSNR)) + ' var:' + str(np.var(PSNR)))
    print('TEST SSIM: mean:' + str(sum(SSIM) / len(SSIM)) + ' max:' + str(max(SSIM)) + ' min:' + str(min(SSIM)) + ' var:' + str(np.var(SSIM)))
    print('TEST MAE: mean:' + str(sum(MAE) / len(MAE)) + ' max:' + str(max(MAE)) + ' min:' + str(min(MAE)) + ' var:' + str(np.var(MAE) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EDDM parameters')
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')

    #mapping_model
    parser.add_argument('--num_channels_dae', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True, help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'], help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'], help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'], help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true',default=False)

    #training
    parser.add_argument('--input_channels', type=int, default=1, help='channel of image')
    parser.add_argument('--input_path', help='path to input data')
    parser.add_argument('--checkpoint_path', help='path to exp/checkpoint saves')
    parser.add_argument('--dataset', default='BrainTs20', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=256, help='size of image')
    parser.add_argument('--normed', action='store_true', default=False)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=2, help='sample generating batch size')

    #parameters
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--source', type=str, default='T1', help='contrast selection for model')
    parser.add_argument('--target', type=str, default='T2', help='contrast selection for model')
    parser.add_argument('--which_epoch', type=int, default=120)
    parser.add_argument('--gpu_chose', type=int, default=0)

    #tuning
    parser.add_argument('--phase', type=str, default='test', help='model train, tuning or test')
    parser.add_argument('--sample_fixed', action='store_true', default=False)
    parser.add_argument('--vp_t', type=int, default=4)
    parser.add_argument('--vp_max', type=float, default=20)
    parser.add_argument('--vp_k', type=float, default=5)
    parser.add_argument('--vp_sparse', type=float, default=1)
    parser.add_argument('--vp_noise', type=float, default=1)
    parser.add_argument('--vp_prior', type=float, default=0)
    parser.add_argument('--MO', type=float, default=1)
    parser.add_argument('--AO', type=float, default=0)
    args = parser.parse_args()
    
    sample_and_test(args)
    
