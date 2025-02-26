
import argparse
import torch
import numpy as np
from torch.utils.data import Subset
from tqdm import tqdm

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
        noise = torch.randn_like(x_t)
        unit_noise = noise / np.linalg.norm(noise[0][0].cpu().detach().numpy().flatten())
        noise = unit_noise * args.vp_noise + unit_prior * args.vp_prior
        nonzero_mask = (1 - (t == 0).type(torch.float32))
        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

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
            latent_z = torch.zeros(x_t.size(0), args.nz, device=x_t.device)
            h_t = mapping_model(torch.cat((x_t, source),axis=1), t_module, latent_z)
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

def evaluate_samples(real_data, fake_sample):
    to_range_0_1 = lambda x: (x + 1.) / 2.
    real_data = real_data.cpu().numpy()
    fake_sample = fake_sample.cpu().numpy()
    psnr_list = []
    ssim_list = []
    mae_list = []
    for i in range(real_data.shape[0]):
        real_data_i = real_data[i]
        fake_sample_i = fake_sample[i]
        real_data_i = to_range_0_1(real_data_i)
        real_data_i = real_data_i / real_data_i.max()
        fake_sample_i = to_range_0_1(fake_sample_i)
        fake_sample_i = fake_sample_i / fake_sample_i.max()
        psnr_val = psnr(real_data_i, fake_sample_i, data_range=real_data_i.max() - real_data_i.min())
        mae_val = np.mean(np.abs(real_data_i - fake_sample_i))
        if args.input_channels == 1:
            ssim_val = ssim(real_data_i[0], fake_sample_i[0], data_range=real_data_i.max() - real_data_i.min())
        elif args.input_channels == 3:
            real_data_i = np.squeeze(real_data_i).transpose(1, 2, 0)
            fake_sample_i = np.squeeze(fake_sample_i).transpose(1, 2, 0)
            ssim_val = ssim(real_data_i, fake_sample_i, channel_axis=-1, data_range=real_data_i.max() - real_data_i.min())
        else:
            raise ValueError("Unsupported number of input channels")
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val * 100)
        mae_list.append(mae_val)
    return psnr_list, ssim_list, mae_list

def sample_b2(dataloader, mapping_model, tuning_dict, device):
    args.vp_t = tuning_dict['vp_t']
    args.vp_k = tuning_dict['vp_k']
    args.vp_max = tuning_dict['vp_max']
    args.vp_sparse = tuning_dict['vp_sparse']
    args.vp_noise = tuning_dict['vp_noise']
    args.vp_prior = tuning_dict['vp_prior']
    pos_coeff = Posterior_Coefficients(device)
    PSNR = []
    SSIM = []
    MAE = []
    progress_bar = tqdm(dataloader, desc="Processing", colour='green')
    for iteration, (source_data, target_data, _, _) in enumerate(progress_bar):
        target_data = target_data.to(device, non_blocking=True)
        source_data = source_data.to(device, non_blocking=True)
        if args.input_channels == 3:
            target_data = target_data.squeeze(1)
            source_data = source_data.squeeze(1)
        x_T = torch.randn_like(target_data)
        fake_sample = sample_from_model(pos_coeff, mapping_model, x_T, source_data, args)
        psnr_list, ssim_list, mae_list = evaluate_samples(target_data, fake_sample)
        PSNR.extend(psnr_list)
        SSIM.extend(ssim_list)
        MAE.extend(mae_list)

    vv = 0
    mean_p = np.mean(PSNR)
    std_p = np.std(PSNR)
    mean_s = np.mean(SSIM)
    std_s = np.std(SSIM)
    mean_m = np.mean(MAE)
    std_m = np.std(MAE)
    if args.use_vv:
        if mean_p == 0 or mean_s == 0 or mean_m == 0:
            assert False
        vv = args.lambda_vp * std_p / mean_p + args.lambda_vs * std_s / mean_s + args.lambda_vm * std_m / mean_m
    v = args.lambda_p * mean_p + args.lambda_s * mean_s + args.lambda_m * mean_m
    return v, vv

def is_n_consecutive_decreasing(records, n):
    return len(records) > n and all(records[i] >= records[i + 1] for i in range(-n, -1))

#%% MAIN FUNCTION
def sample_and_test(args):
    torch.manual_seed(42)

    torch.cuda.set_device(args.gpu_chose)
    device = torch.device('cuda:{}'.format(args.gpu_chose))
    epoch_chosen=args.which_epoch

    train_dataset = GetDataset("train", args.input_path, args.source, args.target, dim=args.input_channels, normed=args.normed)
    tuning_indices = np.random.choice(len(train_dataset), args.tuning_dataset_num, replace=False)
    tuning_dataset = Subset(train_dataset, tuning_indices)
    tuning_dataloader = torch.utils.data.DataLoader(tuning_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mapping_model = NCSNpp(args).to(device)

    checkpoint_file = args.checkpoint_path + "/{}_{}.pth"
    load_checkpoint(checkpoint_file, mapping_model, '{}_EDDM'.format(args.network_type), epoch=str(epoch_chosen), device = device)

    omega_dict = {
        'vp_t': [4, 5, 6, 7, 8, 9, 10],  # tuning vp_t
        'vp_k': [3, 4, 5, 6, 7, 8],  # tuning vp_k
        'vp_max': [17.5, 20., 22.5, 25., 27.5, 30, 32.5],  # tuning vp_max
        'vp_sparse': [1, 2],  # tuning vp_sparse
        'vp_noise': [0., 50., 100., 150., 200., 250., 300., 350.],  # tuning vp_noise
        'vp_prior': [0.0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],  # tuning vp_prior
    }
    default_dict = {
        'vp_t': 4,  # default, tuning vp_t
        'vp_k': 5,  # default, tuning vp_k
        'vp_max': 20,  # default, tuning vp_max
        'vp_sparse': 1,  # default, tuning vp_sparse
        'vp_noise': 0.,  # default, tuning vp_noise
        'vp_prior': 0.,  # default, tuning vp_prior
    }
    opt_dict = default_dict.copy()
    search_length = sum(len(values) for values in omega_dict.values())
    search_count = 0

    for tuning_key, tuning_values in omega_dict.items():
        opt_v = 0.
        history_v = []
        print(" ")
        for tuning_value in tuning_values:
            tuning_dict = default_dict.copy()
            tuning_dict[tuning_key] = tuning_value
            v, vv = sample_b2(tuning_dataloader, mapping_model, tuning_dict, device)
            print(f"[{search_count}/{search_length}] {tuning_key}: tuning_value: {tuning_value}, v: {v}, vv: {vv}")
            history_v.append(v - vv)
            search_count += 1
            if v > opt_v:
                opt_v = v
                opt_dict[tuning_key] = tuning_value
                default_dict[tuning_key] = tuning_value
            if is_n_consecutive_decreasing(history_v, args.decrease_threshold):
                search_length = search_length + len(history_v) - len(tuning_values) - 1
                break

    print("best:", opt_dict)

    command = f"""python test_EDDM.py \\
        --input_channels {args.input_channels} \\
        --source {args.source} \\
        --target {args.target} \\
        --batch_size {args.batch_size} \\
        --which_epoch {args.which_epoch} \\
        --gpu_chose {args.gpu_chose} \\
        --input_path {args.input_path} \\
        --checkpoint_path {args.checkpoint_path} \\
        --vp_t {opt_dict["vp_t"]} \\
        --vp_k {opt_dict["vp_k"]} \\
        --vp_max {opt_dict["vp_max"]} \\
        --vp_sparse {opt_dict["vp_sparse"]} \\
        --vp_noise {opt_dict["vp_noise"]} \\
        --vp_prior {opt_dict["vp_prior"]} """

    print("Please execute the following command to run the test:")
    print("=" * 50)
    print("\033[92m" + command + "\033[0m")
    print("=" * 50 + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('EDDM parameters')
    parser.add_argument('--seed', type=int, default=1024, help='seed used for initialization')

    # mapping_model
    parser.add_argument('--num_channels_dae', type=int, default=64, help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 1, 2, 2, 4, 4], help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,), help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True, help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan', help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')
    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # training
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

    # parameters
    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    parser.add_argument('--source', type=str, default='T1', help='contrast selection for model')
    parser.add_argument('--target', type=str, default='T2', help='contrast selection for model')
    parser.add_argument('--which_epoch', type=int, default=120)
    parser.add_argument('--gpu_chose', type=int, default=0)

    #tuning
    parser.add_argument('--sample_fixed', action='store_true', default=True)
    parser.add_argument('--tuning_dataset_num', type=int, default=4)
    parser.add_argument('--lambda_p', type=float, default=25, help='weightening of PSNR')
    parser.add_argument('--lambda_s', type=float, default=1.1, help='weightening of SSIM')
    parser.add_argument('--lambda_m', type=float, default=-2000, help='weightening of MAE')
    parser.add_argument('--lambda_vp', type=float, default=750, help='weightening of vPSNR')
    parser.add_argument('--lambda_vs', type=float, default=750, help='weightening of vSSIM')
    parser.add_argument('--lambda_vm', type=float, default=750, help='weightening of vMAE')
    parser.add_argument('--decrease_threshold', type=int, default=3)
    parser.add_argument('--vp_t', type=int, default=4)
    parser.add_argument('--vp_max', type=float, default=20)
    parser.add_argument('--vp_k', type=float, default=5)
    parser.add_argument('--vp_sparse', type=float, default=1)
    parser.add_argument('--vp_noise', type=float, default=0)
    parser.add_argument('--vp_prior', type=float, default=0)
    parser.add_argument('--use_vv', action='store_true', default=False)
    parser.add_argument('--predict_middle', action='store_true', default=False)
    parser.add_argument('--lambda_middle', type=float, default=0.5)
    parser.add_argument('--output_complete', action='store_true', default=False)
    parser.add_argument('--use_multi_flow', action='store_true', default=False)
    parser.add_argument('--driving_flow', type=float, default=0.0)
    parser.add_argument('--network_type', default='normal', help='choose of normal, large, max')

    args = parser.parse_args()

    if args.network_type == 'normal':
        print("Using normal network configuration.")
    elif args.network_type == 'large':
        print("Using large network configuration.")
        args.num_channels_dae = 128
        args.num_res_blocks = 3
    elif args.network_type == 'max':
        print("Using max network configuration.")
        args.num_channels_dae = 128
        args.num_res_blocks = 4
        args.ch_mult = [1, 1, 2, 2, 4, 8]
    else:
        print(f"Unknown network type: {args.network_type}")
    
    sample_and_test(args)
    
