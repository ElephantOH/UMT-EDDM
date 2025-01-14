from . import eddm_layers
import torch.nn as nn
import functools
import torch

conv3x3 = eddm_layers.conv3x3
conv1x1 = eddm_layers.conv1x1
dense = eddm_layers.dense
default_init = eddm_layers.default_init
get_t_embed = eddm_layers.get_t_embed
ResnetBlockBigGAN = eddm_layers.ResnetBlockBigGAN
AttnBlock = eddm_layers.AttnBlock
CrossAttnBlock = eddm_layers.CrossAttnBlock
PixelNorm = eddm_layers.PixelNorm

class Config:
    def __init__(self):
        self.image_size = 256
        self.input_channels = 3
        self.num_channels = 64
        self.conditional = True
        self.level_channels = [1, 1, 2, 2, 4, 4]
        self.z_emb_dim = 100
        self.z_emb_channels = [256, 256, 256, 256]
        self.t_emb_dim = 64
        self.t_emb_channels = [256, 256, 256]
        self.num_resblocks = 2
        self.attn_levels = [16, ]
        self.use_cross_attn = True
        self.dropout = 0.1
        self.use_tanh_final = True
        self.fir_kernel = [1, 3, 3, 1]
        self.skip_rescale = True
        self.resblock_type = 'biggan'
        self.output_complete = True
        self.phase = 'train'

@eddm_layers.register_model(name='eddm')
class EDDM(nn.Module):
    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = self.get_default_config()

        self.image_size = config.image_size
        self.input_channels = config.input_channels
        self.output_channels = config.input_channels
        self.num_channels = config.num_channels
        self.conditional = config.conditional
        self.level_channels = config.level_channels
        self.num_levels = len(self.level_channels)
        self.all_levels = [self.image_size // (2 ** i) for i in range(self.num_levels)]

        self.z_emb_dim = config.z_emb_dim
        self.z_emb_channels = config.z_emb_channels
        self.t_emb_dim = config.t_emb_dim
        self.t_emb_channels = config.t_emb_channels

        self.num_resblocks = config.num_resblocks
        self.attn_levels = config.attn_levels
        self.use_cross_attn = config.use_cross_attn
        self.dropout = config.dropout
        self.use_tanh_final = config.use_tanh_final
        self.act = nn.SiLU()

        self.fir_kernel = config.fir_kernel
        self.skip_rescale = config.skip_rescale
        self.resblock_type = config.resblock_type

        self.output_complete = config.output_complete
        self.init_scale = 0.
        self.phase = config.phase

        self.ResnetBlock = functools.partial(ResnetBlockBigGAN, act=self.act, dropout=self.dropout, fir_kernel=self.fir_kernel,
                                        init_scale=self.init_scale, skip_rescale=self.skip_rescale,
                                        t_emb_dim=self.t_emb_channels[-1], z_emb_dim=self.z_emb_channels[-1])
        self.AttnBlock = functools.partial(AttnBlock, init_scale=self.init_scale, skip_rescale=self.skip_rescale)
        self.CrossAttnBlock = functools.partial(CrossAttnBlock, init_scale=self.init_scale, skip_rescale=self.skip_rescale)
        self.t_modules = None
        self.z_modules = None
        self.down_modules = []
        self.middle_modules = []
        self.up_modules = []

        self.model_layer_info = {}

        self.load_main_modules()
        self.load_t_modules()
        self.load_z_modules()

        # print(self.model_layer_info)

    def load_main_modules(self):
        down_modules = []
        middle_modules = []
        up_modules = []
        input_channel = self.input_channels
        image_size = self.image_size

        if self.conditional:
            input_channel = input_channel * 2

        # down_sample
        down_modules.append(conv3x3(input_channel, self.num_channels))

        residual_channels = [self.num_channels]
        input_channel = self.num_channels
        output_channel = self.num_channels

        for i_level in range(self.num_levels):
            for i_block in range(self.num_resblocks):
                output_channel = self.num_channels * self.level_channels[i_level]
                down_modules.append(self.ResnetBlock(in_ch=input_channel, out_ch=output_channel))
                input_channel = output_channel

                if self.all_levels[i_level] in self.attn_levels:
                    down_modules.append(self.AttnBlock(channels=input_channel))

                    if self.use_cross_attn:
                        down_modules.append(self.CrossAttnBlock(channels=input_channel))

                residual_channels.append(input_channel)


            if i_level != self.num_levels - 1: # if not last levels, down_sample
                down_modules.append(self.ResnetBlock(down=True, in_ch=input_channel))

                image_size = image_size / 2
                residual_channels.append(input_channel)


        # middle_sample
        input_channel = residual_channels[-1]
        output_channel = residual_channels[-1]

        middle_modules.append(self.ResnetBlock(in_ch=input_channel, out_ch=output_channel))


        if self.use_cross_attn:
            middle_modules.append(self.CrossAttnBlock(channels=input_channel))
        middle_modules.append(self.AttnBlock(channels=input_channel))

        if self.use_cross_attn:
            middle_modules.append(self.CrossAttnBlock(channels=input_channel))

        middle_modules.append(self.ResnetBlock(in_ch=input_channel, out_ch=output_channel))


        # up_sample
        for i_level in reversed(range(self.num_levels)):
            for i_block in range(self.num_resblocks + 1):
                output_channel = self.num_channels * self.level_channels[i_level]
                up_modules.append(self.ResnetBlock(in_ch=input_channel + residual_channels.pop(), out_ch=output_channel))

                input_channel = output_channel

            if self.all_levels[i_level] in self.attn_levels:
                up_modules.append(self.AttnBlock(channels=input_channel))
                if self.use_cross_attn:
                    up_modules.append(self.CrossAttnBlock(channels=input_channel))

            if i_level != 0: # if not last levels, up_sample
                up_modules.append(self.ResnetBlock(in_ch=input_channel, up=True))


        output_channel = self.output_channels
        if self.conditional:
            output_channel = output_channel * 2

        up_modules.append(nn.GroupNorm(num_groups=min(input_channel  // 4, 32), num_channels=input_channel , eps=1e-6))

        up_modules.append(conv3x3(input_channel , output_channel, init_scale=self.init_scale))

        self.down_modules = nn.ModuleList(down_modules)
        self.middle_modules = nn.ModuleList(middle_modules)
        self.up_modules = nn.ModuleList(up_modules)

    def load_t_modules(self):
        t_modules = [nn.Linear(self.t_emb_dim, self.t_emb_channels[0])]
        t_modules[-1].weight.data = default_init()(t_modules[-1].weight.shape)
        nn.init.zeros_(t_modules[-1].bias)
        t_modules.append(self.act)
        for i in range(len(self.t_emb_channels) - 1):
            t_modules.append(nn.Linear(self.t_emb_channels[i], self.t_emb_channels[i + 1]))
            t_modules[-1].weight.data = default_init()(t_modules[-1].weight.shape)
            nn.init.zeros_(t_modules[-1].bias)
            if i != len(self.t_emb_channels) - 2:
                t_modules.append(self.act)
        self.t_modules = nn.Sequential(*t_modules)

    def load_z_modules(self):
        z_modules = [PixelNorm(), dense(self.z_emb_dim, self.z_emb_channels[0]), self.act, ]
        for i in range(len(self.z_emb_channels) - 1):
            z_modules.append(dense(self.z_emb_channels[i], self.z_emb_channels[i + 1]))
            z_modules.append(self.act)
        self.z_modules = nn.Sequential(*z_modules)

    def forward(self, x, t, z):
        z_emb = self.z_modules(z)

        t_emb = get_t_embed(t, self.num_channels)
        t_emb = self.t_modules(t_emb)

        # down_sample
        m_idx = 0
        residual_parts = [self.down_modules[m_idx](x)]
        m_idx += 1

        for i_level in range(self.num_levels):
            for i_block in range(self.num_resblocks):
                h = self.down_modules[m_idx](residual_parts[-1], t_emb, z_emb)
                m_idx += 1

                if h.shape[-1] in self.attn_levels:
                    h = self.down_modules[m_idx](h)
                    m_idx += 1

                    if self.use_cross_attn:
                        h = self.down_modules[m_idx](h)
                        m_idx += 1

                residual_parts.append(h)

            if i_level != self.num_levels - 1:
                h = self.down_modules[m_idx](residual_parts[-1], t_emb, z_emb)
                m_idx += 1
                residual_parts.append(h)

        # middle_sample
        m_idx = 0
        h = residual_parts[-1]
        h = self.middle_modules[m_idx](h, t_emb, z_emb)
        m_idx += 1

        if self.use_cross_attn:
            h = self.middle_modules[m_idx](h)
            m_idx += 1

        h = self.middle_modules[m_idx](h)
        m_idx += 1

        if self.use_cross_attn:
            h = self.middle_modules[m_idx](h)
            m_idx += 1

        h = self.middle_modules[m_idx](h, t_emb, z_emb)
        m_idx += 1

        # up_sample
        m_idx = 0
        for i_level in reversed(range(self.num_levels)):
            for i_block in range(self.num_resblocks + 1):
                h = self.up_modules[m_idx](torch.cat([h, residual_parts.pop()], dim=1), t_emb, z_emb)
                m_idx += 1

            if h.shape[-1] in self.attn_levels:
                h = self.up_modules[m_idx](h)
                m_idx += 1
                if self.use_cross_attn:
                    h = self.up_modules[m_idx](h)
                    m_idx += 1

            if i_level != 0:
                h = self.up_modules[m_idx](h, t_emb, z_emb)
                m_idx += 1

        h = self.act(self.up_modules[m_idx](h))
        m_idx += 1
        h = self.up_modules[m_idx](h)
        m_idx += 1

        out = h
        if self.use_tanh_final:
            out = torch.tanh(out)

        if self.output_complete and self.phase == 'train':
            if self.output_channels == 3:
                return out[:, [0, 1, 2], ...], out[:, [3, 4, 5], ...]
            elif self.output_channels == 1:
                return out[:, [0], ...], out[:, [1], ...]
        else:
            if self.output_channels == 3:
                return out[:, [0, 1, 2], ...], None
            elif self.output_channels == 1:
                return out[:, [0], ...], None

    def get_default_config(self):
        return Config()

    def record_operations(self, op, channel1, size1, channel2, size2):
        if op == "residual_input" or op == "residual_output":
            self.model_layer_info.setdefault((size1+size2)/2., []).append('*')
        elif channel1 == channel2:
            self.model_layer_info.setdefault((size1+size2)/2., []).append('[{},{},{}]'.format(op, channel1, size1))
        else:
            self.model_layer_info.setdefault((size1+size2)/2., []).append('[{},{}->{}]'.format(op, channel1, channel2))