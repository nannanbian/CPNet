import torch
from torch import nn
from itertools import chain


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class TokenProjection(nn.Module):

    def __init__(self, configs, seg_len):

        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=seg_len, padding=(seg_len-1)//2)
        self.project = MLP(configs.seq_len, self.pred_len, configs.d_hid)
        self.layer_norm = nn.LayerNorm(self.pred_len)
        self.act = nn.Tanh()

    def forward(self, x):

        x_out = self.conv(x)
        x_out = self.act(x_out)
        x_out = self.project(x_out)
        x_out = self.layer_norm(x_out)

        return x_out


class ContextualSampling(nn.Module):

    def __init__(self, configs, length):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.length = length

        if (self.seq_len + self.pred_len) % self.length != 0:
            self.nums = (self.seq_len + self.pred_len) // self.length + 1
            self.padding = self.nums * self.length - (self.seq_len + self.pred_len)
        else:
            self.nums = (self.seq_len + self.pred_len) // self.length
            self.padding = 0

        self.padding_sample_layer = nn.ReplicationPad1d((0, self.padding))
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size=configs.k, dilation=self.nums,
                               padding=self.nums*(configs.k-1)//2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1,
                               kernel_size=self.nums, stride=self.nums)
        self.act = nn.Tanh()

    def forward(self, x_enc, x):
        x = torch.cat([x_enc, x], dim=-1)
        x_pad = self.padding_sample_layer(x)

        out = self.conv1(x_pad)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)

        return out


class ContextualSampling_Multi(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.length = configs.sample_len

        self.single = nn.ModuleList([ContextualSampling(configs, self.length[i]) for i in range(len(self.length))])
        self.project = nn.ModuleList([MLP(self.length[i], self.pred_len + self.seq_len, configs.d_hid)
                                      for i in range(len(configs.sample_len))])
        self.layer_norm = nn.LayerNorm(self.pred_len + self.seq_len)

    def forward(self, x_enc, x):

        branch = []
        for i in range(len(self.length)):
            x_out = self.single[i](x_enc, x)
            x_out = self.project[i](x_out)
            x_out = self.layer_norm(x_out)
            branch.append(x_out)

        return branch


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = configs.seg_len
        self.sample_len = configs.sample_len

        self.tp = nn.ModuleList([TokenProjection(configs, self.seg_len[i])
                                      for i in range(len(self.seg_len))])
        self.cs = nn.ModuleList([ContextualSampling_Multi(configs)
                                      for _ in range(len(self.seg_len))])

        self.merge = nn.Conv2d(in_channels=1, out_channels=1,
                             kernel_size=[3, len(configs.seg_len)*len(configs.sample_len)], padding = [1, 0])
        self.layer_norm = nn.LayerNorm(self.pred_len + self.seq_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(B * N, x_enc.shape[2]).unsqueeze(-2)

        pattern = []
        for i in range(len(self.seg_len)):
            enc_out = self.tp[i](x_enc)

            branch = self.cs[i](x_enc, enc_out)
            pattern.append(branch)

        pattern_a = list(chain(*pattern))
        dec_out = torch.stack(pattern_a, dim=-1)

        dec_out = self.merge(dec_out).squeeze(dim=-1)
        dec_out = self.layer_norm(dec_out)

        dec_out = dec_out.squeeze(-2).reshape(B, N, -1)
        dec_out = dec_out.permute(0, 2, 1)

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out[:, -self.pred_len:, :]
