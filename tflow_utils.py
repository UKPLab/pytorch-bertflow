import torch
from typing import Callable, Iterable, Tuple
from torch import nn
from torch.optim import Optimizer
from torch.nn import functional as F
from transformers import AutoModel
from torch.optim.lr_scheduler import LambdaLR
from math import log, pi, exp
import numpy as np
import random
from scipy import linalg as la
import os


def simple_pooling(hidden_states, attention_mask, pooling_method):
    if 'cls' == pooling_method:
        semb = hidden_states[:, 0]  # (bsz, hdim)    
    elif 'mean' == pooling_method:
        assert attention_mask is not None
        lengths = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
        assert lengths.gt(0).sum() == lengths.shape[0]
        ###### remember to * attention_mask!!!! ######
        semb = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1).div(lengths)  # (bsz, hdim)
    elif 'max' == pooling_method:
        assert attention_mask is not None
        mask = (1 - attention_mask.unsqueeze(dim=-1)).bool()  # (bsz, seq_len) -> (bsz, seq_len, 1)
        semb = hidden_states.masked_fill(mask, -1e20).max(dim=1)[0]
    else:
        raise NotImplementedError
    return semb  # (bsz, hdim)


def compute_unconditional_prior(z):
    h = z.new_zeros(z.shape)
    prior_dist = torch.distributions.normal.Normal(h, torch.exp(h))
    return prior_dist


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True, logscale_factor=3., ddi=True):
        super().__init__()
        self.in_channel = in_channel

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.logscale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        torch.nn.init.xavier_uniform_(self.loc.data)
        torch.nn.init.xavier_uniform_(self.logscale.data)

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
        self.logscale_factor = logscale_factor
        self.ddi = ddi

    def initialize(self, input):
        # input: (bsz, hdim, 1, 1)
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)  # (hdim, bsz)
            mean = flatten.mean(1)
            std = torch.sqrt(((flatten - mean.unsqueeze(-1)) ** 2).mean(dim=1))
            mean = mean.unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = std.unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.logscale.data.copy_(torch.log(1 / (std + 1e-6)) / self.logscale_factor)

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0 and self.ddi:
            self.initialize(input)
            self.initialized.fill_(1)

        logs = self.logscale * self.logscale_factor
        logdet = height * width * torch.sum(logs)
        output = torch.exp(logs) * (input + self.loc)

        if self.logdet:
            return output, logdet
        else:
            return output


# A random permutation
class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        weight = torch.zeros(in_channel, in_channel)
        for i in range(in_channel):
            weight[i, in_channel-1-i] = 1.
        weight = weight.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input):
        out = F.conv2d(input, self.weight)
        return out, 0.0


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.in_channel = in_channel

        self.conv = nn.Conv2d(in_channel, out_channel, [1, 1], padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)

        return out


class AdditiveCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=32):
        super().__init__()
        self.in_channel = in_channel

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, [1, 1], padding=0, bias=False),
            ActNorm(filter_size, logdet=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, [1, 1], padding=0, bias=False),
            ActNorm(filter_size, logdet=False),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel // 2),
        )

        # weight initialization
        for module in self.net:
            if type(module) != nn.Conv2d:
                continue
            module.weight.data.normal_(0, 0.05)

    def forward(self, x):
        x1, x2 = x.chunk(2, 1)
        z1 = x1
        shift = self.net(x1)
        z2 = x2 + shift
        output = torch.cat([z1, z2], dim=1)
        return output, 0.0


class Flow(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2d(in_channel)
        self.coupling = AdditiveCoupling(in_channel)

    def forward(self, x):
        objective = 0
        for fn in [self.actnorm, self.invconv, self.coupling]:
            x, obj = fn(x)
            objective += obj
        return x, objective


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True):
        super().__init__()
        self.in_channel = in_channel

        self.flows = nn.ModuleList([Flow(in_channel) for _ in range(n_flow)])
        self.split = split

    def _get_eps(self, dist, x):
        return (x - dist.loc) / dist.scale
    
    def _set_eps(self, dist, eps):
        return eps * dist.scale + dist.loc

    def forward(self, x):
        b_size = x.shape[0]
        objective = 0

        for flow in self.flows:
            x, obj = flow(x)
            objective += obj

        eps = None
        if self.split:
            x1, x2 = x.chunk(2, 1)
            prior_dist = compute_unconditional_prior(x1)
            log_p = prior_dist.log_prob(x2). \
                        sum_to_size(b_size, 1, 1, 1). \
                        view(b_size)
            eps = self._get_eps(prior_dist, x2)
            x = x1
            objective = objective + log_p

        return x, objective, eps


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow=3, n_block=3):
        super().__init__()
        self.in_channel = in_channel

        self.blocks = nn.ModuleList()
        for _ in range(n_block - 1):
            self.blocks.append(Block(in_channel, n_flow))
            in_channel //= 2
        self.blocks.append(Block(in_channel, n_flow, split=False))

    def forward(self, emb):
        # emb: (bsz, hdim)
        x = emb[:, :, None, None]  # b_size, n_channel, height, width
        b_size, c, h, w = x.shape
        
        log_p_sum = 0
        all_eps = []

        obj = 0
        for block in self.blocks:
            x, log_p, eps = block(x)
            if eps is not None:
                all_eps.append(eps)

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        
        obj += log_p_sum
        z = x
        b_size = z.shape[0]
        prior_dist = compute_unconditional_prior(z)
        prior_objective = prior_dist.log_prob(z). \
                        sum_to_size(b_size, 1, 1, 1). \
                        view(b_size)
        if obj.shape != prior_objective.shape:
            obj = obj.unsqueeze(-1)
        obj = obj + prior_objective
        loss = (-obj / (np.log(2) * h * w * c)).mean()
        z = torch.cat(all_eps + [z], dim=1).view(b_size, c)
        return z, loss


class TransformerGlow(nn.Module):

    def __init__(self, model_name_or_path=None, pooling=None):
        super().__init__()
        if model_name_or_path is None and pooling is None:
            self.transformer = None
            self.glow = None
            return 
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.glow = Glow(self.transformer.config.hidden_size)
        self.transformer.config.pooling = pooling
        assert pooling in ['mean', 'cls', 'max', 'first-last-avg']
        self.pooling = pooling
        self._fix()

    @property
    def config(self):
        return self.transformer.config

    @property
    def device(self):
        return self.transformer.device

    def _fix(self):
        for param in self.transformer.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids, attention_mask, return_loss=False):
        with torch.no_grad():
            if self.pooling == 'first-last-avg':
                all_hidden_states = self.transformer(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states
                lengths = attention_mask.sum(dim=1, keepdim=True)  # (bsz, 1)
                semb = (
                        # (all_hidden_states[-1] * attention_mask.unsqueeze(-1)).sum(dim=1) + \
                        # (all_hidden_states[-2] * attention_mask.unsqueeze(-1)).sum(dim=1)
                        (all_hidden_states[-0] * attention_mask.unsqueeze(-1)).sum(dim=1) + \
                        (all_hidden_states[-1] * attention_mask.unsqueeze(-1)).sum(dim=1)
                    ).div(2 * lengths)  # (bsz, hdim)
            else:
                hidden_states = self.transformer(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )[0]
                semb = simple_pooling(hidden_states, attention_mask, self.pooling)  # (bsz, hdim)
        z, loss = self.glow(semb)
        if return_loss:
            return z, loss
        return z
    
    def save_pretrained(self, save_directory):
        self.transformer.save_pretrained(save_directory)
        output_model_file = os.path.join(save_directory, 'glow.bin')
        torch.save(self.glow.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        model = cls()
        model.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path)
        model_file_to_load = os.path.join(pretrained_model_name_or_path, 'glow.bin')
        model.glow = Glow(model.transformer.config.hidden_size)
        model.glow.load_state_dict(torch.load(model_file_to_load, map_location='cpu'))
        model.pooling = model.transformer.config.pooling
        model._fix()
        model.eval()
        return model


class AdamWeightDecayOptimizer(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                m, v = state['m'], state['v']  # m, v
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                lr = group['lr']

                next_m = beta1 * m + (1 - beta1) * grad
                next_v = beta2 * v + (1 - beta2) * (grad ** 2)

                update = next_m / (torch.sqrt(next_v) + eps)
                if weight_decay != 0:
                    update += weight_decay * p

                updata_with_lr = lr * update
                p.add_(-updata_with_lr)
                state['m'] = next_m
                state['v'] = next_v
                state['step'] += 1
        return loss