import torch
import torch.nn as nn
# import satnet
import numpy as np
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import einops
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from stable_baselines3.common import logger as L
def create_mlp(
    input_dim,
    output_dim,
    net_arch,
    activation_fn= nn.ReLU,
    squash_output=False,
    create_layer=nn.Linear,
    return_seq =False,
):
    if len(net_arch) > 0:
        modules = [create_layer(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(create_layer(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(create_layer(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    if return_seq:
      modules = nn.Sequential(*modules)
      apply_init(modules)
    return modules


class MultiLinear(nn.Module):
  def __init__(self, in_channels, out_channels, num_linears=2, add_bias=True,):
    super().__init__()
    self.W = nn.Parameter(torch.randn(num_linears, in_channels, out_channels))
    self.b = nn.Parameter(torch.zeros(num_linears, out_channels))
    self.add_bias = add_bias
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.num_linears = num_linears
    nn.init.orthogonal_(self.W, nn.init.calculate_gain('relu'))
  @staticmethod
  def make_layer(num_linears):
    def create_layer(din, dout):
      return MultiLinear(din, dout, num_linears)
    return create_layer
  @staticmethod
  def broadcast(x, num_linears):
    shape = list(x.shape)
    extended_shape = shape[:-1] + [num_linears] + shape[-1:] 
    return x.unsqueeze(-2).expand(extended_shape)
  @staticmethod
  def reduce(x, reduce_type):
    if reduce_type == 'sum':
      return x.sum(-2)
    elif reduce_type == 'max':
      return x.amax(-2) 
    elif reduce_type == 'mean':
      return x.mean(-2)
    elif reduce_type == 'flat':
      return x.reshape(np.prod(x.shape[:-1]), x.shape[-1])
  """
    input:  (..., n, cin)
    output: (..., n, cout)
  """
  def forward(self, x):
    x = x.unsqueeze(-2)               # (..., n, 1, in_channels)
    out = torch.matmul(x, self.W)     # (..., n, 1, out_channels)
    out = out.squeeze(-2)
    if self.add_bias:
      out = out + self.b    # (..., n, out_channels)
    return out
  def __repr__(self):
    return f'MultiLinear(in={self.in_channels}, out={self.out_channels}, num={self.num_linears}, add_bias={self.add_bias})'

def apply_init(module):
  for m in module.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
      nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain('relu'))
    elif isinstance(m, MultiLinear):
      nn.init.orthogonal_(m.W.data, nn.init.calculate_gain('relu'))

class MultiAttention(nn.Module):
    def __init__(self, dim, heads, dim_head=None, dropout = 0.,
                qdim=None, kdim=None, vdim=None,
                layer_norm=False,
                to_q_net=[], to_k_net=[], to_v_net=[], to_out_net=[]):
      super().__init__()
      if dim_head is None:
        dim_head = dim // heads
      self.dim = dim
      self.heads = heads
      self.dim_head = dim_head
      qdim = dim if qdim is None else qdim
      vdim = dim if vdim is None else vdim
      kdim = dim if kdim is None else kdim
      self.scale = dim_head ** -0.5
      self.to_q = create_mlp(qdim, heads * dim_head, to_q_net, return_seq=True)
      self.to_k = create_mlp(kdim, heads * dim_head, to_k_net, return_seq=True)
      self.to_v = create_mlp(vdim, heads * dim_head, to_v_net, return_seq=True)
      self.to_out = create_mlp(dim_head*heads, dim, to_out_net, return_seq=True)
      self.layer_norm = nn.LayerNorm(qdim) if layer_norm else nn.Identity()
      self.ac = nn.Softmax(-1)
    def forward(self, q, k, v, mask=None, attn_dim=None, q_fn=None, hard=False, tau=1.):
      """
        input: q(L, B, qE), kv(N, B, E)
        output: out(L, B, E), atten(B L N)
        mask: (B, h, L, N)
      """
      origin_q = q
      qkv = self.to_q(q), self.to_k(k), self.to_v(v)
      q, k, v = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h = self.heads), qkv)
      dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
      if not hard:
        attn = self.ac(dots)  # (b, h, l, n)
        if q_fn is not None: 
          attn = q_fn(attn) # b h l n
        if mask is not None:
          assert torch.all(mask >= 0)
          attn = attn * mask
          attn = attn / (attn.sum(-1, keepdim=True) + 1e-4)
      else:
        if mask is not None:
          dots = dots * mask + (1 - mask) * -1e4
        # L.record_mean('atten_ent', torch.distributions.Categorical(logits=dots).entropy().mean())
        attn = F.gumbel_softmax(dots, tau=tau, hard=True)
      out = einsum('b h i j, b h j d -> b h i d', attn, v)
      out = rearrange(out, 'b h n d -> n b (h d)')
      out =  self.to_out(out)
      out = self.layer_norm(out)
      if attn_dim is None:
        return out, attn.mean(dim=1)
      elif attn_dim == 'all':
        return out, attn
      elif attn_dim == 'logits':
        return out, dots
      else:
        return out, attn[:,attn_dim,...]

class QueryMultiHeadAttention(nn.Module):
  def __init__(self, L, E, head_num = 4, qE=None, kE=None, vE=None, **kwargs):
    super().__init__()
    self.query = nn.Parameter(torch.randn(L, 1, E))
    self.atten = MultiAttention(E, head_num, qdim=qE, kdim=kE, vdim=vE, **kwargs)
    self.L = L
    self.E = E
  """
    key, value: (S, N, E)
    out:        (L, N, E)
  """
  def forward(self, kv, mask=None, q_fn=None):
    if isinstance(kv, list):
      key, value = kv
    else:
      key = kv
      value = kv
    # if q_fn is not None:
    #   query = q_fn(self.query)
    # else:
    query = self.query.expand(self.L, key.shape[1], self.E)
    out = self.atten(query, key, value, mask = mask, q_fn = q_fn)
    return out[0]

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
    def forward(self, x):
        inputs = x
        x = F.relu(x, inplace=True)
        x = self.conv0(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x
    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2



def make_encoder(shape, channels):
    conv_seqs = []
    for out_channels in channels:
        conv_seq = ConvSequence(shape, out_channels)
        shape = conv_seq.get_output_shape()
        conv_seqs.append(conv_seq)
    return conv_seqs, shape

class NormAndTrans(nn.Module):
    def __init__(self, norm=True, inds=[0,3,1,2], device=None):
        super().__init__()
        self.norm = norm
        self.inds = inds
        self.device = device
    def forward(self, x):
        if self.device is not None:
          x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if self.norm:
            x = x / 255. 
        assert len(x.shape) == len(self.inds), f'NormAndTrans: {x.shape} v.s. {self.inds}'
        return x.permute(*self.inds)

class ImpalaEncoder(nn.Module):
    def __init__(self, obs_space, latent_dim=256, activation_fn=nn.ReLU, lnorm=False, flatten=True, channels=[16,32,32]):
        super().__init__()
        h, w, c = obs_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in channels:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.final_shape = shape
        if flatten:
          self.cnn = nn.Sequential(*conv_seqs, nn.Flatten(), nn.ReLU(inplace=True))
        else:
          self.cnn = nn.Sequential(*conv_seqs, )
        n_flatten = int(np.prod(shape))
        if flatten:
          self.linear = nn.Linear(n_flatten, latent_dim)
          self.latent_dim = latent_dim
        else:
          self.linear = nn.Identity()
          self.latent_dim = n_flatten
        self.lnorm = nn.LayerNorm(latent_dim) if lnorm else nn.Identity()
        self.final_ac = activation_fn()
        apply_init(self)
    def forward(self, obs, ret_latent=False, **kwargs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        latent = self.linear(self.cnn(x))
        latent = self.final_ac(self.lnorm(latent))
        if ret_latent:
          return latent, latent
        else:
          return latent

class AsLayer(nn.Module):
    def __init__(self, layer_fn):
        super().__init__()
        self.layer_fn = layer_fn
    def forward(self, x):
        return self.layer_fn(x)
