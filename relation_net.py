from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
import stable_baselines3.common.logger as L
import functools
import gym
import numpy as np
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from encoder import *
import einops

class AddSInfoV2(nn.Module):
  def __init__(self, h, w, c, cout=32, **kwargs):
    super().__init__()
    grid_h = F.one_hot(torch.arange(h), h) # h h
    grid_w = F.one_hot(torch.arange(w), w)
    self.grid_h = einops.repeat(grid_h, 'h c -> h w c', w=w).float()
    self.grid_w = einops.repeat(grid_w, 'w c -> h w c', h=h).float()
    self.h_embed = nn.Linear(h, 8)
    self.w_embed = nn.Linear(w, 8)
    self.all_embed = nn.Linear(c + 8 + 8, cout)
  def forward(self, x):
    h_info, w_info = self.h_embed(self.grid_h), self.w_embed(self.grid_w)
    B = x.shape[0]
    h_info = h_info.expand(B, -1, -1, -1)
    w_info = w_info.expand(B, -1, -1, -1)
    # print(x.shape, h_info.shape, w_info.shape)
    x = torch.cat([x, h_info, w_info], dim=-1)
    return self.all_embed(x)
    

class AddSInfo(nn.Module):
  def __init__(self, h, w, c, cout=32, channel_first=False, use_mlp=True):
    super().__init__()
    identity = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float32)
    grid = F.affine_grid(identity, [1, 1, h, w])
    grid = grid.permute(0, 3, 1, 2).contiguous()
    # (1, 2, h, w)
    self.register_buffer('grid', grid)
    assert channel_first == False
    if not channel_first:
      # (1, h, w, 2)
      self.grid = grid.permute(0,2,3,1)
    self.use_mlp = use_mlp
    if self.use_mlp:
      self.mlp = nn.Linear(c+2, cout)
  def forward(self, x):
    x = torch.cat([x, self.grid.to(x.device).expand(x.shape[0], -1, -1, -1)], dim=-1)
    if self.use_mlp:
      x = self.mlp(x)
    return x

class ObjSummary(nn.Module):
  def __init__(self, c, obj_cat_num):
    super().__init__()
    self.head = 4
    self.query_atten = QueryMultiHeadAttention(obj_cat_num, c, self.head,
                        to_q_net=[32], to_k_net=[32], to_v_net=[32], to_out_net=[])
    self.out_dim = c * obj_cat_num
  """
    x: (N, B, E)
    obj_cat: (N, B, S)
    out: (B, S*E)
  """
  def forward(self, x, obj_cat):
    mask = einops.repeat(obj_cat, 'n b s -> b h s n', h=self.head)
    out = self.query_atten(x, mask=mask)
    out = einops.rearrange(out, 's n e -> n (s e)')
    return out

class RNModule(nn.Module):
  def __init__(self, cfg, input_shape, num_layers=2):
    super().__init__()
    self.cfg = cfg
    h, w, c = input_shape
    self.trans = Rearrange('n h w c -> (h w) n c')
    if cfg.get('sinfo_v2', False):
      self.add_sinfo = AddSInfoV2(h, w, c, cout=c)
    else:
      self.add_sinfo = AddSInfo(h, w, c, cout=c)
    self.num_layers = num_layers
    self.atten = nn.MultiheadAttention(c, 4)
    self.linear = create_mlp(2 * c, c, [c], return_seq=True)
  def forward(self, x):
    *dims, H, W, C = x.shape
    x = x.reshape(np.prod(dims), H, W, C)
    x = self.add_sinfo(x)
    x = self.trans(x)
    for _ in range(self.num_layers):
      atten_out, atten_wts = self.atten(x,x,x)
      x_atten = torch.cat([x, atten_out], dim=-1)
      x = x + self.linear(x_atten)
    x = x.amax(0).unsqueeze(1)
    x = x.reshape(*dims, *x.shape[1:])
    return x

class RNModuleV2(nn.Module):
  def __init__(self, cfg, input_shape, num_layers=2):
    super().__init__()
    self.cfg = cfg
    h, w, cin = input_shape
    c = cfg.hidden_dim
    self.embed = nn.Linear(cin, c)
    self.trans = Rearrange('n h w c -> (h w) n c')
    if cfg.get('sinfo_v2', False):
      self.add_sinfo = AddSInfoV2(h, w, c, cout=c)
    else:
      self.add_sinfo = AddSInfo(h, w, c, cout=c)
    self.num_layers = num_layers
    self.atten = nn.MultiheadAttention(c, 4)
    self.linear = create_mlp(2 * c, c, [c], create_layer=functools.partial(MultiLinear, num_linears=cin), return_seq=True)
  def forward(self, x):
    *dims, H, W, C = x.shape
    xin = x.reshape(np.prod(dims), H, W, C)
    x = self.embed(xin)
    x = self.add_sinfo(x)
    x = self.trans(x)
    xin = self.trans(xin)
    for _ in range(self.num_layers):
      atten_out, atten_wts = self.atten(x,x,x)
      x_atten = torch.cat([x, atten_out], dim=-1) # (S, B, 2C)
      xout = (xin.unsqueeze(-1) * self.linear(MultiLinear.broadcast(x_atten, C))).sum(-2)
      x = x + xout
    x = x.amax(0).unsqueeze(1)
    x = x.reshape(*dims, *x.shape[1:])
    return x