import collections
import torchvision
import time
import torch
from encoder import *
import functools
import dgl
import copy
import matplotlib.pyplot as plt
from stable_baselines3.common import logger as L
from sklearn.svm import LinearSVC
import umap
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from umap.parametric_umap import load_ParametricUMAP
import joblib
import os

class PerceptionV2(nn.Module):
  def __init__(self, cfg, use_svm=True, use_pca=True, large_ae=False):
    super().__init__()
    self.cfg = cfg
    patch_shape = self.cfg.patch_shape
    patch_dim = np.prod(patch_shape)
    self.patch_shape = patch_shape
    if large_ae:
      self.enc = create_mlp(patch_dim, 64, [256,256], return_seq=True)
      self.dec = create_mlp(64, patch_dim, [256,256], return_seq=True)
    else:
      self.enc = create_mlp(patch_dim, 32, [64,64], return_seq=True)
      self.dec = create_mlp(32, patch_dim, [64,64], return_seq=True)
    self.eps = -1
    self.use_svm = use_svm
    self.use_pca = use_pca
    self.umap = umap.UMAP()
    self.pca = PCA(n_components=4)
    self.svm = LinearSVC()
    self.loss_fn = nn.CrossEntropyLoss(reduction='none')
    self.to(cfg.device)
  def save(self, path):
    print('Saving percept model to', path)
    os.makedirs(path, exist_ok=True)
    torch.save(self.state_dict(), os.path.join(path, 'percept_model.pt'))
    joblib.dump(dict(umap=self.umap, svm=self.svm, pca=self.pca, eps=self.eps),
                os.path.join(path, 'percept_model.m'))
  def load(self, path, load_model_only=False):
    print('Loading percept model from', path)
    self.load_state_dict(torch.load(os.path.join(path, 'percept_model.pt'), map_location=self.cfg.device))
    if not load_model_only:
      model = joblib.load(os.path.join(path, 'percept_model.m'))
      self.svm = model['svm']
      self.eps = model['eps']
      self.umap = model['umap']
      self.pca = model['pca']
    
  def adapt(self, oimgs, ocats, epoch=0, full_img=True):
    print('Adaptation...')
    if full_img:
      print(oimgs.shape, ocats.shape)
      all_patches = einops.rearrange(to_numpy(oimgs), 'b (h p1) (w p2) c -> (b h w) (p1 p2 c)',
                                 p1=self.patch_shape[0], p2=self.patch_shape[1])

      all_labels = to_numpy(ocats.argmax(-1))
      if self.cfg.get('is_crafter', False):
        all_labels = to_numpy(all_labels).transpose((0,2,1))
      all_labels = all_labels.flatten()
      all_select_data = []
      all_select_label = []
      for c in range(self.cfg.num_cat):
        if self.cfg.get('is_crafter', False) and (c > 6 or c == 0):
          continue
        select_data = all_patches[all_labels == c]
        if len(select_data) > 0:
          select_data = select_data[:4]
          all_select_data.append(select_data)
          all_select_label.append([c] * len(select_data))
      patches = np.concatenate(all_select_data, 0)
      labels = np.concatenate(all_select_label, 0)
    else:
      patches = oimgs
      labels = ocats
    if self.use_pca:
      embeds = self.pca.fit_transform(patches)
      self.svm.fit(embeds, labels)
      pred = self.svm.predict(self.pca.transform(patches))
    else:
      self.svm.fit(patches, labels)
      pred = self.svm.predict(patches)
    if epoch <= 0:
      return
    optim = torch.optim.SGD(self.parameters(), lr=0.001)
    data = to_torch(Batch(oimg=patches, ocat=labels), device=self.cfg.device, dtype=torch.float32)
    for ep in range(epoch):
      for b in data.split(16, merge_last=True):
        conf = self.cal_conf(b.oimg)
        conf_loss = -conf.mean()
        loss = conf_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
    self.fit_eps(data.oimg)
    L.record('percept/final_loss', loss.item())
    L.record('percept/conf_loss', conf_loss.item())
    return {'all_loss': loss.item(), 'conf_loss': conf_loss.item()}
      
  def relabel_seen(self, oimg, ocat=None):
    oimg = to_numpy(oimg)
    B, H, W = oimg.shape[:3]
    H = H // self.patch_shape[0]
    W = W // self.patch_shape[1]
    patches = einops.rearrange(oimg, 'b (h p1) (w p2) c -> (b h w) (p1 p2 c)',
                               p1=self.patch_shape[0], p2=self.patch_shape[1])
    if self.use_pca:
      embeds = self.pca.transform(patches)
    else:
      embeds = patches
    pred = self.svm.predict(embeds)
    pred = pred.reshape(B, H, W)
    if self.cfg.get('is_crafter', False) and ocat is not None:
      pred = pred.transpose((0,2,1))
      ocat = to_numpy(ocat).argmax(-1)
      mask = (ocat < 7) & (ocat > 0)
      pred = np.where(mask, pred, ocat).astype(np.int)
    return np.eye(self.cfg.num_cat)[pred]
  def relabel_unseen(self, oimg, ocat):
    patches = einops.rearrange(oimg, 'b (h p1) (w p2) c -> b h w (p1 p2 c)',
                               p1=self.patch_shape[0], p2=self.patch_shape[1])
    ood_flag = self.check_ood(patches) # b h w
    ocat = ocat.argmax(-1)
    if self.cfg.num_cat == 9:
      ocat = ocat + ood_flag * (self.cfg.cat_to_expl - 1)
    elif self.cfg.num_cat >= 12:
      if self.cfg.get('is_crafter', False):
        ood_flag = ood_flag.permute((0,2,1))
        ood_inds = ood_flag.flatten() ^ ((ocat <= 6).flatten())
        ocat = ocat + ood_flag * (self.cfg.cat_to_expl - 1) * (ocat > 0) # bypass background
      else:
        ocat = ocat + ood_flag * self.cfg.cat_to_expl * (ocat > 0) # bypass background
    else:
      assert False
    return F.one_hot(ocat, self.cfg.num_cat)
    
  def cal_conf(self, x, preprocess=False):
    x = x / 255.
    if preprocess:
      x = einops.rearrange(x, 'b (h p1) (w p2) c -> b h w (p1 p2 c)',
                                 p1=self.patch_shape[0], p2=self.patch_shape[1])
    fea = self.enc(x)
    recon = self.dec(fea)
    loss = (x - recon).pow(2).sum(-1)
    conf = -loss
    return conf
  @torch.no_grad()
  def check_ood(self, x):
    conf = self.cal_conf(x)
    return conf < 1.05 * self.eps
  @torch.no_grad()
  def fit_eps(self, x, preprocess=False):
    conf = self.cal_conf(x, preprocess=preprocess)
    self.eps = conf.amin()
    return self.eps
