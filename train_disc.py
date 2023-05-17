import einops
import io
from moviepy.video.io.bindings import mplfig_to_npimage
import collections
import argparse
import tqdm
from builtins import isinstance
import matplotlib
# matplotlib.use('agg')  # turn off interactive backend
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.policy.base import _nstep_return
from tianshou.policy import DQNPolicy
from graph_gru import *
import itertools
from stable_baselines3.common import logger as L
import os
from traj_buf import VecHistReplayBuffer, VecTrajectoryReplayBuffer
from tianshou.utils import BaseLogger, LazyLogger, MovAvg, tqdm_config
def traverseBatch(b, fn):
    for k,v in b.items():
        if isinstance(v, torch.Tensor):
            b[k] = fn['tensor'](v)
        elif isinstance(v, np.ndarray):
            b[k] = fn['ndarray'](v)
        elif isinstance(v, Batch):
            traverseBatch(v, fn)
        else:
            pass
def load_all_data(data_dir, cfg):
    bufs = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".hd5"):
                version = int(file.split('_')[0][3:])
                if version < cfg.data_version or version > cfg.data_version_end:
                    print(f'Bypass {file} with version {version}')
                    continue
                print(f'Loading {file}')
                bufs.append(VecTrajectoryReplayBuffer.load_hdf5(os.path.join(root, file)))
    all_data = Batch.cat([buf.sample(0)[0] for buf in bufs])
    return all_data

idx_to_name = {0: 'none',  1: 'zombie', 2: 'agent', 3: 'cow', 4: 'wall', 5 :'unknown'}
permutation = np.arange(len(idx_to_name.keys()))
name_to_idx = {str(v):k for k,v in idx_to_name.items()}
unknowns = ['zombie', 'agent', 'cow', 'wall']
def preprocess_fn(b, cfg):
    b = to_torch(b, dtype=torch.float32, device=cfg.device)
    obs, obs_next = b.obs, b.obs_next
    cat_to_expl = cfg.cat_to_expl
    C = cfg.num_cat
    B, T = obs.view.shape[:2]
    map = einops.repeat(torch.arange(C), 'c -> b c', b=B).long()
    labels = []
    for i in range(B):
        select_inds = np.asarray([name_to_idx[ind] for ind in unknowns], dtype=np.int)
        to_inds = torch.as_tensor(select_inds).long() + cfg.cat_to_expl
        map[i][select_inds] = torch.arange(len(select_inds)).long() + cfg.cat_to_expl
        label = torch.zeros(C).long()
        label[cfg.cat_to_expl:cfg.cat_to_expl+len(select_inds)] = torch.from_numpy(select_inds).long()
        labels.append(label)
    labels = einops.repeat(torch.stack(labels, dim=0), 'b c -> b t c', t=T)
    map = map.flatten()
    view_dim = len(obs['view'].shape)
    view = obs['view'] + C * torch.arange(B).reshape([-1]+[1]*(view_dim-1)).long()
    prev_view = obs['prev_view'] + C * torch.arange(B).reshape([-1]+[1]*(view_dim-1)).long()
    view_next = obs_next['view'] + C * torch.arange(B).reshape([-1]+[1]*(view_dim-1)).long()
    obs['view'] = F.one_hot(map[view.long()], C)
    obs['prev_view'] = F.one_hot(map[prev_view.long()], C)
    obs_next['view'] = F.one_hot(map[view_next.long()], C)
    obs_next['prev_view'] = F.one_hot(map[view.long()], C)
    obs['label'] = labels
    obs_next['label'] = labels
    return to_torch(Batch(obs=obs, obs_next=obs_next, mask=b.policy.valid_mask),
                        dtype=torch.float32, device=cfg.device)

device = 'cpu' 
if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  device = 'cuda'
defalut_cfg = dict(num_cat=12, num_act=9, 
      hidden_dim=32, info_spec = dict(), 
      cat_to_expl=6,
      num_unknown=3,
      device=device,
      logdir='train_disc', 
      loaddir = 'train_disc',
      data_version=0, data_version_end=8,
      epoch=5,
      data_dir='ppo_rn_all/gsZ1C1/s1507604',
      eval_data_dir = '',
      disc_type='RefTrajEmbed',
      pred_multi_layer=True,
)

def eval(args, cfg=None, disc=None, all_data=None, step=0, preprocess=True, version=1):
    from PIL import Image, ImageDraw, ImageFont
    cfg = oc.DictConfig(defalut_cfg)
    cfg_cli = oc.OmegaConf.from_dotlist(args.opts)
    print(f'Args: {args.opts}')
    cfg = oc.OmegaConf.merge(cfg, cfg_cli)
    if step >= 0:
        L.configure(cfg.logdir, ['tensorboard'])
        oc.OmegaConf.save(cfg, os.path.join(cfg.logdir, 'config.yaml'))
    if disc is None:
        disc = RefTrajEmbed(cfg)
        disc.load_state_dict(torch.load(cfg.loaddir))
    if all_data is None:
        all_data = load_all_data(cfg.eval_data_dir, cfg)
    disc.to(cfg.device)
    N, _ = all_data.obs.view.shape[:2]
    inds = np.random.choice(np.arange(N), 16, replace=False)
    all_data = all_data[inds]
    if preprocess:
        all_data = preprocess_fn(all_data, cfg)
    else:
        all_data.mask = all_data.policy.valid_mask
        all_data = to_torch(all_data, dtype=torch.float32, device=cfg.device)
    # omit the first frame.
    traverseBatch(all_data, dict(tensor=lambda x: x[:,1:]))
    N, T = all_data.obs.view.shape[:2]
    rews, loss, _, other_info = disc.label_rewards(all_data, all_data.mask)
    select_inds = np.asarray([name_to_idx[ind] for ind in unknowns],
                             dtype=np.int)
    num_unknowns = len(unknowns)
    probs = other_info['probs']
    targ_p= other_info['targ_p']
    targ_logp = other_info['targ_logp']
    torch.set_printoptions(precision=2)
    def plot_matric(conf_matric):
        conf_matric = torchvision.utils.make_grid(
                conf_matric.reshape(-1, 1, num_unknowns, num_unknowns), nrow=4, padding=0)[0]
        with io.BytesIO() as buff:
            plt.figure()
            plt.matshow(to_numpy(conf_matric), 'hot')
            plt.savefig(buff, format='png')
            buff.seek(0)
            conf_matric = plt.imread(buff).astype(np.float32)
        return conf_matric
    inds_to_expl = select_inds + cfg.cat_to_expl
    conf_matric_max = probs.amax(1)[...,select_inds][..., inds_to_expl, :]
    final_idx = (1 - all_data.mask.reshape(N,T)).argmax(1)
    print(final_idx)
    if version == 1:
        gt_labels = torch.arange(num_unknowns).long()
    elif version == 2:
        gt_labels = torch.as_tensor([2,0,1]).long()
    final_idx = einops.repeat(final_idx, 'n -> n 1 c1 c2', c1=num_unknowns, c2=num_unknowns)
    conf_matric_final = probs[...,select_inds][...,inds_to_expl, :]
    conf_matric_final = conf_matric_final.gather(1, final_idx).squeeze(1) # N C C
    correct_ratio = (conf_matric_final.argmax(-1) == gt_labels).all(-1).float().mean()
    print('CorrectRatio/Final', correct_ratio.item())
    L.record_mean('CorrectRatio/Final', correct_ratio.item())
    L.record_tb('ConfusionMatrix/Final', L.Image(plot_matric(conf_matric_final), 'HWC'))
    L.record_tb('Rewards', rews.flatten()[to_numpy(all_data.mask.flatten()) > 0.5])
    print('rews: ', rews.mean(), rews.std())
    if step >= 0:
        L.dump(step=step)


def main(args):
    cfg = oc.DictConfig(defalut_cfg)
    cfg_cli = oc.OmegaConf.from_dotlist(args.opts)
    cfg = oc.OmegaConf.merge(cfg, cfg_cli)
    L.configure(cfg.logdir, ['stdout', 'tensorboard'])
    oc.OmegaConf.save(cfg, os.path.join(cfg.logdir, 'config.yaml'))
    # disc = RefTrajEmbed(cfg)
    disc = globals()[cfg.disc_type](cfg)
    print(cfg, cfg.device)
    print(disc)
    disc.to(cfg.device)
    all_data = load_all_data(cfg.data_dir, cfg)
    print(all_data)
    N, T = all_data.obs.view.shape[:2]
    batch_size = 128
    optim = torch.optim.Adam(disc.parameters(), lr=1e-3)
    for epoch in range(cfg.epoch):
        with tqdm.tqdm(total=N, desc=f'Epoch #{epoch}', **tqdm_config) as t:
            for b in all_data.split(batch_size):
                b = preprocess_fn(b, cfg)
                # omit the first frame
                traverseBatch(b, {'tensor': lambda x: x[:,1:]})
                rew, loss, _, other_info = disc.label_rewards(b, masks=b.mask)
                optim.zero_grad()
                loss.backward()
                optim.step()
                L.record('rew_dist', to_numpy(rew).flatten()[to_numpy(b.mask).flatten() > 0.5])
                L.record_mean('loss', loss.item())
                L.record_mean('return', rew.sum(1).mean().item())
                L.record_mean('targp/max', other_info['max_targ_p'])
                L.record_mean('targp/mean', other_info['mean_targ_p'])
                t.update(batch_size)
                t.set_postfix(Ret=rew.sum(1).mean().item(), Loss=loss.item(), TargPMax=other_info['max_targ_p'])
        torch.save(disc.state_dict(), os.path.join(cfg.logdir, 'disc.pt'))
        if cfg.eval_data_dir != '':
            args.opts = [f'loaddir={cfg.logdir}/disc.pt', f'logdir={cfg.logdir}/eval',  f'eval_data_dir={cfg.eval_data_dir}']
            eval(args, disc=disc, step=-1)
        L.dump(step=epoch)
    return disc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='main', type=str)
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    globals()[args.mode](args)
