import argparse
import tqdm
import itertools
import datetime
import torch.nn as nn
import os
import pprint

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import VectorReplayBuffer
from collector import Collector
from tianshou.policy import PPOPolicy
from onpolicy import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from stable_baselines3.common import logger as L
import omegaconf as oc
from graph_gru import ObsEmbed
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from traj_buf import VecHistReplayBuffer, VecTrajectoryReplayBuffer
import copy
from hunter_game import HunterObjEnv
from graph_gru import *

from percept import *

class PerceptionRelabel:
    def __init__(self, cfg, all_data, refs):
        self.cfg = cfg
        self.refs = refs
        self.percept = PerceptionV2(cfg)
        obs = all_data.obs_next
        oimgs = to_numpy(all_data.obs_next.image).reshape(-1, 64, 64, 3)
        ocats = to_numpy(all_data.policy.ori_obs_next_view).reshape(-1, 8, 8, cfg.num_cat).argmax(-1)
        ocats = np.eye(cfg.num_cat)[self.refs[ocats]]
        self.percept.adapt(oimgs, ocats)
    def __call__(self, data=None, obs=None, state=None, **kwargs): 
        if data is not None:
            data.obs_next.view = self.percept.relabel_seen(data.obs_next.image)
            data.obs.view = self.percept.relabel_seen(data.obs.image)
        elif obs is not None:
            data = Batch(obs=obs)
            data.obs.view = self.percept.relabel_seen(data.obs.image)
        return data            

class ExplAndTest:
    def __init__(self, cfg, exp_col, exp_buf, task_col):
        self.cfg = cfg
        self.exp_col = exp_col # assert collector.env.num_envs == 1
        self.exp_buf = exp_buf
        self.task_col = task_col
    def get_relabel_fn(self, n_episode, other_data=None):
        self.exp_buf.reset()
        self.exp_col.reset()
        self.exp_col.collect(n_episode=n_episode)
        all_data = self.exp_buf.sample(0)[0][:n_episode]
        all_data = Batch(**{k:all_data[k] for k in ['obs', 'obs_next', 'policy']})
        all_data = to_torch(all_data, dtype=torch.float32, device=self.cfg.device)
        if other_data is not None:
            all_data = Batch.cat([all_data, other_data])
        B, T, H, W, C = all_data.obs.view.shape
        final_idx = (1 - all_data.policy.valid_mask.reshape(B,T)).argmax(1)
        probs = all_data.policy.probs.reshape(B, T, C, C)
        # N, C, C
        final_probs = probs.gather(1,
            einops.repeat(final_idx, 'b -> b 1 c1 c2', c1=C, c2=C)).squeeze(1)
        print('final_probs', final_probs[0, self.cfg.cat_to_expl:].max(-1))
        refs = torch.mean(final_probs, 0).argmax(-1)
        refs[:self.cfg.cat_to_expl] = torch.arange(self.cfg.cat_to_expl)
        print(refs)
        return PerceptionRelabel(self.cfg, all_data, to_numpy(refs)), all_data
    def expl_and_test(self, expl_episodes, test_episodes):
        print('Eval and Test...')
        data = None
        all_res = []
        for ep in expl_episodes:
            relabel_fn, data = self.get_relabel_fn(ep, data)
            self.task_col.reset()
            self.task_col.preprocess_fn = relabel_fn
            res = self.task_col.collect(n_step=test_episodes * 32)
            print(ep, 'Ret:', res['rew'])
            all_res.append(res)
        return all_res, data
                  
def get_task_pol(cfg):
    cfg = cfg.copy()
    model_dir = cfg.load_task_pol_dir
    model = ObsEmbed(cfg, out_dim=cfg.hidden_dim, view_type='rn', use_mlp=False)
    output_dim = cfg.hidden_dim
    class ActorNet(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.out = nn.Linear(output_dim, out_dim)
        def forward(self, obs, **kwargs):
            if not isinstance(model, ObsEmbed):
                obs = to_torch(obs['image'], dtype=torch.float32, device=cfg.device)
                obs = obs.permute(0, 3, 1, 2) / 255.
                res  = self.out(model(obs))
            else:
                obs = to_torch(obs, dtype=torch.float32, device=cfg.device)
                res  = self.out(model(obs)[0])
            if 'state' in kwargs:
                return res, None
            else:
                return res
    actor = ActorNet(cfg.num_act)
    critic = ActorNet(1)
    para = torch.load(model_dir, map_location=cfg.device)
    model.load_state_dict(para['model'])
    actor.load_state_dict(para['actor'])
    critic.load_state_dict(para['critic'])
    optim = torch.optim.Adam(itertools.chain(model.parameters(), actor.parameters(), critic.parameters()), cfg.lr)
    def dist(x):
        return torch.distributions.Categorical(logits=x)
    policy = PPOPolicy(actor, critic, optim, dist)
    return policy.to(cfg.device)


def get_percept(cfg, env_fn, pol, load=True):
    percept = PerceptionV2(cfg)
    if load:
        percept.load('pretrain_models/percept')
    else:
        envs = DummyVectorEnv([env_fn])
        buf = VecTrajectoryReplayBuffer(1, 8, 32)
        col = Collector(pol, envs, buf) 
        col.collect(n_step=32*2, random=True)
        all_data = Batch(buf.sample(0)[0])
        oimgs, ocats = all_data.obs['image'].reshape(-1,64,64,3), all_data.obs['view'].reshape(-1,8,8,cfg.num_cat)
        res = percept.adapt(oimgs, ocats, epoch=512*4)
        print(res)
        percept.save(f'percept_model')        
    return percept
            
class DiscWrapper(nn.Module):
    def __init__(self, cfg, disc, update_freq=2, percept=None):
        super().__init__()
        self.disc = disc
        self.percept = percept
        self.use_percept = False
        self.record_data = True
        self.cfg = cfg
        self.update_freq = update_freq
        self.episode_ret = None
        if update_freq > 0:
            self.buf = VecTrajectoryReplayBuffer(cfg.num_envs, 16384, 32)
            self.optim = torch.optim.Adam(self.disc.parameters(), lr=3e-4)
            self._iter = 0
        else:
            self._iter = 0
            self.buf = None
        self.to(cfg.device)
    def acc_ret(self, rew, done):
        if self.episode_ret is None:
            self.episode_ret = rew
        else:
            self.episode_ret += rew
        ret = self.episode_ret
        done = to_numpy(done)
        self.episode_ret = self.episode_ret * (1-done)
        episodic_rew = ret * done
        return episodic_rew
    @torch.no_grad()
    def relabel(self, data, state):
        if 'policy' not in data:
            data.policy = Batch()
        data = to_torch(data, dtype=torch.float32, device=self.cfg.device)
        if state is not None:
            state = to_torch(state, dtype=torch.float32, device=self.cfg.device)
        B, H, W, C = data.obs_next.view.shape
        b = Batch(obs=copy.deepcopy(data.obs_next)) # , device=self.cfg.device, dtype=torch.float32)
        if self.use_percept:
            b.obs.view = self.percept.relabel_unseen(b.obs.image, b.obs.view)
            b.obs.prev_view = self.percept.relabel_unseen(b.obs.prev_image, b.obs.prev_view)
            data.obs_next.view = to_torch_as(self.percept.relabel_unseen(data.obs_next.image, data.obs_next.view), data.obs_next.view)
            data.obs_next.prev_view = to_torch_as(self.percept.relabel_unseen(data.obs_next.prev_image, data.obs_next.prev_view), data.obs_next.prev_view)
        batch_unflatten(b, B, 1)
        mask = torch.ones(B, 1)
        rews, loss, state, other_info = self.disc.label_rewards(b, mask, state)
        prob_refs = other_info['prob_refs'].squeeze(1)
        data.rew = rews.squeeze(-1)
        data.policy.probs = other_info['probs'].squeeze(1)
        data.state = state
        data.policy.hidden_state = state
        data.policy.ori_obs_next_view = data.obs_next.view
        data.obs_next.view = (data.obs_next.view.reshape(B, H*W, C) @ prob_refs).reshape(B, H, W, C)
        return to_numpy(data)
    def update_disc(self, data):
        if self.buf is not None and self.record_data:
            self.buf.add(data)
            self._iter += 1
            if self._iter % self.update_freq == 0 and len(self.buf) >= 512:
                batch, inds = self.buf.sample(32)
                batch = to_torch(batch, dtype=torch.float32, device=self.cfg.device)
                # obs = batch.obs[:,1:]
                # BYPASS the first frame.
                rew, loss, state, other_info = self.disc.label_rewards(Batch(obs=batch.obs[:,1:]), batch.policy.valid_mask[:,1:])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                L.record('disc/rew_dist', rew)
                L.record_mean('disc/exp/ret', rew.sum(1).mean())
                L.record_mean('disc/loss', loss.item())
                L.record_mean('disc/max_targ_p', other_info['max_targ_p'])
                L.record_mean('disc/mean_targ_p', other_info['mean_targ_p'])
    def preprocess_fn(self, data=None, obs=None, state=None, **kwargs):
        self.record_data = False
        if data is not None:
            self.update_disc(data)
            data = self.relabel(data, state)
        elif obs is not None:
            record_data = self.record_data
            self.record_data = False
            data = Batch(obs_next=obs, done=torch.zeros(obs.shape[0]))
            data = self.relabel(data, None)
            data = Batch(obs=data.obs_next)
            self.record_data = record_data
        return data

def main(args):
    device = 'cpu' 
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = 'cuda' 
    cfg = dict(num_cat=12, num_act=9, 
        env_kwargs=dict(symbolic=False, max_len=30, clear_reward=0, zombie_movable=True, avoid_agent=True, fire_range=3, spawn_args='Z3C3'),
        hidden_dim=32, info_spec = dict(), 
        cat_to_expl=6,
        device=device,
        patch_shape=(8,8,3),
        aim_log=False, aim_tags=[], n_step=1, thres=0.3, objto=19, max_num=256,
        exp_only=False, load_task_path='', task_only=True,
        lr = 3e-4, num_envs=16,
        epoch=200, step_per_epoch=51200, repeat_per_collect=3, step_per_collect=4096,
        batch_size=256, buffer_size=8192,
        ppo_kwargs = dict(
          discount_factor=0.9,
          max_grad_norm=0.5,
          eps_clip=0.2,
          vf_coef=0.5,
          ent_coef=0.01,
          gae_lambda=0.95,
          deterministic_eval=False,
          advantage_normalization=1,
        ),
        update_freq=4,
        view_type='rn',
        eval_only = False,
        eval_on_test = False,
        save_hist = False,
        load_disc=True,
        load_disc_dir=f'pretrain_models/disc/Z3C3/disc.pt',
        load_task_pol_dir='pretrain_models/policy/Z3C3/s4801973/all_model.pth',
        logdir=f'ppo_expl',
        sinfo_v2=True,
    )
    cfg = oc.DictConfig(cfg)
    cfg_cli = oc.OmegaConf.from_dotlist(args.opts)
    cfg = oc.OmegaConf.merge(cfg, cfg_cli)
    args = cfg
    args.seed = np.random.randint(1, 10000000)
    log_path = os.path.join(args.logdir, f's{args.seed}')
    args.logdir = log_path
    print('======== Config ========')
    print(args)
    print('======== Config ========')

    model = ObsEmbed(args, out_dim=cfg.hidden_dim, view_type=cfg.view_type, use_mlp=False)
    output_dim = cfg.hidden_dim
    class ActorNet(nn.Module):
        def __init__(self, out_dim):
            super().__init__()
            self.out = nn.Linear(output_dim, out_dim)
        def forward(self, obs, **kwargs):
            if not isinstance(model, ObsEmbed):
                obs = to_torch(obs['image'], dtype=torch.float32, device=args.device)
                obs = obs.permute(0, 3, 1, 2) / 255.
                res  = self.out(model(obs))
            else:
                obs = to_torch(obs, dtype=torch.float32, device=args.device)
                # obs.image = obs.image / 255.
                res  = self.out(model(obs)[0])
            if 'state' in kwargs:
                return res, None
            else:
                return res

    actor = ActorNet(args.num_act)
    critic = ActorNet(1)
    print(model)
    print(actor)
    print(critic) 
    if cfg.load_disc:
        print('Loading Disc from ', cfg.load_disc_dir)
        disc_cfg = oc.OmegaConf.load(cfg.load_disc_dir.replace('disc.pt', 'config.yaml'))
        disc_cfg.device = cfg.device
        disc = RefTrajEmbed(disc_cfg)
        disc.load_state_dict(torch.load(cfg.load_disc_dir, map_location=cfg.device))
        cfg.update_freq = -1 # freeze disc
    else:
        disc = RefTrajEmbed(cfg)
    disc = DiscWrapper(cfg, disc, cfg.update_freq)
    optim = torch.optim.Adam(itertools.chain(model.parameters(), actor.parameters(), critic.parameters()), args.lr)
    def dist(x):
        return torch.distributions.Categorical(logits=x)
    policy = PPOPolicy(actor, critic, optim, dist, **args.ppo_kwargs)
    task_pol = get_task_pol(cfg) 
    percept = get_percept(cfg, lambda: HunterObjEnv(args, version=0, ret_img=True), task_pol)
    disc.percept = percept
    if args.eval_only:
        args.epoch = 1
        args.step_per_epoch = 256 
        args.step_per_collect = 256
        args.repeat_per_collect = 0
        args.num_envs = 4

    VecEnvCls = DummyVectorEnv
    train_envs = VecEnvCls([lambda: HunterObjEnv(args, version=3, env_version=0, ret_img=args.eval_only) for _ in range(args.num_envs)])
    test_envs = VecEnvCls([lambda: HunterObjEnv(args, version=2, env_version=1, ret_img=args.eval_only) for _ in range(1)])
        
    buf = VecHistReplayBuffer(len(train_envs), args.buffer_size,
                              1_024_000 if not args.eval_only else args.step_per_epoch // 2, 
                              args.logdir if args.save_hist else '',
                              save_img=args.eval_only and not cfg.eval_on_test,
                              maxlen=32)

    
    disc.record_data = False
    train_collector = Collector(
        policy, train_envs,
        buf,
        preprocess_fn=disc.preprocess_fn,
    )
    disc.record_data = True
    test_collector = Collector(policy, test_envs)
    # log
    L.configure(log_path, ['csv', 'tensorboard', 'stdout'])
    oc.OmegaConf.save(cfg, os.path.join(log_path, 'config.yaml'))

    class MyDict(dict):
      def __getattr__(self, key):
        return self.get(key, None)

    from train_disc import eval
    test_eval_envs = DummyVectorEnv([lambda: HunterObjEnv(cfg, version=0, env_version=3, ret_img=True) for _ in range(1)])
    test_task_envs = DummyVectorEnv([lambda: HunterObjEnv(cfg, version=0, env_version=3, ret_img=True) for _ in range(4)])
    test_eval_buf = VecTrajectoryReplayBuffer(1, 32, 32)
    disc.record_data = False
    test_eval_collector = Collector(policy, test_eval_envs, test_eval_buf, exploration_noise=True, preprocess_fn=disc.preprocess_fn)
    disc.record_data = True
    test_task_collector = Collector(task_pol, test_task_envs)
    expl_and_test = ExplAndTest(cfg, test_eval_collector, test_eval_buf, test_task_collector)
    def save_fn(policy, epoch):
        torch.save(dict(model=model.state_dict(), actor=actor.state_dict(), critic=critic.state_dict(), disc=disc.state_dict()), os.path.join(args.logdir, 'all_model.pth'))
        print('Evaluation...')
        disc.use_percept = True
        all_res, all_data = expl_and_test.expl_and_test([1,1,2,12], 8)
        for k, res in zip([1,2,4,16], all_res): 
            L.record(f'Eval{k}', res['rew'])
        disc.use_percept = False 
        all_data.obs.view[:,1:] = all_data.policy.ori_obs_next_view[:,:-1]
        all_data.obs_next.view = all_data.policy.ori_obs_next_view
        eval(MyDict(opts=[f'logdir={cfg.logdir}/eval', f'num_cat={cfg.num_cat}']), disc=disc.disc, all_data=all_data, step=-1, preprocess=False, version=1)

    if False and cfg.eval_only:
        save_fn(None, 0)
        info_dict = L.get_log_dict()
        res = {k: info_dict[k] for k in ['CorrectRatio/Final', 'Eval1', 'Eval2', 'Eval4']}
        print(res)
        return res
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        max(8, len(test_envs)),
        args.batch_size,
        step_per_collect=args.step_per_collect,
        save_fn=save_fn,
        L=L,
        test_in_train=False,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='main', type=str)
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    globals()[args.mode](args)
