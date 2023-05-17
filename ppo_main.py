import argparse
import itertools
import datetime
import torch.nn as nn
import os
import pprint

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
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
# from dqn import traverseBatch
import copy
from hunter_game import HunterObjEnv
from hunter_game import HunterObjEnv as HunterObjExtenedEnv
from graph_gru import *

def main(args):
    device = 'cpu' 
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = 'cuda' 
    cfg = dict(num_cat=12, num_act=9, 
        env_kwargs=dict(symbolic=False, max_len=30, clear_reward=0, zombie_movable=True, avoid_agent=True, fire_range=3, spawn_args='Z4C4'),
        hidden_dim=32, info_spec = dict(), 
        cat_to_expl=6,
        device=device,
        patch_shape=(8,8,3),
        aim_log=False, aim_tags=[], n_step=1, thres=0.3, objto=19, max_num=256,
        exp_only=False, load_task_path='', task_only=True,
        lr = 3e-4, num_envs=16,
        epoch=500, step_per_epoch=51200, repeat_per_collect=3, step_per_collect=4096,
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
        seed = -1,
        view_type='rn',
        eval_only = False,
        ret_img_when_eval = True,
        use_vec_traj_to_save = True,
        eval_on_test = False,
        save_hist = True,
        save_model_all=False,
        logdir = 'ppo_task',
        loaddir = './max/s3399937/all_model.pth',
        sinfo_v2=True,
    )
    cfg = oc.DictConfig(cfg)
    cfg_cli = oc.OmegaConf.from_dotlist(args.opts)
    cfg = oc.OmegaConf.merge(cfg, cfg_cli)
    args = cfg
    if args.seed == -1:
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
    if args.eval_only:
        args.epoch = 1
        args.step_per_epoch = 1_024_000 if not cfg.eval_on_test else 20480
        args.step_per_collect = 4096
        args.repeat_per_collect = 0
        args.num_envs = 16
        para = torch.load(args.loaddir, map_location=args.device)
        print(list(para['model'].keys()))
        model.load_state_dict(para['model'])
        actor.load_state_dict(para['actor'])
        critic.load_state_dict(para['critic'])
    optim = torch.optim.Adam(itertools.chain(model.parameters(), actor.parameters(), critic.parameters()), args.lr)
    def dist(x):
        return torch.distributions.Categorical(logits=x)
    policy = PPOPolicy(actor, critic, optim, dist, **args.ppo_kwargs)
    
    VecEnvCls = DummyVectorEnv
    train_envs = VecEnvCls([lambda: HunterObjEnv(args, version=0, ret_img=args.eval_only and args.ret_img_when_eval) for _ in range(args.num_envs)])
    test_envs = VecEnvCls([lambda: HunterObjEnv(args, version=0, ret_img=args.eval_only and args.ret_img_when_eval) for _ in range(1)])
    if args.use_vec_traj_to_save or not args.eval_only:
        buf = VecHistReplayBuffer(len(train_envs), args.buffer_size,
                              1_024_000 if not args.eval_only else args.step_per_epoch // 2, 
                              args.logdir if args.save_hist else '',
                              save_img=args.eval_only and not cfg.eval_on_test and args.ret_img_when_eval,
                              maxlen=32)
    else:
        buf = VectorReplayBuffer(args.step_per_epoch + 4096, len(train_envs))
    train_collector = Collector(
        policy, train_envs,
        buf,
    )
    L.configure(log_path, ['csv', 'tensorboard', 'stdout'])
    oc.OmegaConf.save(cfg, os.path.join(log_path, 'config.yaml'))
    if args.eval_only:
        res = train_collector.collect(n_step=args.step_per_epoch)
        print(res)
        data = buf.sample(0)[0]
        data.obs = data.obs.view.argmax(-1)
        data.obs_next = data.obs_next.view.argmax(-1)
        data.info = Batch()
        buf = ReplayBuffer(len(data.obs))
        buf.set_batch(data)
        buf._size = len(data.obs)
        buf.save_hdf5(os.path.join(args.logdir, f'buffer_size{args.step_per_epoch}_ret{res["rew"]:.1f}.hd5'))
        return
        
    test_collector = Collector(policy, test_envs)
    # log
    def save_fn(policy, epoch):
        if args.save_model_all and (epoch % (args.epoch // 5) == 0):
            torch.save(dict(model=model.state_dict(), actor=actor.state_dict(), critic=critic.state_dict()),
                   os.path.join(args.logdir, f'all_model_epoch{epoch}.pth'))
        torch.save(dict(model=model.state_dict(), actor=actor.state_dict(), critic=critic.state_dict()),
                   os.path.join(args.logdir, 'all_model.pth'))
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
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()
    main(args)
