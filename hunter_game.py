# from dis import Instruction
import numpy as np
import gym
import torch
import os
import matplotlib.image as mpimg
from PIL import Image
import functools
import matplotlib
# import dgl
import omegaconf as oc
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random
import gym
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
import einops
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans  #, AffinityPropagation

def move_away(check_bound, fire_range, ex, ey, ax, ay):
  probs_map = {}
  dist = abs(ex - ax) + abs(ey - ay)
  in_fire_range = abs(ex - ax) <= fire_range and abs(ey - ay) <= fire_range 
  for a_type in ['U', 'D', 'L', 'R']:
    try_pos = ex - int(a_type == 'U') + int(a_type == 'D'), ey - int(a_type=='L') + int(a_type=='R') 
    if check_bound(try_pos) and in_fire_range:
      try_dist = abs(try_pos[0] - ax) + abs(try_pos[1] - ay)
      if try_dist > dist:
        probs_map[a_type] = 4
      else:
        probs_map[a_type] = 1
    else:
      probs_map[a_type] = 1
  probs = [probs_map[k] for k in ['U', 'D', 'L', 'R']]
  total = float(sum(probs))
  probs = [float(p) / total for p in probs]
  return probs

class Hunter(gym.Env):
  def __init__(self, area=(8,8), img_size=(64,64),
               spawn_args = 'Z4C4',
               max_len=32, symbolic=True,
               ret_raw_state=False, zombie_movable=True, avoid_agent=True,
               clear_reward=0, fire_range=3,
               hard_obj=False, fix_len=False,
               abs_rew = False,
               no_wall=False, version=0, maintain_graph=False,  **kwargs):
    self.abs_rew = abs_rew
    self.area = area
    self.hard_obj = hard_obj
    self.fix_len = fix_len
    self.max_len = max_len
    self.cur_len = 0
    self.img_size = img_size
    self.clear_reward = clear_reward
    random_sample_args = [(int(v[1]), int(v[3])) for v in spawn_args.split('/')]
    self.avoid_agent = avoid_agent
    print(f'ENV Random Sample Args: {random_sample_args}')
    self.symbolic = symbolic
    self.ret_raw_state = ret_raw_state
    # zombie_movable = True
    self.zombie_movable = zombie_movable
    self.fire_range = fire_range
    self.no_wall = no_wall
    self.random_sample_args = random_sample_args
    self.state = np.zeros(area)
    self.next_id = 1
    self.obj_id = np.zeros(area)
    actions = ['NONE', 'U', 'D', 'L', 'R', 'FU', 'FD', 'FL', 'FR']
    self.act_map = {k:v for k,v in zip(np.arange(len(actions)), actions)}
    self.object_map = {'zombie': [], 'cow': [], 'agent':[]}
    self.idx2obj_map = {k:v for k, v in zip(np.arange(6), ['none', 'zombie', 'agent', 'cow', 'wall', 'unknown'])}
    self.obj2idx_map = {v:k for k, v in zip(np.arange(6), ['none', 'zombie', 'agent', 'cow', 'wall', 'unknown'])}
    self.block_size = img_size[0] // self.area[0], img_size[1] // self.area[1]
    self.version = version
    self.maintain_graph = maintain_graph
    self.textures = self.load_textures()
    o = self.reset()
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=o.shape)
    self.action_space = gym.spaces.Discrete(len(actions))
  def seed(self, seed):
    pass
  def load_textures(self):
    if self.version == 0:
      assets_list = ['sand', 'zombie', 'player', 'cow', 'stone',  'unknown']
      assets = {k: os.path.join('assets', f'{v}.png')
                  for k,v in zip(['none', 'zombie', 'agent', 'cow', 'wall', 'unknown'],
                                 assets_list)}
      textures = {k:np.asarray(Image.open(v).resize(self.block_size, Image.ANTIALIAS)) for k,v in assets.items()}
    elif self.version == 1:
      assets_list = ['sand', 'zombie', 'player', 'cow', 'stone',  'unknown']
      assets = {k: os.path.join('assets', f'{v}.png')
                  for k,v in zip(['none', 'zombie', 'agent', 'cow', 'wall', 'unknown'],
                                 assets_list)}
      textures = {k:np.asarray(Image.open(v).resize(self.block_size, Image.ANTIALIAS)) for k,v in assets.items()}
      item_list = ['zombie', 'agent', 'cow']
      color_list = np.asarray([[0,0,1], [0,1,0], [1,0,0], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]) * 128
      for item, color in zip(item_list, color_list):
        textures[item] = np.ones((*self.block_size, 3)) * color
      # textures = {k:np.ones((*self.block_size, 3)) * color_list[i] for i, k in enumerate(item_list)}
    elif self.version == 2:
      assets_list = ['sand', 'zombie', 'player', 'cow', 'stone',  'unknown']
      assets = {k: os.path.join('assets', f'{v}.png')
                  for k,v in zip(['none', 'zombie', 'agent', 'cow', 'wall', 'unknown'],
                                 assets_list)}
      textures = {k:np.asarray(Image.open(v).resize(self.block_size, Image.ANTIALIAS)) for k,v in assets.items()}
      for item in ['zombie', 'agent', 'cow']:
        # print(item, textures[item].shape, textures[item].max())
        textures[item][...,:3] = 255 -  textures[item][...,:3]
        # textures[item] = np.concatenate([255 - textures[item][...,:3], textures[item][...,-1:]], axis=-1)
    elif self.version == 3:
      assets_list = ['path', 'skeleton', 'Caveman', 'cow4', 'water',  'unknown']
      assets = {k: os.path.join('assets', f'{v}.png')
                  for k,v in zip(['none', 'zombie', 'agent', 'cow', 'wall', 'unknown'],
                                 assets_list)}
      import imageio, pathlib

      textures = {k:np.asarray(Image.open(v).resize(self.block_size, Image.ANTIALIAS)) for k,v in assets.items()}
      for k in ['agent', 'cow']:
        image = imageio.imread(pathlib.Path(assets[k]).read_bytes())
        textures[k] = np.asarray(Image.fromarray(image).resize(self.block_size))
      textures['none'] = textures['none'] * 0 + np.asarray([71, 108, 108, 255])
      # print(textures['agent'])
      print({k:v.shape for k,v in textures.items()})
    else:
      assert False
    return textures

  def draw_item(self, canvas, pos, obj_type, ret_patch=False):
    sx, sy = pos
    bx, by = self.block_size
    obj_texture = self.textures[obj_type]
    if obj_texture.shape[-1] == 3:
      canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] = obj_texture
    elif obj_texture.shape[-1] == 4:
      cur_texture = canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] 
      alpha = obj_texture[...,-1:] / 255.
      obj_texture = obj_texture[...,:-1]
      canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] = cur_texture * (1-alpha) + alpha * obj_texture 
    elif len(obj_texture.shape) == 2:
      canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)] = obj_texture.reshape([bx, by, 1])
    else:
      assert False
    if ret_patch:
      return canvas, canvas[bx*sx:bx*(sx+1), by*sy:by*(sy+1)].copy()
    return canvas
    
  @functools.lru_cache(10)
  def background(self):
    canvas = np.zeros([*self.img_size, 3])
    for i in range(self.area[0]):
      for j in range(self.area[1]):
        canvas = self.draw_item(canvas, (i,j), 'none')
    return canvas
    
  def render(self, mode='rgb', state=None, obj_id=None, print_out=False, ret_obj_patch=False, force_rgb=False):
    if force_rgb:
      mode = 'rgb'
    if self.ret_raw_state:
      return self.state.copy()
    if (self.symbolic or mode == 'symbolic') and mode != 'tty_symbolic' and not force_rgb:
      return np.eye(len(self.idx2obj_map))[self.state.copy().astype(np.int)].flatten()
    elif mode == 'tty_symbolic':
      array = []
      state = state if state is not None else self.state
      idx2symb_map = {k:v for k, v in zip(np.arange(6), [' ', 'Z', 'A', 'C', 'W', 'U'])}
      for si in state:
        array.append([])
        for sij in si:
          array[-1].append(idx2symb_map[sij])
      # if self.maintain_graph:
      #   print(self.graph_input)
      if print_out:
        print('#'*10)
        for a in array:
          print(a)
        print('#'*10)
      return np.asarray(array)
    elif mode == 'rgb':
      canvas = self.background().copy()
      state = self.state if state is None else state
      patches = []
      for i in range(self.area[0]):
        rows = []
        for j in range(self.area[1]):
          obj_type = self.idx2obj_map[state[(i,j)]]
          if obj_type in ['zombie', 'agent', 'cow', 'wall', 'unknown']:
            canvas, patch = self.draw_item(canvas, (i,j), obj_type, ret_patch=True)
      return (canvas).astype(np.uint8)
    else:
      assert False
      
  def random_spawn(self, obj_type):
    pos = np.random.randint(0, self.area[0]), np.random.randint(0, self.area[1])
    if obj_type in ['cow', 'zombie']:
      ax, ay = self.object_map['agent'][0]
      in_fire_range = self.avoid_agent and abs(pos[0] - ax) <= self.fire_range and abs(pos[1] - ay) <= self.fire_range
    else:
      in_fire_range = False 
    if self.state[pos] != 0 or in_fire_range:
      # print('collision', obj_type, pos, self.state[pos], (ax, ay), in_fire_range)
      self.random_spawn(obj_type)
    else:
      self.state[pos] = self.obj2idx_map[obj_type]
      self.object_map[obj_type].append(pos)
      self.obj_id[pos] = self.next_id
      self.next_id += 1
  def random_wall(self):
    if self.no_wall:
      return
    pos1 = np.random.randint(1, self.area[0]//2-1), np.random.randint(1, self.area[1]//2-1)          # (1,2,3)
    pos2 = np.random.randint(self.area[0]//2, self.area[0] - 1), np.random.randint(self.area[1] // 2, self.area[1] - 1) # (4,5,6)
    dir1 = np.random.randint(0,4)
    dir2 = np.random.randint(0,4)
    wall_idx = self.obj2idx_map['wall']
    self.all_wall = np.zeros((2,2,2), dtype=np.int)
    for n, (d, pos) in enumerate(zip([dir1, dir2],[pos1, pos2])):
      if d == 0:
        self.state[pos[0]:, pos[1]] = wall_idx
        length = self.area[0] - pos[0]
        self.obj_id[pos[0]:, pos[1]] = np.arange(self.next_id, self.next_id + length)
        self.next_id += length
        self.all_wall[n, 0] = pos
        self.all_wall[n, 1] = [self.area[0]-1, pos[1]]
      elif d == 1:
        self.state[:pos[0]+1, pos[1]] = wall_idx
        length = pos[0] + 1
        self.obj_id[:pos[0]+1, pos[1]] = np.arange(self.next_id, self.next_id + length)
        self.next_id += length
        self.all_wall[n, 0] = [0, pos[1]] 
        self.all_wall[n, 1] = pos
      elif d == 2:
        self.state[pos[0], pos[1]:] = wall_idx
        length = self.area[1] - pos[1]
        self.obj_id[pos[0], pos[1]:] = np.arange(self.next_id, self.next_id + length)
        self.next_id += length
        self.all_wall[n, 0] = pos 
        self.all_wall[n, 1] = [pos[0], self.area[1]-1]
      elif d == 3:
        self.state[pos[0], :pos[1]+1] = wall_idx
        length = pos[1] + 1
        self.obj_id[pos[0], :pos[1]+1] = np.arange(self.next_id, self.next_id + length)
        self.next_id += length
        self.all_wall[n, 0] = [pos[0], 0] 
        self.all_wall[n, 1] = pos
      else:
        assert False
    if dir1 == 0 and dir2 == 3:
      self.state[pos2[0],:pos1[1]] = 0
      self.obj_id[pos2[0],:pos1[1]] = 0
      self.all_wall[1,0] = [pos2[0], pos1[1]]
    elif dir1 == 2 and dir2 == 1:
      self.state[:pos1[0],pos2[1]] = 0
      self.obj_id[:pos1[0],pos2[1]] = 0
      self.all_wall[1,0] = [pos1[0], pos2[1]]
    
  def text_obs(self):
    text = f'You are currently at {self.cur_len}th time step in the episode.'
    text += f'You are at position {tuple(self.object_map["agent"][0])}.'
    text += f'There is a wall from {tuple(self.all_wall[0,0])} to {tuple(self.all_wall[0,1])}. '
    text += f'There is another wall from {tuple(self.all_wall[1,0])} to {tuple(self.all_wall[1,1])}. '
    for name in ['zombie', 'cow']:
      objs = self.object_map[name]
      if len(objs) == 1:
        text += f'There is a {len(objs)} {name} at {objs[0]}. '
      else:
        text += f'There are {len(objs)} {name}s at '\
                + ','.join([str(o) for o  in objs])\
                + ' respectively. '
    return text
  def reset(self):
    self.cur_len = 0
    self.next_id = 1
    self.state = np.zeros(self.area)
    self.obj_id = np.zeros(self.area)
    self.object_map = {'zombie': [], 'cow': [], 'agent':[]}
    self.random_wall()
    self.random_spawn('agent')
    self.next_done = False
    num_zombie, num_cow = random.choice(self.random_sample_args)
    for i in range(num_zombie):
      self.random_spawn('zombie')
    for i in range(num_cow):
      self.random_spawn('cow')
    # print(self.object_map)
    if self.maintain_graph:
      self.create_graph()
    return self.render()
  def handle_collosion(self, pos, try_pos):
    valid = True
    die   = False
    if try_pos[0] < 0 or try_pos[0] >= self.area[0] or try_pos[1] < 0 or try_pos[1] >= self.area[1]:
      valid = False
    targ_type = self.idx2obj_map(self.state[try_pos])
  def check_bound(self, try_pos): 
    return not(try_pos[0] < 0 or try_pos[0] >= self.area[0] or try_pos[1] < 0 or try_pos[1] >= self.area[1])
  def remove(self, pos, check_hard=True):
    obj = self.idx2obj_map[self.state[pos]]
    if check_hard and self.hard_obj:
      return
    assert obj in ['agent', 'cow', 'zombie']
    assert pos in self.object_map[obj]
    self.state[pos] = 0
    self.obj_id[pos] = 0
    self.object_map[obj] = list(set(self.object_map[obj]) - set([pos]))
    
  def move(self, pos, pos2):
    o1= self.state[pos]
    obj = self.idx2obj_map[o1]
    self.state[pos2] = o1
    self.obj_id[pos2] = self.obj_id[pos]
    if obj in self.object_map:
      self.remove(pos, check_hard=False)
      self.object_map[obj].append(pos2) #= list(set(self.object_map[obj]) - set([pos]))

    
  def step(self, a):
    a = int(a)
    ag_pos = self.object_map['agent']
    assert len(ag_pos) == 1
    ag_pos = ag_pos[0]
    a_type = self.act_map[a]
    reward = 0
    done = False
    if a_type in ['U', 'D', 'L', 'R']:
      try_pos = ag_pos[0] - int(a_type == 'U') + int(a_type == 'D'), ag_pos[1] - int(a_type=='L') + int(a_type=='R') 
      if self.check_bound(try_pos):
        obj_type = self.idx2obj_map[self.state[try_pos]]
        if obj_type  == 'none':
          self.move(ag_pos, try_pos)
        elif obj_type == 'zombie':
          reward -= 1
          self.remove(try_pos)
          self.move(ag_pos, try_pos)
          # done = True
        elif obj_type == 'cow':
          self.remove(try_pos)
          self.move(ag_pos, try_pos)
          reward += 1
    elif 'F' in a_type:
      ax, ay = ag_pos
      rew_map = {'zombie':1, 'cow':-1}
      # rew_map = {'zombie':1, 'cow':0}
      for k, r in rew_map.items():
        for ex, ey in self.object_map[k]:
          assert (ex, ey) != (ax, ay)
          destroyed = False
          if ex == ax:
            destroyed = (ey > ay and a_type == 'FR') or (ey < ay  and a_type == 'FL')
          if ey == ay:
            destroyed = (ex > ax and a_type == 'FD') or (ex < ax  and a_type == 'FU')
          if destroyed and abs(ex - ax) <= self.fire_range and abs(ey - ay) <= self.fire_range:
            self.remove((ex, ey))
            reward += r
    # randomly move enemies
    if self.zombie_movable and self.cur_len % 2 == 1:
      for obj_type in np.random.choice(['zombie', 'cow'], size=2, replace=False):
        # print('obj type:', obj_type)
        for ex, ey in self.object_map[obj_type]: 
          if self.avoid_agent:
            ax, ay = ag_pos
            a_type = np.random.choice(['U', 'D', 'L', 'R'],
                                      p=move_away(self.check_bound, self.fire_range, ex, ey, ax, ay))
          else:
            a_type = np.random.choice(['U', 'D', 'L', 'R'])
          try_pos = ex - int(a_type == 'U') + int(a_type == 'D'), ey - int(a_type=='L') + int(a_type=='R') 
          if self.check_bound(try_pos):
            obj_type2 = self.idx2obj_map[self.state[try_pos]]
            if obj_type2 == 'agent':
              # reward -= 1
              reward += 1 if obj_type2 == 'cow' else -1
              # done = True
              self.remove((ex, ey))
            elif obj_type2 == 'none':
              self.move((ex, ey), try_pos)
    # Delayed Done.
    # done = done or self.next_done
    cur_done = done
    done = self.next_done
    self.next_done = cur_done or self.next_done
    clear = ((len(self.object_map['cow']) + len(self.object_map['zombie'])) == 0)
    if clear:
      reward += self.clear_reward
    self.cur_len += 1
    if self.cur_len >= self.max_len or clear:
      self.next_done = True
    if self.fix_len:
      done = self.cur_len >= self.max_len
    ag_pos = self.object_map['agent']
    assert len(ag_pos) == 1
    ag_pos = ag_pos[0]
    if self.abs_rew:
      reward = abs(reward)
    return self.render(), reward, done, dict(ag_pos=np.asarray(ag_pos),
              num_zombie=len(self.object_map['zombie']),
              state=self.state.copy(),
              metric_clear=int(clear),
              obj_id = self.obj_id.copy(),
              num_cow=len(self.object_map['cow']))

import copy

class HunterObjEnv(gym.Env):
  def __init__(self, cfg, version, env_version=None, ret_img=False):
    super().__init__()
    if env_version is None:
      env_version = version
    self.env = Hunter(**cfg.env_kwargs, version=env_version)
    self.cfg = cfg
    self.version = version
    self.ret_img = ret_img
    C = cfg.num_cat
    if version == 0:
      self.id_map  = np.arange(C, dtype=np.int)
      self.label  = np.zeros(C, dtype=np.int)
    elif version == 1:
      self.id_map = np.arange(C, dtype=np.int)
      self.label = np.zeros(C, dtype=np.int)
      idx = 6
      unknowns = ['zombie', 'agent', 'cow']#, 'wall']
      for k in unknowns:
        v = self.env.obj2idx_map[k]
        if cfg.num_cat == 12:
          idx = v + cfg.cat_to_expl
        self.id_map[v] = idx
        self.label[idx] = v
        idx += 1
    elif version == 2:
      self.id_map = np.arange(C, dtype=np.int)
      self.label = np.zeros(C, dtype=np.int)
      unknowns = ['zombie', 'agent', 'cow']
      # avoiding label leaking
      inds = [1,2,0]
      for i, k in enumerate(unknowns):
        v = self.env.obj2idx_map[k]
        idx = self.env.obj2idx_map[unknowns[inds[i]]]
        self.id_map[v] = idx
        self.label[idx + cfg.cat_to_expl] = v
    elif version == 3:
      self.id_map = np.arange(C, dtype=np.int) +  cfg.cat_to_expl
      self.id_map[0] = 0
      self.label = np.zeros(C, dtype=np.int)
      self.label[cfg.cat_to_expl:] = np.arange(cfg.cat_to_expl, dtype=np.int)
      self.label[-1] = 0
      
    self.action_space = self.env.action_space
  
    obs = self.reset()
    self.observation_space = gym.spaces.Dict(
        {k:gym.spaces.Box(low=0, high=10000, shape=v.shape) for k,v in obs.items()}
    )
  def reset(self):
    self.prev_rew = 0
    self.prev_act = 0
    self.estep = 0
    if self.ret_img:
      self.prev_info = dict(view=None, image=None)
    else:
      self.prev_info = dict(view=None)
    self.env.reset()
    return self._get_obs()
  def step(self, a):
    prev_obs = self._get_obs()
    self.prev_info = {k: prev_obs[k] for k in self.prev_info}
    o, r, d, i = self.env.step(a)
    self.prev_rew, self.prev_act = r, a
    self.estep += 1
    return self._get_obs(), r, d, i
  def render(self):
    return self.env.render()
  def _get_obs(self):
    state = self.env.state.copy().astype(np.int)
    state = self.id_map[state]
    view = np.eye(self.cfg.num_cat)[state]
    H, W = state.shape
    img = self.env.render(force_rgb=True)
    unit = img.shape[0] // H
    patches = img.reshape((H,unit,W,unit,3)).transpose(0,2,1,3,4).astype(np.float32) / 255.
    new_o = {'view': view, 'patches': patches, 'image': img,
              'prev_rew': np.asarray([self.prev_rew]).flatten(),
              'label': self.label.copy(),
              'prev_act': np.eye(self.action_space.n)[self.prev_act]}
    prev_info = {k:(self.prev_info[k] if self.prev_info[k] is not None else new_o[k].copy()) for k in self.prev_info}
    new_o.update(**{f'prev_{k}':v for k,v in prev_info.items()})
    if not self.ret_img:
      del new_o['image']
      del new_o['patches']
    return new_o

import functools
def run_gui():
  cfg = oc.DictConfig(dict(env_kwargs=dict(spawn_args='Z2C2', symbolic=False, avoid_agent=True), num_cat=9))
  env = HunterObjEnv(cfg, version=0)
  env.render = functools.partial(env.render, mode='tty_symbolic')
  o = env.reset()
  keymap = {k:v for k, v in zip([' ',
                                'w','s','a','d',
                                'i','k', 'j', 'l',
                                ], np.arange(9))}
  running = True
  print(o['view'].argmax(-1))
  while running:
    event = input('Enter an action:')
    if event in keymap.keys():
      a = keymap[event]
      o, r, d, i = env.step(a)
      print(o['view'].argmax(-1))
      print(r, d)
      if d:
        print('RESET')
        o = env.reset()
        print(o['view'])

if __name__ == '__main__':
  run_gui()

