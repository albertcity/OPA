from tianshou.data import Batch, ReplayBuffer, ReplayBufferManager, VectorReplayBuffer
import copy
import os
import numpy as np
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch, to_torch_as
def extend(b, size):
    len_b = len(b)
    new_b = Batch.cat([b] + [b[-1:]] * (size - len(b)))
    valid_mask = np.asarray([1] * len_b + [0] * (size-len_b))
    return new_b, valid_mask
def extend_with_mask(b, size):
    len_b = len(b)
    new_b = Batch.cat([b] + [b[-1:]] * (size - len(b)))
    valid_mask = np.asarray([1] * len_b + [0] * (size-len_b))
    new_b.policy.valid_mask = valid_mask
    return new_b
  
class TrajectoryReplayBuffer(ReplayBuffer):
  def __init__(self, size, maxlen):
    self.maxlen = maxlen
    self._traj = ReplayBuffer(self.maxlen)
    super().__init__(size)
  def reset(self, keep_statistics=False):
    self._traj = ReplayBuffer(self.maxlen)
    super().reset(keep_statistics)
  def add(self, batch, buffer_ids = None):
    batch = to_numpy(batch)
    assert len(self._traj) <= self.maxlen
    stacked_batch = buffer_ids is not None
    if stacked_batch:
      assert len(batch) == 1
      batch = batch[0]
    res = self._traj.add(batch)
    done = bool(batch.done)
    if done or len(self._traj) >= self.maxlen:
      traj, _ = self._traj.sample(0)
      if 'hidden_state' in traj.policy:
        del traj.policy.hidden_state
      # if done:
      self._traj.reset()
      extended_traj, valid_mask = extend(traj, self.maxlen)
      extended_traj.policy.real_done = extended_traj.done
      extended_traj.policy.real_rew  = extended_traj.rew
      extended_traj.policy.valid_mask  = valid_mask
      extended_traj.rew = sum(extended_traj.rew)
      extended_traj.done = done
      super().add(extended_traj)
    return res
 
class VecTrajectoryReplayBuffer(ReplayBuffer):
  def __init__(self, nenvs, size, maxlen):
    self.maxlen = maxlen
    self._trajs = [ReplayBuffer(self.maxlen) for _ in range(nenvs)]
    self.nenvs = nenvs
    super().__init__(size)
  def reset(self, keep_statistics=False):
    self._trajs = [ReplayBuffer(self.maxlen) for _ in range(self.nenvs)]
    super().reset(keep_statistics)
  def add(self, batch, buffer_ids = None):
    batch = to_numpy(batch)
    for traj_buf in self._trajs:
      assert len(traj_buf) <= self.maxlen
    if buffer_ids is None:
      buffer_ids = np.arange(len(self._trajs))
    all_res = []
    for i, b in zip(buffer_ids, batch):
      traj_buf = self._trajs[i]
      res = traj_buf.add(b)
      all_res.append(res)
      done = bool(b.done)
      if done or len(traj_buf) >= self.maxlen:
        traj, _ = traj_buf.sample(0)
        traj_buf.reset(keep_statistics=True)
        extended_traj, valid_mask = extend(traj, self.maxlen)
        extended_traj.policy.real_done = extended_traj.done
        extended_traj.policy.real_rew  = extended_traj.rew
        extended_traj.policy.valid_mask  = valid_mask
        extended_traj.rew = sum(extended_traj.rew)
        extended_traj.done = done
        super().add(extended_traj)
    all_res = [np.asarray([res[k] for res in all_res]) for k in range(len(all_res[0]))]
    return all_res

class VecHistReplayBuffer(VectorReplayBuffer):
  def __init__(self, nenvs, size, his_size, his_dir='', save_img=False, maxlen=32):
    super().__init__(size, nenvs)
    # self.his = VectorReplayBuffer(his_size, nenvs)
    his_size = his_size // maxlen
    self.his = VecTrajectoryReplayBuffer(nenvs, his_size, maxlen)
    self.his_id = 0
    self.his_dir = his_dir
    self.save_img = save_img
    if self.his_dir:
      self.his_dir = os.path.join(self.his_dir, 'history')
      os.makedirs(self.his_dir, exist_ok=True)
    self.his_size = his_size
  def _save_his(self):
    if self.his_dir:
      his_dir = os.path.join(self.his_dir, f'his{self.his_id}_size{self.his_size}.hd5') 
      self.his.save_hdf5(his_dir)
  def filter_obs(self, batch):
    b = to_numpy(Batch(batch))
    new_b = Batch(dict(view=np.argmax(b.view, -1),
            **{k:b[k] for k in ['label', 'step', 'inventory', 'status', 'prev_rew', 'prev_act'] if k in b},
    ))
    for k in ['prev_view', 'prev_inventory', 'prev_status']:
      if k in b:
        new_b[k] = np.argmax(b[k], -1) if k == 'prev_view' else b[k]
    if self.save_img:
      new_b.image = b.image.astype(np.uint8)
      if 'all_image' in b:
        new_b.all_image = b.all_image.astype(np.uint8)
    return new_b
  def filter_data(self, batch):
    new_b = Batch()
    new_b.obs = self.filter_obs(batch.obs)
    if 'obs_next' in batch:
      new_b.obs_next = self.filter_obs(batch.obs_next)
    for k in ['rew', 'act', 'done']:
      new_b[k] = batch[k] 
    return new_b
  def add(self, batch, buffer_ids = None):
    self.his.add(self.filter_data(batch), buffer_ids)
    if len(self.his) >= self.his_size:
      self._save_his()
      self.his_id += 1
      self.his.reset()
    return super().add(batch, buffer_ids)

