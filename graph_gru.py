from encoder import *
import cv2
import functools
# from skillhack_wrapper import CAT_TO_EXPL
import omegaconf as oc
import gym
import torchvision
import dgl
import copy
from stable_baselines3.common import logger as L

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
import copy
def printShape(b):
    b = copy.deepcopy(b)
    traverseBatch(b, {'tensor':lambda x:x.shape, 'ndarray':lambda x:x.shape})
    print(b)
def batch_flatten(batch):
    traverseBatch(batch, {'tensor': lambda x: x.flatten(end_dim=1),
                        'ndarray': lambda x: x.reshape((np.prod(x.shape[:2]), *x.shape[2:]))})
def batch_unflatten(batch, B, T):
    traverseBatch(batch, {'tensor': lambda x: x.reshape(B, T, *x.shape[1:]),
                        'ndarray': lambda x: x.reshape(B, T, *x.shape[1:])})


class RefConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=1):
        super().__init__()
        convs = []
        convs.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
        for l in range(num_layers-1):
            convs.append(nn.ReLU())
            convs.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
        self.convs = nn.Sequential(*convs)
        self.sparse_select = None
        
    def forward(self, o, o2, refs, other_fea=None):
        """
        o&o2: (...,H,W,C), refs: (...,C, -1), masks: (...,C, 1), other_fea: (...,-1) 
        out:  (...,C,-1)
        """
        dims = o.shape[:-3]
        H, W, C = o.shape[-3:]
        fea_o = torch.matmul(o.reshape(*dims, H*W, C), refs).reshape(*dims, H, W, -1)
        fea_o2 = torch.matmul(o2.reshape(*dims, H*W, C), refs).reshape(*dims, H, W, -1)

        if other_fea is not None:
            other_fea = other_fea.reshape(*dims, 1, 1, other_fea.shape[-1])
            other_fea = torch.broadcast_to(other_fea, (*dims, H, W, -1))
            all_fea = torch.cat([fea_o, fea_o2, other_fea], dim=-1)
        else:
            all_fea = torch.cat([fea_o, fea_o2], dim=-1)
        all_fea = all_fea.reshape(np.prod(dims), H, W, -1).permute(0,3,1,2)
        out = self.convs(all_fea) # ..., -1, H, W

        out_fea = out.reshape(*dims, -1, H*W).transpose(-1,-2) # ..., H*W, F
        o = o.reshape(*dims, H*W, C).transpose(-1,-2) # ..., C, H*W
        o2 = o2.reshape(*dims, H*W, C).transpose(-1,-2) # ..., C, H*W
        out = torch.matmul(o, out_fea) / torch.clamp(o.sum(-1, keepdim=True), min=1)
        out = out + torch.matmul(o2, out_fea) / torch.clamp(o2.sum(-1, keepdim=True), min=1)
        return out

class RefGRU(nn.Module):
    def __init__(self, rdim, fdim, mask):
        super().__init__()
        self.rdim, self.fdim = rdim, fdim
        self.ref_net = RefConv(rdim * 2 + fdim, rdim, num_layers=3)
        self.num_layers = 2
        self.gru = nn.GRU(rdim, rdim, self.num_layers, batch_first=True)
        self.mask = torch.as_tensor(mask).float().flatten().unsqueeze(-1)
        self.initial_state = nn.Parameter(torch.randn(rdim))
        C = self.mask.shape[0]
        if True:
            self.initial_all_state = nn.Parameter(torch.randn(C, rdim))
        else:
            self.initial_all_state = torch.eye(C, rdim)
    def none_or_valid(self, x, shape, val):
        if x is None:
            return torch.ones(shape) * val
        else:
            return x
    def reset_state(self, B, C):
        hids = self.none_or_valid(None, (self.num_layers,B*C,self.rdim), 0)
        masks = self.mask
        # print(self.initial_all_state.device, masks.device)
        refs = einops.repeat(self.initial_all_state * (1-masks) + self.initial_state * masks, 'n r -> b n r', b=B)
        return refs, hids
    def pack_hids(self, refs, hids):
        B, C = refs.shape[:2]
        refs = refs.reshape(B,-1)
        hids = einops.rearrange(hids, 'l (b n) r -> b (l n r)', b=B)
        all = torch.cat([refs, hids], dim=-1)
        return all
    def unpack_hids(self, all, C, check_zero=True):
        B = all.shape[0]
        refs = all[..., :C*self.rdim].reshape(B,C,-1)
        if check_zero:
            zero_mask = (refs.abs().amax([-1,-2], keepdim=True) < 0.01).float() # B
            # zero_mask = zero_mask.reshape(B,1,1)
            new_refs, _ = self.reset_state(B, C) # B,C,-1
            refs = new_refs * zero_mask + refs * (1 - zero_mask)
        hids = einops.rearrange(all[...,C*self.rdim:], 'b (l n r) -> l (b n) r', l=self.num_layers, n=C)
        return refs, hids
    """
    out: (B, T, C, -1)
    """
    def forward(self, o, o2, hids=None, other_fea=None):
        B, T = o.shape[:2]
        C = o.shape[-1]
        outs = []
        masks = self.mask
        if hids is None:
            refs, hids = self.reset_state(B, C)
            assert C <= self.rdim
        else:
            refs, hids = self.unpack_hids(hids, C, check_zero=True)
        other_fea = self.none_or_valid(other_fea, (B,T,self.fdim), 0)
        if False:
            refs, _ = self.reset_state(B, C)
            m = self.ref_net(o, o2, refs.unsqueeze(1), other_fea) # B,T,C,-1
            new_refs, hids = self.gru(einops.rearrange(m, 'b t c f -> (b c) t f'), hids)
            refs = einops.rearrange(new_refs, '(b c) t r -> b t c r', b=B, c=C) * masks
            outs = refs
            refs = refs[:,-1]
        else:
            for t in range(T):
                m = self.ref_net(o[:,t], o2[:,t], refs, other_fea[:,t])
                # TODO: gru takes T already.
                new_refs, hids = self.gru(m.reshape(B*C, 1, -1), hids)
                refs = new_refs.reshape(B,C,-1) * masks + refs * (1 - masks)
                outs.append(refs)
            outs = torch.stack(outs, dim=1)
        return outs, self.pack_hids(refs, hids)
class AggNet(nn.Module):
    def __init__(self, out_dim, net_archs):
        super().__init__()
        self.nets = nn.ModuleDict()
        dim = 0
        for k, net_arch in net_archs.items():
            pre_fn = net_arch[0]
            if pre_fn is None:
                pre_fn = lambda x: x
            net_arch = net_arch[1:]
            self.nets[k] = nn.Sequential(AsLayer(lambda x: pre_fn(x)), create_mlp(*net_arch, return_seq=True))
            dim += net_arch[1]
        self.linear = nn.Linear(dim, out_dim)
    def forward(self, kv):
        out = []
        for k, net in self.nets.items():
            net_in = kv[k]
            out.append(net(net_in))
        out = torch.cat(out, dim=-1)
        return self.linear(out)
class StackedConv_8x8(nn.Module):
    def __init__(self, in_dim, out_dim, reduce='linear'):
        super().__init__()
        self.linear = nn.Linear(in_dim, 16)
        self.convs = nn.Sequential(
            nn.Conv2d(16,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
        )
        if reduce == 'linear':
            self.final = nn.Sequential(nn.Flatten(), nn.Linear(2*2*32, out_dim))
        elif reduce == 'max':
            self.final = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, out_dim))
        elif reduce == 'max_nolinear':
            assert out_dim == 32
            self.final = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten())
    def forward(self, x):
        *dims, h, w, c = x.shape
        x = self.linear(x.reshape(np.prod(dims), h, w, c))
        x = self.convs(x.permute(0,3,1,2))
        x = self.final(x)
        return x.reshape(*dims, 1, -1)

from relation_net import RNModule, RNModuleV2
class ObsEmbed(nn.Module):
    def __init__(self, cfg, out_dim=None,view_shape=(8,8),
                 view=True, info=True, view_dim=None, view_type='linear', use_mlp=True):
        super().__init__()
        hdim = cfg.hidden_dim
        out_dim = out_dim or hdim
        self.cfg = cfg
        hdim = self.cfg.hidden_dim
        self.view, self.info = view, info
        all_dim = 0
        view_dim = view_dim or cfg.num_cat
        if view:
            if view_type in ['linear', 'max', 'max_nolinear']:
                self.view_enc = StackedConv_8x8(view_dim, hdim, view_type)
            elif view_type == 'rn':
                self.view_enc = nn.Sequential(nn.Linear(cfg.num_cat, hdim),
                                              RNModule(cfg, (*view_shape,hdim), num_layers=2))
            elif view_type == 'rn_v2':
                self.view_enc = RNModuleV2(cfg, (*view_shape,cfg.num_cat), num_layers=2)
            all_dim += hdim
        else:
            self.view_enc = None
        if self.cfg.get('info_spec', {}) and info:
            self.info_enc = AggNet(hdim, {
                k: (None, v, hdim//2, [hdim]) for k,v in self.cfg.info_spec.items()
            })
            all_dim += hdim
        else:
            self.info_enc = None
        if use_mlp:
            self.mlp = create_mlp(all_dim, out_dim, [hdim], return_seq=True)
        else:
            assert all_dim == out_dim
            self.mlp = nn.Identity()
        self.out_dim = out_dim
        self.to(cfg.device)
    def maybe_cat(self, view, info):
        if view is None:
            out = info
        elif info is None:
            out = view
        else:
            out = torch.cat([view, info], dim=-1)
        return self.mlp(out)
    def forward(self, obs, refs=None, **kwargs):
        view = None
        if self.view_enc is not None:
            view = obs['view']
            if refs is not None:
                *dims, H,W,C = view.shape
                view = torch.matmul(view.reshape(*dims, H*W, C), refs).reshape(*dims, H, W, -1)
            view = self.view_enc(view).amax(-2)
        info = None if self.info_enc is None else self.info_enc(obs)
        out = self.maybe_cat(view, info)
        return out, refs

class RefTrajEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hdim = cfg.hidden_dim
        self.cfg = cfg
        wts = np.ones(cfg.num_cat)
        wts[0] = 0 
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(wts).float())
        if cfg.get('pred_multi_layer', False):
            self.pred = create_mlp(hdim, cfg.num_cat, [hdim], return_seq=True)
        else:
            self.pred = create_mlp(hdim, cfg.num_cat, [], return_seq=True)
        info_spec = dict(prev_rew=1, prev_act=cfg.num_act, **cfg.get('info_spec', {}))
        info_spec.update(**{f'prev_{k}': v for k,v in cfg.get('info_spec', {}).items()})
        self.info_enc = ObsEmbed(oc.DictConfig(dict(hidden_dim=8,
                                        num_cat=cfg.num_cat,
                                        device=cfg.device,
                                        info_spec=info_spec,)), 
                                view=False, info=True, out_dim=16)
        mask = np.zeros(cfg.num_cat)
        mask[cfg.cat_to_expl:] = 1
        # mask[7:10] = 1
        self.ref_gru = RefGRU(hdim, self.info_enc.out_dim, mask)
    def step_forward(self, trajs, mask, hidden_state=None):
        B = trajs.obs.shape[0]
        trajs = to_torch(trajs, dtype=torch.float32, device=self.cfg.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.cfg.device)
        traverseBatch(trajs, {'tensor': lambda x: x.unsqueeze(1)})
        mask = mask.reshape(B, 1)
        targ_logp, loss, hidden_state, other_info = self._compute_context(trajs, mask, hidden_state)
        traverseBatch(trajs, {'tensor': lambda x: x.squeeze(1)})
        for k, v in other_info.items():
            if len(v.shape) >= 2:
                other_info[k] = v.squeeze(1)
        return targ_logp.unsqueeze(1), hidden_state, other_info
    def _compute_logits(self, trajs, mask, hidden_state=None):
        B, T = trajs.obs.shape[:2]
        C = self.cfg.num_cat
        # info_embeds = torch.cat([self.info_enc(trajs.obs)[0], elf.info_enc(trajs.obs_next)[0]], dim=-1)
        info_embeds = self.info_enc(trajs.obs)[0]
        # embeds: B,T,C,-1  hidden_state: B,-1
        embeds, hidden_state = self.ref_gru(trajs.obs.prev_view, trajs.obs.view, hidden_state, info_embeds)
        # B,T,C,C
        logits = self.pred(embeds)
        addition_loss = 0
        return logits, embeds, hidden_state, addition_loss
        
    def _compute_context(self, trajs, mask, hidden_state=None):
        """
        trajs: batch of obs, obs=Batch(view, prev_rew, prev_act, prev_view, ...)
        mask:  (1,1,1,1,1,...,1,0,0,0,0,...,0)
        """
        trajs = to_torch(trajs, dtype=torch.float32, device=self.cfg.device)
        mask = to_torch(mask, dtype=torch.float32, device=self.cfg.device)
        B, T = trajs.obs.shape[:2]
        C = self.cfg.num_cat
        logits, embeds, hidden_state, addi_loss = self._compute_logits(trajs, mask, hidden_state)
        probs = F.softmax(logits, dim=-1)
        label = trajs.obs.label.reshape(B,T,-1)
        label = (label * mask.reshape(B,T,1)).long()

        loss = self.loss_fn(logits.reshape(B*T*C, -1), label.flatten())
        loss = loss + addi_loss
        targ_logp = torch.distributions.Categorical(logits=logits).log_prob(label) * (label > 0)
        max_targ_p = torch.where(label > 0.5, targ_logp, torch.ones_like(targ_logp) * -100.0).amax(1)[label.amax(1) > 0.5].exp().mean().item()
        mean_targ_p = targ_logp[label>0.5].exp().mean().item()
        prob_refs = torch.eye(C) * (1-self.ref_gru.mask) + probs * self.ref_gru.mask
        other_info = {'targ_p': targ_logp.exp() * (label > 0), 
                    'hidden_state': hidden_state,
                    'prob_refs': prob_refs,
                    'targ_logp': targ_logp * (label > 0.5) + (label < 0.5) * -1000.0, 
                    'probs': probs, 'logits': logits,
                    'max_targ_p': max_targ_p, 'mean_targ_p': mean_targ_p,
                    'embeds': embeds,
                    }
        return targ_logp, loss, hidden_state, other_info
    def label_rewards(self, trajs, masks, hidden_state=None):
        B, T, C = trajs.obs.label.shape
        label = (trajs.obs.label * masks.reshape(B,T,1)).long()
        if hidden_state is not None:
            rst_hid = False
            refs, _ = self.ref_gru.unpack_hids(hidden_state, self.cfg.num_cat, check_zero=True)
        else:
            rst_hid = True
            refs, _ = self.ref_gru.reset_state(len(trajs), self.cfg.num_cat)
        logits =  self.pred(refs)
        if rst_hid:
            addi_loss = self.loss_fn(logits.reshape(B*C, -1), label[:,0].flatten())
        initial_targ_logp = torch.distributions.Categorical(logits=logits).log_prob(label[:,0])
        initial_targ_logp = (initial_targ_logp * (label[:,0] > 0.5)).sum(-1)
        targ_logp, loss, hidden_state, other_infos = self._compute_context(trajs, masks, hidden_state)
        targ_logp = targ_logp.sum(-1)
        rews = targ_logp - torch.cat([initial_targ_logp[:,None], targ_logp[:,:-1]], dim=1)
        if self.cfg.get('clip_rews', False):
            rews = torch.clamp(rews, min=0) * (targ_logp.exp() >= 0.5)
        if rst_hid:
            loss = loss + addi_loss
        return to_numpy(rews) * to_numpy(masks), loss, hidden_state, other_infos

    @torch.no_grad()
    def get_labels_for_unknown(self, trajs, masks, hidden_state=None):
        *args, other_infos = self._compute_context(trajs, masks, hidden_state)
        probs, patches = other_infos['probs'], trajs.obs.patches 
        B, T, C, _ = probs.shape
        prediction = (probs.amax([0,1]) * (1-self.ref_gru.mask)).argmax(-1)
        patches, labels = [], []
        for c in range(self.cfg.cat_to_expl, C):
            patch_c = patches[trajs.obs.view[...,c] > 0.5]
            if len(patch_c) > 0:
                patches.append(patch_c)
                cat = prediction[c].item()
                labels.append(torch.ones(len(patch_c)) * cat)
        if patches:
            patches = torch.cat(patches, dim=0)
            labels = torch.cat(labels, dim=0).long()
            self.record_patches(patches, labels, key='labels_for_unknown')
        else:
            patches = labels = None
        return patches, labels
    def record_patches(self, patches, label, key='labels_for_unknown'):
        patches = patches.reshape(-1, *self.cfg.patch_shape)
        ind = np.random.choice(np.arange(len(patches)), size=16, replace=True)
        patches = to_numpy(patches[ind] * 255).astype(np.uint8)
        label = to_numpy(label.flatten().long())[ind]
        new_p = []
        for i, p in enumerate(patches):
            p = cv2.resize(p, (64, 64))
            p = cv2.putText(p, str(int(label[i])), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            new_p.append(p)
        obs = torch.as_tensor(np.stack(new_p, axis=0)).permute(0, 3, 1, 2)
        obs = torchvision.utils.make_grid(obs, nrow=4, padding=2, pad_value=0.2)
        L.record_tb(key, L.Image(obs, 'CHW'))

