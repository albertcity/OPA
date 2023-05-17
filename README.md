This is a public implementation of the paper "Online Prototype Alignment for Few-shot Policy Transfer" (OPA).

The main implementation of PPO is based on the `tianshou` library, which is also included in this repository. However, the logger is borrowed form the  `stable-baselines3` and is not included. You should install  `stable-baselines3` or disable the logger first.

To reproduce the results of OPA on Hunter-Z3C3, you should:
- train a policy $\pi_{task}$ to solve the task in the source domain and save the history trajectories at the same time:
```
python ppo_main.py env_kwargs.spawn_args=Z3C3 logdir=log/policy/Z3C3 seed=12345 save_hist=1 use_vec_traj_to_save=1
```

- train the discriminator (i.e. the inference model $q_\theta$) using the saved trajectories:

```
python train_disc.py data_dir=log/policy/Z3C3 logdir=log/disc/Z3C3
```

- train the exploration policy $\pi_{exp}$ using $q_\theta$

```
python ppo_expl_main.py env_kwargs.spawn_args=Z3C3 load_task_pol_dir=log/policy/Z3C3/s12345/all_model.pth load_disc_dir=log/disc/Z3C3/disc.pt logdir=log/expl/Z3C3
```

For simplicity, we also provide some pretrained models in the `pretrain_model` folder, including:
- `pretrain_models/percept`: the novelty detection model $\Psi_{unseen},f_{ND}$ for Hunter.
- `pretrain_models/policy/Z3C3/s4801973`: $\pi_{task}$ for Hunter-Z3C3
- `pretrain_models/disc/Z3C3`: $q_\theta$ for Hunter-Z3C3

Feel free to raise a issue to communicate with us.
