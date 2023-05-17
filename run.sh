python ppo_main.py env_kwargs.spawn_args=Z3C3 logdir=log/policy/Z3C3 seed=12345 save_hist=1 use_vec_traj_to_save=1 epoch=22
python train_disc.py data_dir=log/policy/Z3C3 logdir=log/disc/Z3C3  epoch=1
python ppo_expl_main.py env_kwargs.spawn_args=Z3C3 load_task_pol_dir=log/policy/Z3C3/s12345/all_model.pth load_disc_dir=log/disc/Z3C3/disc.pt logdir=log/expl/Z3C3 epoch=2
