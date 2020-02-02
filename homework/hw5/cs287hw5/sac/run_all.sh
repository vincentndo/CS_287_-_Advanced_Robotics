#! /bin/bash

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf_2qf -e 3 --two_qf

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam -e 3 --reparameterize

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam_2qf -e 3 --reparameterize --two_qf



python train_mujoco.py --env_name Ant-v2 --exp_name reinf -e 3

python train_mujoco.py --env_name Ant-v2 --exp_name reinf_2qf -e 3 --two_qf

python train_mujoco.py --env_name Ant-v2 --exp_name reparam -e 3 --reparameterize

python train_mujoco.py --env_name Ant-v2 --exp_name reparam_2qf -e 3 --reparameterize --two_qf



python train_mujoco.py --env_name Hopper-v2 --exp_name reinf -e 3

python train_mujoco.py --env_name Hopper-v2 --exp_name reinf_2qf -e 3 --two_qf

python train_mujoco.py --env_name Hopper-v2 --exp_name reparam -e 3 --reparameterize

python train_mujoco.py --env_name Hopper-v2 --exp_name reparam_2qf -e 3 --reparameterize --two_qf






python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf_seed2 -e 3 --seed 2

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf_2qf_seed2 -e 3 --two_qf --seed 2

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam_seed2 -e 3 --reparameterize --seed 2

python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reparam_2qf_seed2 -e 3 --reparameterize --two_qf --seed 2



python train_mujoco.py --env_name Ant-v2 --exp_name reinf_seed2 -e 3 --seed 2

python train_mujoco.py --env_name Ant-v2 --exp_name reinf_2qf_seed2 -e 3 --two_qf --seed 2

python train_mujoco.py --env_name Ant-v2 --exp_name reparam_seed2 -e 3 --reparameterize --seed 2

python train_mujoco.py --env_name Ant-v2 --exp_name reparam_2qf_seed2 -e 3 --reparameterize --two_qf --seed 2



python train_mujoco.py --env_name Hopper-v2 --exp_name reinf_seed2 -e 3 --seed 2

python train_mujoco.py --env_name Hopper-v2 --exp_name reinf_2qf_seed2 -e 3 --two_qf --seed 2

python train_mujoco.py --env_name Hopper-v2 --exp_name reparam_seed2 -e 3 --reparameterize --seed 2

python train_mujoco.py --env_name Hopper-v2 --exp_name reparam_2qf_seed2 -e 3 --reparameterize --two_qf --seed 2
