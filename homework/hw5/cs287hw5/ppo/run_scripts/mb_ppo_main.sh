#!/usr/bin/env bash

## 2A. Model-based PPO with a single model

echo "---- 2A ----"

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name HalfCheetah \
                               --exp_name mbppo_HalfCheetah \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name Hopper \
                               --exp_name mbppo_Hopper \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name Swimmer \
                               --exp_name mbppo_Swimmer \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done


## 2B. 

echo "---- 2B ----"

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name HalfCheetah \
                               --exp_name mbppo_HalfCheetah_l1 \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name Hopper \
                               --exp_name mbppo_Hopper_l1 \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name Swimmer \
                               --exp_name mbppo_Swimmer_l1 \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name HalfCheetah \
                               --exp_name mbppo_HalfCheetah_l2_no_square \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name Hopper \
                               --exp_name mbppo_Hopper_l2_no_square \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name Swimmer \
                               --exp_name mbppo_Swimmer_l2_no_square \
                               --exp_num $i \
                               --use_ppo_obj 1 \
                               --use_clipper 1 \
                               --use_entropy 1 \
                               --use_gae 0 \
                               --ensemble 0 &
done
