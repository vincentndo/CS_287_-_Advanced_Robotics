#!/usr/bin/env bash

## 1A. Policy Gradient

echo "---- 1A ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name pg \
                            --exp_num $i \
                            --use_baseline 0 \
                            --use_ppo_obj 0 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name pg \
                            --exp_num $i \
                            --use_baseline 0 \
                            --use_ppo_obj 0 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name pg \
                            --exp_num $i \
                            --use_baseline 0 \
                            --use_ppo_obj 0 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait


## 1B. Baselines

echo "---- 1B ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name pg+baseline \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 0 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name pg+baseline \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 0 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name pg+baseline \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 0 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait


## 1C. Likelihood Ratio

echo "---- 1C ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name ppo\(unclipped\) \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name ppo\(unclipped\) \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name ppo\(unclipped\) \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait


## 1D. Clipping

echo "---- 1D ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name ppo \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name ppo \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name ppo \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 0 \
                            --use_gae 0 &
done

wait


## 1E. Entropy bonus

echo "---- 1E ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name ppo+entropy \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name ppo+entropy \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 0 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name ppo+entropy \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 0 &
done

wait


## 1F. (Extra credits) Generalized Advantage Estimator (GAE)

echo "---- 1F ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name ppo+entropy+gae \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 1 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name ppo+entropy+gae \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 1 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name ppo+entropy+gae \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 1 &
done


## 1G. (Extra credits) Hyperparameter Tuning

echo "---- 1G ----"

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name ppo+entropy+gae+tuning_hidden192 \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 1 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name ppo+entropy+gae+tuning_hidden192 \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 1 &
done

wait

for i in `seq 0 2`;
do
    echo $i
    python ppo_run_sweep.py --env_name Swimmer \
                            --exp_name ppo+entropy+tuning_hidden192 \
                            --exp_num $i \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 1 \
                            --use_entropy 1 \
                            --use_gae 0 &
done

wait
