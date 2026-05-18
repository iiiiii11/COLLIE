skill_algo=$1
seed=$2 

env=half_cheetah
pref_task=not_flip
pref_coef=1

# NEEDS TO BE REPLACED WITH REAL PATH
cp_path=low_policy/${env}_${pref_task}/option_policy_${skill_algo}_sd${seed}.pt

cp_unit_length=1
policy_type=categorical 

job_name=hier_${env}_${skill_algo}_${pref_task}_${pref_coef}


# Run command
python3 -u -m run.train --run_group $job_name \
                        --env half_cheetah_goal_notflip \
                        --max_path_length 8 \
                        --discrete 1 \
                        --dim_option 16 \
                        --num_random_trajectories 48 \
                        --seed $seed \
                        --normalizer_type preset \
                        --use_gpu 1 \
                        --traj_batch_size 64 \
                        --n_parallel 8 \
                        --algo ppo \
                        --n_epochs_per_eval 100 \
                        --n_thread 1 \
                        --model_master_dim 1024 \
                        --n_epochs_per_log 25 \
                        --eval_record_video 1 \
                        --n_epochs 10010 \
                        --sac_discount 0.99 \
                        --eval_plot_axis -50 50 -50 50 \
                        --trans_optimization_epochs 50 \
                        --sac_max_buffer_size 1000000 \
                        --common_lr 0.0001 \
                        --trans_minibatch_size 256 \
                        --goal_range 100 \
                        --cp_multi_step 25 \
                        --downstream_reward_type esparse \
                        --downstream_num_goal_steps 50 \
                        --cp_path $cp_path \
                        --cp_path_idx 0 \
                        --cp_unit_length $cp_unit_length \
                        --policy_type $policy_type \
                        --alpha 0.01 \
                        --pref_coef $pref_coef \
                        --pref_task $pref_task