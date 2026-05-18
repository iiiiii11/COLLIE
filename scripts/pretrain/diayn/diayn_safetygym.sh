# Configs

seed=${1:-0}

discrete=${2:-0}

dim_option=${3:-2}

job_name=diayn_$4


pref_task="all"
pref_coef=1

eval_plot_axis="-10 10 -10 10"

# Run command
python3 -u -m run.train --run_group $job_name \
                        --env $4 \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --normalizer_type off \
                        --eval_plot_axis $eval_plot_axis \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 500 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra \
                        --inner 0 \
                        --unit_length 0 \
                        --dual_reg 0 \
                        --discrete $discrete \
                        --dim_option $dim_option \
                        --diayn_include_baseline 1 \
                        --eval_goal_metrics 1 \
                        --goal_range 50 \
                        --turn_off_dones 1 \
                        --sac_lr_a -1 \
                        --alpha 0.1 \
                        --pref_task $pref_task \
                        --pref_coef $pref_coef


# python3 -u -m run.train --run_group $job_name \
#                         --env ant \
#                         --max_path_length 200 \
#                         --seed $seed \
#                         --traj_batch_size 8 \
#                         --n_parallel 4 \
#                         --normalizer_type preset \
#                         --eval_plot_axis -50 50 -50 50 \
#                         --trans_optimization_epochs 50 \
#                         --n_epochs_per_log 500 \
#                         --n_epochs_per_eval 500 \
#                         --n_epochs_per_save 10000 \
#                         --sac_max_buffer_size 1000000 \
#                         --algo metra \
#                         --discrete 0 \
#                         --dim_option 2 \
#                         --eval_goal_metrics 1 \
#                         --goal_range 50 \
#                         --turn_off_dones 1