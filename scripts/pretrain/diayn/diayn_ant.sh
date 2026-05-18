# Configs
job_name=diayn_ant

seed=${1:-0}

discrete=${2:-0}

dim_option=${3:-2}

pref_task="all"
pref_coef=1


# Run command
python3 -u -m run.train --run_group $job_name \
                        --env ant \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 8 \
                        --normalizer_type preset \
                        --eval_plot_axis -50 50 -50 50 \
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
#                         --n_parallel 8 \
#                         --normalizer_type preset \
#                         --eval_plot_axis -50 50 -50 50 \
#                         --trans_optimization_epochs 50 \
#                         --n_epochs_per_log 100 \
#                         --n_epochs_per_eval 1000 \
#                         --n_epochs_per_save 10000 \
#                         --sac_max_buffer_size 1000000 \
#                         --algo metra \
#                         --inner 0 \                           # no inner in diayn
#                         --unit_length 0 \                     # no length restriction in diayn
#                         --dual_reg 0 \                        # no dual update in diayn
#                         --discrete 1 \                        # skill is onehot
#                         --dim_option 50 \
#                         --diayn_include_baseline 1 \          # ? not in metra
#                         --eval_goal_metrics 1 \
#                         --goal_range 50 \
#                         --turn_off_dones 1 \
#                         --sac_lr_a -1 \
#                         --alpha 0.1