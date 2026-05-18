job_name=lsd_half_cheetah


seed=${1:-0}

discrete=${2:-0}

dim_option=${3:-2}

device=${4:-0}

pref_task="all"
pref_coef=1


# Run command
CUDA_VISIBLE_DEVICES=$device python3 -u -m run.train \
                        --run_group $job_name \
                        --env half_cheetah \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --n_epochs 10010 \
                        --normalizer_type preset \
                        --eval_plot_axis -50 50 -50 50 \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 100 \
                        --n_epochs_per_eval 100 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra \
                        --dual_reg 0 \
                        --spectral_normalization 1 \
                        --discrete $discrete \
                        --dim_option $dim_option \
                        --goal_range 50 \
                        --turn_off_dones 1 \
                        --pref_task $pref_task \
                        --pref_coef $pref_coef



# from metra's readme
# python tests/main.py 
#     --run_group Debug \
#     --env ant \
#     --max_path_length 200 \
#     --seed 0 \
#     --traj_batch_size 8 \
#     --n_parallel 1 \
#     --normalizer_type preset \
#     --eval_plot_axis -50 50 -50 50 \
#     --trans_optimization_epochs 50 \
#     --n_epochs_per_log 100 \
#     --n_epochs_per_eval 1000 \
#     --n_epochs_per_save 10000 \
#     --sac_max_buffer_size 1000000 \
#     --algo metra \
#     --dual_reg 0 \
#     --spectral_normalization 1 \
#     --discrete 0 \
#     --dim_option 2

