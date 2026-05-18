job_name=lsd_quadruped


seed=${1:-0}

discrete=${2:-0}

dim_option=${3:-2}

device=${4:-0}

pref_task="all"
pref_coef=1


# Run command
CUDA_VISIBLE_DEVICES=$device python3 -u -m run.train \
                        --run_group $job_name \
                        --env dmc_quadruped \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --n_epochs 5010 \
                        --normalizer_type off \
                        --video_skip_frames 2 \
                        --frame_stack 3 \
                        --sac_max_buffer_size 300000 \
                        --eval_plot_axis -15 15 -15 15 \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 100 \
                        --n_epochs_per_eval 100 \
                        --n_epochs_per_save 2000 \
                        --algo metra \
                        --dual_reg 0 \
                        --spectral_normalization 1 \
                        --discrete $discrete \
                        --dim_option $dim_option \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --goal_range 10 \
                        --eval_goal_metrics 1 \
                        --turn_off_dones 1 \
                        --pref_task $pref_task \
                        --pref_coef $pref_coef


# difference from metra:
# --dual_reg 0 \
# --spectral_normalization 1 \



