# Configs
job_name=diayn_humanoid


seed=${1:-0}

discrete=${2:-0}

dim_option=${3:-2}

device=${4:-0}

pref_task="all"
pref_coef=1


# Run command
CUDA_VISIBLE_DEVICES=$device python3 -u -m run.train \
                        --run_group $job_name \
                        --env dmc_humanoid \
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
                        --algo metra \
                        --trans_optimization_epochs 200 \
                        --n_epochs_per_log 25 \
                        --n_epochs_per_eval 250 \
                        --n_epochs_per_save 1000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete $discrete \
                        --dim_option $dim_option \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --eval_goal_metrics 1 \
                        --goal_range 10 \
                        --turn_off_dones 1 \
                        --inner 0 \
                        --unit_length 0 \
                        --dual_reg 0 \
                        --diayn_include_baseline 1 \
                        --eval_record_video 0 \
                        --sac_lr_a -1 \
                        --alpha 0.1 \
                        --pref_task $pref_task \
                        --pref_coef $pref_coef