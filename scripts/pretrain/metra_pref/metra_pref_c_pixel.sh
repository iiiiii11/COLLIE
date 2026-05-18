env=$1
dim_option=$2
seed=$3
pref_coef=$4
pref_task=$5
device=$6

job_name=metra_c_${dim_option}_${env}_${pref_task}_${pref_coef}

CUDA_VISIBLE_DEVICES=$device python3 -u -m run.train \
                        --run_group $job_name \
                        --env $env \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --n_epochs 5010 \
                        --normalizer_type off \
                        --video_skip_frames 2 \
                        --frame_stack 3 \
                        --eval_plot_axis -15 15 -15 15 \
                        --trans_optimization_epochs 200 \
                        --n_epochs_per_log 125 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 2000 \
                        --sac_max_buffer_size 300000 \
                        --algo metra_pref \
                        --discrete 0 \
                        --dim_option $dim_option \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --eval_goal_metrics 1 \
                        --goal_range 10 \
                        --turn_off_dones 1 \
                        --pref_coef $pref_coef \
                        --pref_task $pref_task

