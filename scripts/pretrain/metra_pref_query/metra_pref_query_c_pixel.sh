env=$1  
pref_task=$2
pref_coef=$3
seed=$4
device=$5  

dim_option=$6 
query_segmentlen=$7 
query_warmup=$8 
query_freq=$9 
query_limit=${10} 
query_batchsize=${11} 

query_method=${12} 
weight_smooth_decay_speed=${13}


# Configs
job_name=metra_pref_query_${env}_${pref_task}_${query_method}_f${query_freq}_b${query_batchsize}

# Run command
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
                        --sac_max_buffer_size 300000 \
                        --eval_plot_axis -15 15 -15 15 \
                        --algo metra_pref_query \
                        --trans_optimization_epochs 200 \
                        --n_epochs_per_log 25 \
                        --n_epochs_per_eval 125 \
                        --n_epochs_per_save 1000 \
                        --n_epochs_per_pt_save 1000 \
                        --discrete 0 \
                        --dim_option $dim_option \
                        --encoder 1 \
                        --sample_cpu 0 \
                        --eval_goal_metrics 1 \
                        --goal_range 10 \
                        --turn_off_dones 1 \
                        --pref_coef $pref_coef \
                        --pref_task $pref_task \
                        --pb_capacity 500 \
                        --labeled_state_capacity 2010 \
                        --query_warmup $query_warmup \
                        --query_freq $query_freq \
                        --query_limit $query_limit \
                        --query_batchsize $query_batchsize \
                        --query_segmentlen $query_segmentlen \
                        --discriminator_batchsize 128 \
                        --n_sample_times_for_distance -1 \
                        --use_phi_cache 1 \
                        --query_method $query_method \
                        --weight_smooth_decay_speed $weight_smooth_decay_speed 