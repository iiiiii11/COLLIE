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

discrete=${14}

job_name=metra_pref_query_${env}_${pref_task}_${pref_coef}


if [[ "$env" == *"CSafe"* ]]; then
    eval_plot_axis="-10 10 -10 10"
else
    eval_plot_axis="-50 50 -50 50"
fi

CUDA_VISIBLE_DEVICES=$device python3 -u -m run.train \
                        --run_group $job_name \
                        --env $env \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --n_epochs 10010 \
                        --normalizer_type preset \
                        --eval_plot_axis $eval_plot_axis \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 500 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra_pref_query \
                        --discrete $discrete \
                        --dim_option $dim_option \
                        --eval_goal_metrics 1 \
                        --goal_range 50 \
                        --turn_off_dones 1 \
                        --pref_coef $pref_coef \
                        --pref_task $pref_task \
                        --pb_capacity 1000000 \
                        --labeled_state_capacity 10000 \
                        --query_warmup $query_warmup \
                        --query_freq $query_freq \
                        --query_limit $query_limit \
                        --query_batchsize $query_batchsize \
                        --query_segmentlen $query_segmentlen \
                        --discriminator_batchsize 128 \
                        --score_model_name sd \
                        --n_sample_times_for_distance -1 \
                        --use_phi_cache 1 \
                        --query_method $query_method \
                        --weight_smooth_decay_speed $weight_smooth_decay_speed