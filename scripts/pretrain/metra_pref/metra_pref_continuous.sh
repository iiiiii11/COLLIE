
env=$1
pref_task=$2
pref_coef=$3
seed=$4
device=$5  

# Configs
job_name=metra_pref_${env}_${pref_task}_${pref_coef}

if [[ "$env" == *"CSafe"* ]]; then
    eval_plot_axis="-10 10 -10 10"
else
    eval_plot_axis="-50 50 -50 50"
fi

# Run command
CUDA_VISIBLE_DEVICES=$device python3 -u -m run.train \
                        --run_group $job_name \
                        --env $env \
                        --max_path_length 200 \
                        --seed $seed \
                        --traj_batch_size 8 \
                        --n_parallel 4 \
                        --normalizer_type preset \
                        --eval_plot_axis $eval_plot_axis \
                        --trans_optimization_epochs 50 \
                        --n_epochs_per_log 500 \
                        --n_epochs_per_eval 500 \
                        --n_epochs_per_save 10000 \
                        --sac_max_buffer_size 1000000 \
                        --algo metra_pref \
                        --discrete 0 \
                        --dim_option 2 \
                        --eval_goal_metrics 1 \
                        --goal_range 50 \
                        --turn_off_dones 1 \
                        --pref_coef $pref_coef \
                        --pref_task $pref_task