# COLLIE: Guiding Skill Discovery in Semantically Coherent Latent Space

This is the official implementation of COLLIE.


## Requirements

Set up the conda environment:

```bash
conda create --name collie python=3.8
conda activate collie
```

Install basic packages:

```bash
pip install -r requirements.txt --no-deps
```

Install customized packages (please run them in order and ignore any warnings about incompatible versions):

```bash
pip install -e envs/safety-gym
pip install -e garaged
pip install --upgrade joblib
pip install patchelf
```

To reduce the size of this repo, some experiment environments have been omitted.
You will need to use the following command to retrieve these files from the [CSF repository](https://github.com/Princeton-RL/contrastive-successor-features)

```bash
cd ..
git clone git@github.com:Princeton-RL/contrastive-successor-features.git
cp contrastive-successor-features/lexa/d4rl/ -r collie/lexa/
cp contrastive-successor-features/lexa/metaworld/ -r collie/lexa/
```

Experiments require MuJoCo. You can follow the instructions in the [mujoco-py](https://github.com/openai/mujoco-py) to install.



## Run experiments

All training scripts are stored in `scripts/`. Below are some examples of the scripts. 

Train for Ant North task.
```bash
bash scripts/pretrain/metra_pref_query/metra_pref_query_main.sh ant n 1 0 0 2 20 2000 500 100 10 2000 5 0
# bash scripts/pretrain/metra_pref_query/metra_pref_query_main.sh env pref_task pref_coef \
#      seed device dim_option query_segmentlen query_warmup query_freq query_limit \
#      query_batchsize query_method weight_smooth_decay_speed discrete
```

Train Oracle (GSD with oracle guidance signal) for HalfCheetah Not-Flip task.
```bash
bash scripts/pretrain/metra_pref/metra_pref_discrete.sh half_cheetah not_flip 1 0 0
# bash scripts/pretrain/metra_pref_query/metra_pref_query_main.sh env pref_task pref_coef seed device 
```

For zero-shot evaluation, please see `zero_shot_eval.py`.


## Acknowledgement

This code repo based on [CSF repo](https://github.com/Princeton-RL/contrastive-successor-features), and benefits from the following repos. Thanks for their wonderful work.
* Safety Gym: https://github.com/openai/safety-gym
