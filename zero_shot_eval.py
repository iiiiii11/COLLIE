from pathlib import Path
import pandas as pd
import yaml
import os
import torch
import numpy as np

from utils import getDataTimeString, get_file, EasyDict

TRAJENCODER_PREFIX = "traj_encoder"
TRAJENCODER_SUFFIX = ".pt"
OPTIONPOLICY_PREFIX = "option_policy"
OPTIONPOLICY_SUFFIX = ".pt"

N_TRAJ_REQUIRED = 160


def zero_shot_eval(exp_dir, task_name, exp_identifier):
    param_file = open(os.path.join(exp_dir, "args.yaml"), 'r')
    params = EasyDict(yaml.safe_load(param_file))

    assert params.env == 'ant'
    params.env = 'ant_pref_goal_zs'
    params.downstream_reward_type = 'motion'
    if params.pref_task != task_name:
        assert params.algo in ["diayn", "lsd", "metra"]
    params.pref_task = task_name

    from run.train_zero_shot import run
    runner = run(params)

    from iod.metra import METRA
    algo: METRA = runner._algo
    traj_encoder_path = get_file(exp_dir, TRAJENCODER_PREFIX, TRAJENCODER_SUFFIX, index=None)
    traj_encoder_path_id = os.path.basename(traj_encoder_path)[len(TRAJENCODER_PREFIX):-len(TRAJENCODER_SUFFIX)]
    algo.traj_encoder = torch.load(traj_encoder_path)['traj_encoder']

    option_policy_path = get_file(exp_dir, OPTIONPOLICY_PREFIX, OPTIONPOLICY_SUFFIX, index=None)
    option_policy_path_id = os.path.basename(option_policy_path)[len(OPTIONPOLICY_PREFIX):-len(OPTIONPOLICY_SUFFIX)]
    algo.option_policy = torch.load(option_policy_path)['policy']

    n_traj = 0
    reward_list = []
    skill_list = []
    coords_list = []
    while n_traj < N_TRAJ_REQUIRED:
        paths = algo._get_train_trajectories(runner)
        path_data = algo.process_samples(paths)
        reward_arr_list = path_data["rewards"]
        for reward_arr in reward_arr_list:
            reward_list.append(reward_arr.sum())
        for path in paths:
            skill_list.append(path["agent_infos"]["option"][0])
        for path_obs in path_data['obs']:
            coords_list.append(path_obs[:, :2])
        n_traj += len(reward_arr_list)
        print(f"collected {n_traj}/{N_TRAJ_REQUIRED} data")

    reward_mean = np.mean(reward_list)
    reward_std = np.std(reward_list)
    reward_max = np.max(reward_list)
    print(f"reward: {reward_mean:.4f} ± {reward_std:.4f}, max: {reward_max}")

    import atexit
    from utils import kill_all_children
    atexit.register(kill_all_children)

    return reward_mean, reward_std, reward_max
