from typing import Dict, List

import numpy as np
import torch
from torch.nn import functional as F

from iod.metra import METRA
from iod.sac_utils import _clip_actions
from pref.oracle_pref import OracleStatePref


class MetraPref(METRA):
    """
    metra + pref (oracle)
    """

    def __init__(
            self,

            **kwargs,
    ):
        super().__init__(**kwargs)
        self.pref_model = OracleStatePref(kwargs["env_name"], self.pref_task)

    def _update_replay_buffer(self, data: Dict[str, List[np.ndarray]]) -> None:
        """Update the replay buffer with newly collected data.

        Args:
            data (Dict[str, List[np.ndarray]]): data to add to replay buffer
        """

        n_ori_obs = data['next_ori_obs']
        data['next_obs_pref'] = self.pref_model.get_state_pref(n_ori_obs)

        super()._update_replay_buffer(data)

    def _update_rewards(self, train_store: Dict, mini_batch: Dict) -> None:
        """Compute the rewards for the current mini batch using the learned representations.

        Args:
            train_store (Dict): training store
            mini_batch (Dict): mini batch data
        """
        obs = mini_batch['obs']
        next_obs = mini_batch['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean

            target_z = next_z - cur_z

            if self.no_diff_in_rep:
                target_z = cur_z

            if self.self_normalizing:
                target_z = target_z / target_z.norm(dim=-1, keepdim=True)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = target_z @ new_z.t()
                else:
                    pairwise_scores = target_z @ mini_batch['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

            next_obs_pref = mini_batch["next_obs_pref"]
            if self.discrete:
                masks = (mini_batch['options'] - mini_batch['options'].mean(dim=1, keepdim=True)) * \
                    self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks * self.pref_coef).sum(dim=1) * next_obs_pref

            else:
                inner = (target_z * mini_batch['options'] * self.pref_coef).sum(dim=1) * next_obs_pref
                rewards = inner

            mini_batch.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })

        elif self.metra_mlp_rep:

            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            mini_batch.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })

            rep = self.f_encoder(obs, next_obs)
            rewards = (rep * mini_batch['options']).sum(dim=1)

            if self.log_sum_exp:
                if self.sample_new_z:
                    new_z = torch.randn(self.num_negative_z, self.dim_option, device=mini_batch['options'].device)
                    if self.unit_length:
                        new_z /= torch.norm(new_z, dim=-1, keepdim=True)
                    pairwise_scores = rep @ new_z.t()
                else:
                    pairwise_scores = rep @ mini_batch['options'].t()
                log_sum_exp = torch.logsumexp(pairwise_scores, dim=-1)

        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits,
                                                             mini_batch['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(mini_batch['options'])

            if self.diayn_include_baseline:
                rewards -= torch.log(torch.tensor(1 / self.dim_option))

        train_store.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        mini_batch['rewards'] = rewards
        if self.log_sum_exp:
            mini_batch['log_sum_exp'] = log_sum_exp
