from typing import Type
import numpy as np
import torch

from .pref_buffer import PrefBuffer
from .utils import get_matd_func, softmax

class StateScoreModel_PhiDistance:
    def __init__(self, traj_encoder, device, pref_model,
                 query_segmentlen, query_method, query_large_batch_rate, query_state_entropy_batch_size,
                 capacity, labeled_state_capacity, discriminator_batchsize, weight_func,
                 weight_softmax_temp, n_sample_times_for_distance, use_phi_cache, 
                 **kwargs):
        if kwargs:
            print(f"additional params: {kwargs}")
        self.traj_encoder = traj_encoder

        self.device = device
        self.discriminator_batchsize = discriminator_batchsize
        self.d_phi = get_matd_func('l2d_torch')

        self.labeled_state_capacity = labeled_state_capacity
        self.pref_buffer = PrefBuffer(pref_model, capacity, self.device)
        self.query_segmentlen = query_segmentlen
        self.query_method = query_method
        self.query_large_batch_rate = query_large_batch_rate
        self.query_state_entropy_batch_size = query_state_entropy_batch_size

        self.uniform_dist_vec = None

        self.good_input_arr = None
        self.bad_input_arr = None
        self.neutral_input_arr = None

        self.good_loc = 0
        self.bad_loc = 0
        self.neutral_loc = 0
        self.last_good_loc = 0
        self.last_bad_loc = 0
        self.last_neutral_loc = 0

        self.good_info_arr = None
        self.bad_info_arr = None
        self.neutral_info_arr = None
        self.havent_updated = True
        self.is_pixel_env = False
        self.state_dtype = np.float32

        self.get_weight = {
            "softmax": self._get_weight_softmax,
        }.get(weight_func, lambda *args, **kwargs: None)

        self.weight_softmax_temp = weight_softmax_temp
        self.n_sample_times_for_distance = n_sample_times_for_distance
        
        self.use_phi_cache = use_phi_cache
        self.valid_cache_idx = 1
        self.good_phi_arr = None
        self.bad_phi_arr = None
        self.neutral_phi_arr = None
        self.good_phi_idx = None
        self.bad_phi_idx = None
        self.neutral_phi_idx = None

        self.n_cached = 0
        self.n_all = 0

    def batch_traj_encoder(self, state_arr):
        state_arr = torch.as_tensor(state_arr, device=self.device)
        skill_feature_list = []
        n_input = state_arr.shape[0]
        for bi in range(0, n_input, self.discriminator_batchsize):
            bi1 = min(n_input, bi + self.discriminator_batchsize)
            ids = torch.arange(bi, bi1)

            skill_feature_i = self.traj_encoder(state_arr[ids])
            skill_feature_list.append(skill_feature_i)
        skill_feature = torch.concatenate(skill_feature_list, dim=0)
        return skill_feature
    
    def load_data(self, good_states, bad_states, neutral_states, good_infos, bad_infos, neutral_infos):
        
        self.good_input_arr = np.stack([good_states, good_states], axis=1)
        self.bad_input_arr = np.stack([bad_states, bad_states], axis=1)
        self.neutral_input_arr = np.stack([neutral_states, neutral_states], axis=1)
        self.neutral_score_arr = np.zeros((self.neutral_input_arr.shape[0])) + 1  

        self.good_loc = good_states.shape[0]
        self.bad_loc = bad_states.shape[0]
        self.neutral_loc = neutral_states.shape[0]
        self.last_good_loc = 0
        self.last_bad_loc = 0
        self.last_neutral_loc = 0

        self.good_info_arr = np.stack([good_infos, good_infos], axis=1)
        self.bad_info_arr = np.stack([bad_infos, bad_infos], axis=1)
        self.neutral_info_arr = np.stack([neutral_infos, neutral_infos], axis=1)
        self.havent_updated = True

        if neutral_states.shape[1] >= 10000:  
            self.is_pixel_env = True


    def add_data(self, path_data):
        traj_list = path_data['obs']  
        ori_traj_list = path_data.get('ori_obs', None)  

        trajs = np.stack(traj_list, axis=0)
        if ori_traj_list is not None:
            ori_trajs = np.stack(ori_traj_list, axis=0)
        self.pref_buffer.add_data(trajs, ori_trajs=ori_trajs)

    def sample_and_label(self, query_batchsize):
        trajs, labels, ori_trajs = self.pref_buffer.sample_and_label(
            query_batchsize=query_batchsize, query_segmentlen=self.query_segmentlen,
            how_to_sample=self.query_method, query_large_batch_rate=self.query_large_batch_rate,
            query_state_entropy_batch_size=self.query_state_entropy_batch_size,
            traj_encoder=self.batch_traj_encoder, get_softmax_distance=self._get_softmax_distance)
        good_trajs = trajs[labels > 0.5, ...]
        bad_trajs = trajs[labels < -0.5, ...]
        neutral_trajs = trajs[np.bitwise_and(labels > -0.5, labels < 0.5), ...]

        good_ori_traj = None
        bad_ori_traj = None
        neutral_ori_traj = None
        if ori_trajs is not None:
            good_ori_traj = ori_trajs[labels > 0.5, ...]
            bad_ori_traj = ori_trajs[labels < -0.5, ...]
            neutral_ori_traj = ori_trajs[np.bitwise_and(labels > -0.5, labels < 0.5), ...]

        self.update_with_new_trajs(good_trajs, bad_trajs, neutral_trajs,
                                   good_ori_traj, bad_ori_traj, neutral_ori_traj)
        return trajs, labels

    def get_d_to_uniform(self, skill_dist, d_func):
        if self.uniform_dist_vec is None:
            dim_skill = skill_dist.shape[-1]
            self.uniform_dist_vec = np.ones(dim_skill, dtype=np.float32) / dim_skill
        d_to_uniform = d_func(skill_dist, self.uniform_dist_vec)
        return d_to_uniform

    def _update_with_new_traj(self, traj, traj_type, ori_trajs=None):
        
        n_states = traj.shape[0]

        state = traj[:-1]
        next_state = traj[1:]
        if ori_trajs is not None:
            ori_state = ori_trajs[:-1]
            ori_next_state = ori_trajs[1:]

        if self.havent_updated:
            if state.shape[1] >= 10000:  
                self.is_pixel_env = True
                self.state_dtype = np.uint8
            self.good_input_arr = np.zeros((self.labeled_state_capacity, 2, *state.shape[1:]), dtype=self.state_dtype)
            self.bad_input_arr = np.zeros((self.labeled_state_capacity, 2, *state.shape[1:]), dtype=self.state_dtype)
            self.neutral_input_arr = np.zeros((self.labeled_state_capacity, 2, *state.shape[1:]), dtype=self.state_dtype)

            if ori_trajs is not None:
                ori_state = ori_trajs[:-1]
                self.good_info_arr = np.zeros((self.labeled_state_capacity, 2, *ori_state.shape[1:]))
                self.bad_info_arr = np.zeros((self.labeled_state_capacity, 2, *ori_state.shape[1:]))
                self.neutral_info_arr = np.zeros((self.labeled_state_capacity, 2, *ori_state.shape[1:]))
            self.havent_updated = False

        if traj_type == "g":
            input_arr = self.good_input_arr
            info_arr = self.good_info_arr
            loc = self.good_loc
        elif traj_type == "b":
            input_arr = self.bad_input_arr
            info_arr = self.bad_info_arr
            loc = self.bad_loc
        elif traj_type == "n":
            input_arr = self.neutral_input_arr
            info_arr = self.neutral_info_arr
            loc = self.neutral_loc
        else:
            raise Exception(f"traj_type={traj_type}")

        state_1 = state
        next_state_1 = next_state
        n_states_1 = state_1.shape[0]

        input_arr[loc:loc + n_states_1, 0] = state_1
        input_arr[loc:loc + n_states_1, 1] = next_state_1
        if ori_trajs is not None and info_arr is not None:
            info_arr[loc:loc + n_states_1, 0] = ori_state
            info_arr[loc:loc + n_states_1, 1] = ori_next_state

        if traj_type == "g":
            self.good_loc = loc + n_states_1
        elif traj_type == "b":
            self.bad_loc = loc + n_states_1
        elif traj_type == "n":
            self.neutral_loc = loc + n_states_1
        else:
            raise Exception(f"traj_type={traj_type}")

    def update_with_new_trajs(self, good_trajs, bad_trajs, neutral_trajs,
                              good_ori_traj=None, bad_ori_traj=None, neutral_ori_traj=None):
        
        n_good_trajs = good_trajs.shape[0]
        n_bad_trajs = bad_trajs.shape[0]
        n_neutral_trajs = neutral_trajs.shape[0]
        print(f"add {n_good_trajs} good trajs, {n_bad_trajs} bad trajs, {n_neutral_trajs} neutral trajs")
        self.last_good_loc = self.good_loc
        self.last_bad_loc = self.bad_loc
        self.last_neutral_loc = self.neutral_loc
        for i in range(n_good_trajs):
            self._update_with_new_traj(good_trajs[i], traj_type="g", ori_trajs=good_ori_traj[i]
                                       if good_ori_traj is not None else None)
        for i in range(n_bad_trajs):
            self._update_with_new_traj(bad_trajs[i], traj_type="b", ori_trajs=bad_ori_traj[i]
                                       if bad_ori_traj is not None else None)
        for i in range(n_neutral_trajs):
            self._update_with_new_traj(neutral_trajs[i], traj_type="n", ori_trajs=neutral_ori_traj[i]
                                       if neutral_ori_traj is not None else None)
        print(f"total {self.good_loc} good states, {self.bad_loc} bad states, {self.neutral_loc} neutral states")

    def get_last_labeled_states(self):
        
        good_states = self.good_input_arr[self.last_good_loc:self.good_loc, 1]  
        bad_states = self.bad_input_arr[self.last_bad_loc:self.bad_loc, 1]  
        neutral_states = self.neutral_input_arr[self.last_neutral_loc:self.neutral_loc, 1]

        if self.good_info_arr is not None:
            good_infos = self.good_info_arr[self.last_good_loc:self.good_loc, 1]
            bad_infos = self.bad_info_arr[self.last_bad_loc:self.bad_loc, 1]
            neutral_infos = self.neutral_info_arr[self.last_neutral_loc:self.neutral_loc, 1]
            return good_states, bad_states, neutral_states, good_infos, bad_infos, neutral_infos

        return good_states, bad_states, neutral_states, None, None, None

    def get_all_labeled_states(self):
        
        good_states = self.good_input_arr[:self.good_loc, 1]  
        bad_states = self.bad_input_arr[:self.bad_loc, 1]  
        neutral_states = self.neutral_input_arr[:self.neutral_loc, 1]

        if self.good_info_arr is not None:
            good_infos = self.good_info_arr[:self.good_loc, 1]
            bad_infos = self.bad_info_arr[:self.bad_loc, 1]
            neutral_infos = self.neutral_info_arr[:self.neutral_loc, 1]
            return good_states, bad_states, neutral_states, good_infos, bad_infos, neutral_infos

        return good_states, bad_states, neutral_states,  None, None, None

    def print_cache_rate(self, clear_stats=False):
        if self.n_all == 0:
            print(f"cache: {self.n_cached} / {self.n_all}")
        else:
            print(f"cache: {self.n_cached}/{self.n_all} = {self.n_cached/self.n_all}")
        if clear_stats:
            self.n_cached = 0
            self.n_all = 0
        return self.n_cached, self.n_all

    @torch.no_grad()
    def cached_traj_encoder(self, traj_type, ids):
        if traj_type == "g":
            input_arr = self.good_input_arr
        elif traj_type == "b":
            input_arr = self.bad_input_arr
        elif traj_type == "n":
            input_arr = self.neutral_input_arr
        else:
            raise Exception(f"traj_type={traj_type}")

        state_arr = torch.from_numpy(input_arr[ids.cpu().numpy(), 1]).to(self.device).to(torch.float32)

        if self.good_phi_arr is None:
            input_skill_feature = self.traj_encoder(state_arr[2:, ...])
            self.good_phi_arr = torch.zeros((self.labeled_state_capacity, *input_skill_feature.shape[1:])).to(self.device)
            self.bad_phi_arr = torch.zeros((self.labeled_state_capacity, *input_skill_feature.shape[1:])).to(self.device)
            self.neutral_phi_arr = torch.zeros((self.labeled_state_capacity, *input_skill_feature.shape[1:])).to(self.device)
            self.good_phi_idx = torch.zeros((self.labeled_state_capacity,)).to(self.device)
            self.bad_phi_idx = torch.zeros((self.labeled_state_capacity,)).to(self.device)
            self.neutral_phi_idx = torch.zeros((self.labeled_state_capacity,)).to(self.device)

        if traj_type == "g":
            input_arr = self.good_input_arr
            cache_phi_arr = self.good_phi_arr
            cache_phi_idx = self.good_phi_idx
        elif traj_type == "b":
            input_arr = self.bad_input_arr
            cache_phi_arr = self.bad_phi_arr
            cache_phi_idx = self.bad_phi_idx
        elif traj_type == "n":
            input_arr = self.neutral_input_arr
            cache_phi_arr = self.neutral_phi_arr
            cache_phi_idx = self.neutral_phi_idx
        else:
            raise Exception(f"traj_type={traj_type}")

        cached_mask = cache_phi_idx[ids] == self.valid_cache_idx
        not_cached_mask = torch.bitwise_not(cached_mask)
        ids_cached = ids[cached_mask]
        ids_not_cached = ids[not_cached_mask]

        skill_feature_cached = cache_phi_arr[ids_cached]
        state_arr_not_cached = state_arr[not_cached_mask, ...]

        n_input = state_arr_not_cached.shape[0]
        if n_input == 0:
            
            skill_feature_not_cached = torch.zeros((0, *self.good_phi_arr.shape[1:])).to(self.device)
        else:
            skill_feature_not_cached_list = []
            for bi in range(0, n_input, self.discriminator_batchsize):
                bi1 = min(n_input, bi + self.discriminator_batchsize)
                ids = torch.arange(bi, bi1)
                if state_arr_not_cached[ids].shape[0] == 0:
                    continue
                skill_feature_not_cached_i = self.traj_encoder(state_arr_not_cached[ids])
                skill_feature_not_cached_list.append(skill_feature_not_cached_i)
            skill_feature_not_cached = torch.concatenate(skill_feature_not_cached_list, dim=0)

        skill_feature_return = torch.zeros((state_arr.shape[0], *self.good_phi_arr.shape[1:])).to(self.device)
        skill_feature_return[not_cached_mask] = skill_feature_not_cached
        skill_feature_return[cached_mask] = skill_feature_cached

        
        cache_phi_idx[ids_not_cached] = self.valid_cache_idx
        cache_phi_arr[ids_not_cached] = skill_feature_not_cached

        self.n_cached += ids_cached.shape[0]
        self.n_all += state_arr.shape[0]

        return skill_feature_return

    @torch.no_grad()
    def _get_weighted_min_distance(self, state_features, traj_type, default):
        if traj_type == "g":
            input_arr = self.good_input_arr
            n_input = self.good_loc
        elif traj_type == "b":
            input_arr = self.bad_input_arr
            n_input = self.bad_loc
        elif traj_type == "n":
            input_arr = self.neutral_input_arr
            n_input = self.neutral_loc
        else:
            raise Exception(f"traj_type={traj_type}")

        if input_arr is None or n_input == 0:
            return torch.zeros(state_features.shape[0]).to(self.device) + default

        def _get_traj_feature(ids):
            if self.use_phi_cache:
                input_skill_feature = self.cached_traj_encoder(traj_type, ids)  
            else:
                input_skill_feature_list = []
                ids_length = ids.shape[0]
                for bi in range(0, ids_length, self.discriminator_batchsize):
                    bi1 = min(ids_length, bi + self.discriminator_batchsize)
                    subids = torch.arange(bi, bi1)
                    ids_i = ids[subids]
                    state_arr = torch.from_numpy(input_arr[ids_i.cpu().numpy(), 1]).to(self.device).to(torch.float32)
                    input_skill_feature_i = self.traj_encoder(state_arr)
                    input_skill_feature_list.append(input_skill_feature_i)
                input_skill_feature = torch.concatenate(input_skill_feature_list, dim=0)
            return input_skill_feature

        if self.n_sample_times_for_distance > 0:  
            ids_all = torch.randint(0, n_input, (self.n_sample_times_for_distance * self.discriminator_batchsize,))
        else:
            ids_all = torch.arange(0, n_input)
        ids_all = ids_all.to(self.device)

        input_skill_feature = _get_traj_feature(ids_all)
        d_to_input = self.d_phi(state_features, input_skill_feature)
        weighted_d = d_to_input 

        min_weighted_d = torch.min(weighted_d, dim=1)[0]
        ret = min_weighted_d
        return ret

    @torch.no_grad()
    def _get_weight_softmax(self, last_states, states, **kwargs):
        state_features = self.traj_encoder(states)
        weighted_d_good = self._get_weighted_min_distance(state_features, traj_type="g", default=1e10)
        weighted_d_bad = self._get_weighted_min_distance(state_features, traj_type="b", default=1e10)
        weighted_d_neutral = self._get_weighted_min_distance(state_features, traj_type="n", default=1e10)
        weight_stack = torch.stack([weighted_d_good, weighted_d_neutral, weighted_d_bad], dim=-1) * -1
        weight_norm = torch.softmax(weight_stack * self.weight_softmax_temp, dim=-1)
        weight = (weight_norm @ (torch.tensor([2., 1., 0.], device=self.device)[None, :]).T).squeeze(-1)
        return weight.cpu().numpy()
    
    @torch.no_grad()
    def _get_softmax_distance(self, states):
        state_features = self.traj_encoder(states)
        weighted_d_good = self._get_weighted_min_distance(state_features, traj_type="g", default=1e10)
        weighted_d_bad = self._get_weighted_min_distance(state_features, traj_type="b", default=1e10)
        weighted_d_neutral = self._get_weighted_min_distance(state_features, traj_type="n", default=1e10)
        weight_stack = torch.stack([weighted_d_good, weighted_d_neutral, weighted_d_bad], dim=-1) * -1
        weight_norm = torch.softmax(weight_stack * self.weight_softmax_temp, dim=-1)
        return weight_norm

    def update_cache(self, clear_stats=False):
        self.valid_cache_idx += 1
        if clear_stats:
            self.n_cached = 0
            self.n_all = 0


StateScoreModel_Dict = {
    "phi_distance": StateScoreModel_PhiDistance,
    "pd": StateScoreModel_PhiDistance,
}


def get_state_score_model_class(score_model_name) -> Type[StateScoreModel_PhiDistance]:
    StateScoreModel = StateScoreModel_Dict[score_model_name]
    return StateScoreModel
