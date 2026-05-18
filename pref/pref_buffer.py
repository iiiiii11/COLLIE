import math

import numpy as np
import torch

from .utils import compute_state_entropy


class TrajBuffer:
    def __init__(self, capacity):
        self.trajs = None
        self.ori_trajs = None
        self.shape_state = None
        self.is_pixel_env = False
        self.state_dtype = np.float32

        self.capacity = capacity
        self.full = False
        self.idx = 0

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.idx

    def __getitem__(self, idx):
        assert self.full or idx < self.idx
        return self.trajs[idx]

    def add_traj(self, traj, ori_traj=None):
        if self.trajs is None:
            self.len_traj = traj.shape[0]
            self.shape_state = traj.shape[1:]
            if traj.shape[1] >= 10000:
                self.is_pixel_env = True
                self.state_dtype = np.uint8
            self.trajs = np.zeros((self.capacity, self.len_traj, *self.shape_state), dtype=self.state_dtype)
            if ori_traj is not None:
                self.ori_shape_state = ori_traj.shape[1:]
                self.ori_trajs = np.zeros((self.capacity, self.len_traj, *self.ori_shape_state), dtype=np.float32)

        self.trajs[self.idx] = traj
        if ori_traj is not None and self.ori_trajs is not None:
            self.ori_trajs[self.idx] = ori_traj
        self.idx += 1
        if self.idx == self.capacity:
            self.full = True
        self.idx = self.idx % self.capacity

    def add_traj_batch(self, trajs, ori_trajs=None):
        if self.trajs is None:
            self.add_traj(trajs[0], ori_traj=ori_trajs[0] if ori_trajs is not None else None)
            self.add_traj_batch(trajs[1:], ori_trajs=ori_trajs[1:] if ori_trajs is not None else None)
            return

        n_traj = trajs.shape[0]

        if trajs.shape[1:] != (self.len_traj, *self.shape_state):
            self.add_traj(trajs, ori_traj=ori_trajs)
            return

        if self.idx + n_traj > self.capacity:
            self.add_traj_batch(trajs[:self.capacity - self.idx],
                                ori_trajs=ori_trajs[:self.capacity - self.idx] if ori_trajs is not None else None)
            self.add_traj_batch(trajs[self.capacity - self.idx:],
                                ori_trajs=ori_trajs[self.capacity - self.idx:] if ori_trajs is not None else None)
            return

        self.trajs[self.idx: self.idx + n_traj] = trajs
        if ori_trajs is not None and self.ori_trajs is not None:
            self.ori_trajs[self.idx: self.idx + n_traj] = ori_trajs
        self.idx += n_traj
        assert self.idx <= self.capacity
        if self.idx == self.capacity:
            self.full = True
        self.idx = self.idx % self.capacity

    def sample(self, batch_size):
        real_size = len(self)
        idxs = np.random.randint(0, real_size, batch_size)
        return self.trajs[idxs, ...], self.ori_trajs[idxs, ...] if self.ori_trajs is not None else None


class TrajBufferWithLabel:
    def __init__(self, capacity):
        self.trajs = None
        self.labels = None
        self.shape_state = None

        self.capacity = capacity
        self.full = False
        self.idx = 0

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.idx

    def __getitem__(self, idx):
        assert self.full or idx < self.idx
        return self.trajs[idx]

    def add_traj(self, traj, label):
        if self.trajs is None:
            self.len_traj = traj.shape[0]
            self.shape_state = traj.shape[1:]
            self.trajs = np.zeros((self.capacity, self.len_traj, *self.shape_state), dtype=np.float32)
            self.labels = np.zeros(self.capacity)

        self.trajs[self.idx] = traj
        self.labels[self.idx] = label
        self.idx += 1
        if self.idx == self.capacity:
            self.full = True
        self.idx = self.idx % self.capacity

    def add_traj_batch(self, trajs, labels):
        if self.trajs is None:
            self.add_traj(trajs[0], labels[0])
            self.add_traj_batch(trajs[1:], labels[1:])
            return

        n_traj = trajs.shape[0]

        if trajs.shape[1:] != (self.len_traj, *self.shape_state):
            self.add_traj(trajs, labels)
            return

        if self.idx + n_traj > self.capacity:
            self.add_traj_batch(trajs[:self.capacity - self.idx], labels[:self.capacity - self.idx])
            self.add_traj_batch(trajs[self.capacity - self.idx:], labels[self.capacity - self.idx:])
            return

        self.trajs[self.idx: self.idx + n_traj] = trajs
        self.labels[self.idx: self.idx + n_traj] = labels

        self.idx += n_traj
        assert self.idx <= self.capacity
        if self.idx == self.capacity:
            self.full = True
        self.idx = self.idx % self.capacity


class PrefBuffer:
    def __init__(self, pref_model, capacity, device):
        self.pref_model = pref_model
        self.traj_buffer = TrajBuffer(capacity)
        self.pref_buffer = TrajBufferWithLabel(capacity)
        self.device = device

        self.first_sample = True

    def add_data(self, trajs, ori_trajs=None):

        self.traj_buffer.add_traj_batch(trajs, ori_trajs=ori_trajs)

    def uniform_sample(self, query_batchsize, query_segmentlen):
        trajs, ori_trajs = self.traj_buffer.sample(query_batchsize)
        len_traj = trajs.shape[1]

        if query_segmentlen <= 0 or query_segmentlen >= len_traj:
            return trajs, ori_trajs

        time_index = np.array([list(range(query_segmentlen)) for i in range(query_batchsize)])
        ep_start = np.random.choice(len_traj - query_segmentlen, size=query_batchsize, replace=True).reshape(-1, 1)
        time_index = time_index + ep_start
        row_indices = np.arange(trajs.shape[0])[:, None]
        segments = trajs[row_indices, time_index]
        ori_trajs = ori_trajs[row_indices, time_index] if ori_trajs is not None else None
        return segments, ori_trajs

    def max_entropy_sample(self, query_batchsize, query_segmentlen, query_large_batch_rate, query_state_entropy_batch_size):
        sample_trajs, ori_trajs = self.traj_buffer.sample(query_batchsize * query_large_batch_rate)
        n_sample_traj = sample_trajs.shape[0]
        len_traj = sample_trajs.shape[1]

        if query_segmentlen <= 0 or query_segmentlen >= len_traj:
            sample_segments = sample_trajs
            sample_ori_trajs = ori_trajs
        else:
            time_index = np.array([list(range(query_segmentlen)) for i in range(n_sample_traj)])
            ep_start = np.random.choice(len_traj - query_segmentlen, size=n_sample_traj, replace=True).reshape(-1, 1)
            time_index = time_index + ep_start
            row_indices = np.arange(sample_trajs.shape[0])[:, None]
            sample_segments = sample_trajs[row_indices, time_index]
            sample_ori_trajs = ori_trajs[row_indices, time_index] if ori_trajs is not None else None

        len_segment = sample_segments.shape[1]

        sample_segments_states = torch.from_numpy(sample_segments).to(self.device).reshape(
            [n_sample_traj * len_segment, *sample_segments.shape[2:]])
        all_labeled_states = torch.from_numpy(self.pref_buffer.trajs[:len(self.pref_buffer)]).to(self.device)
        all_labeled_states = all_labeled_states.reshape(
            [all_labeled_states.shape[0] * all_labeled_states.shape[1], *all_labeled_states.shape[2:]])

        sample_segments_states_entropy = compute_state_entropy(
            sample_segments_states, all_labeled_states, k=5, batch_size=query_state_entropy_batch_size)

        sample_segments_entropy = sample_segments_states_entropy.reshape([n_sample_traj, len_segment])
        sample_segments_entropy_sum = sample_segments_entropy.sum(dim=1)

        top_idxs = torch.argsort(sample_segments_entropy_sum, descending=True)[: query_batchsize]

        top_idxs = top_idxs.cpu().numpy()
        segments = sample_segments[top_idxs]
        return segments, sample_ori_trajs[top_idxs] if sample_ori_trajs is not None else None

    def max_uncertainty_sample(self, query_batchsize, query_segmentlen, query_large_batch_rate, get_softmax_distance):

        sample_trajs, ori_trajs = self.traj_buffer.sample(query_batchsize * query_large_batch_rate)
        n_sample_traj = sample_trajs.shape[0]
        len_traj = sample_trajs.shape[1]

        if query_segmentlen <= 0 or query_segmentlen >= len_traj:
            sample_segments = sample_trajs
            sample_ori_trajs = ori_trajs
        else:
            time_index = np.array([list(range(query_segmentlen)) for i in range(n_sample_traj)])
            ep_start = np.random.choice(len_traj - query_segmentlen, size=n_sample_traj, replace=True).reshape(-1, 1)
            time_index = time_index + ep_start
            row_indices = np.arange(sample_trajs.shape[0])[:, None]
            sample_segments = sample_trajs[row_indices, time_index]
            sample_ori_trajs = ori_trajs[row_indices, time_index] if ori_trajs is not None else None

        len_segment = sample_segments.shape[1]

        sample_segments_states = torch.from_numpy(sample_segments).to(self.device).reshape(
            [n_sample_traj * len_segment, *sample_segments.shape[2:]])
        all_labeled_states = torch.from_numpy(self.pref_buffer.trajs[:len(self.pref_buffer)]).to(self.device)
        all_labeled_states = all_labeled_states.reshape(
            [all_labeled_states.shape[0] * all_labeled_states.shape[1], *all_labeled_states.shape[2:]])

        sample_segments_softmax_distance = get_softmax_distance(sample_segments_states) + 1e-8
        sample_segments_states_uncertainty = torch.sum(sample_segments_softmax_distance *
                                                       torch.log(sample_segments_softmax_distance), dim=-1) * -1

        sample_segments_uncertainty = sample_segments_states_uncertainty.reshape([n_sample_traj, len_segment])
        sample_segments_uncertainty_sum = sample_segments_uncertainty.sum(dim=1)

        top_idxs = torch.argsort(sample_segments_uncertainty_sum, descending=True)[: query_batchsize]

        top_idxs = top_idxs.cpu().numpy()
        segments = sample_segments[top_idxs]
        return segments, sample_ori_trajs[top_idxs] if sample_ori_trajs is not None else None

    def sample_and_label(self, query_batchsize, query_segmentlen, how_to_sample, query_large_batch_rate, query_state_entropy_batch_size,
                         use_traj_insteadof_oritraj=False,
                         get_softmax_distance=None):

        if self.first_sample or how_to_sample in ('uniform', 'u'):
            trajs, ori_trajs = self.uniform_sample(query_batchsize, query_segmentlen)
        elif how_to_sample in ('state_entropy', 'se'):
            trajs, ori_trajs = self.max_entropy_sample(
                query_batchsize, query_segmentlen, query_large_batch_rate,
                query_state_entropy_batch_size)
        elif how_to_sample in ('uncertainty', 'uc'):
            assert get_softmax_distance is not None
            trajs, ori_trajs = self.max_uncertainty_sample(
                query_batchsize, query_segmentlen, query_large_batch_rate, get_softmax_distance)

        self.first_sample = False
        if use_traj_insteadof_oritraj:
            labels = self.pref_model.get_trajs_pref(trajs)
            assert False
        else:
            labels = self.pref_model.get_trajs_pref(ori_trajs)
        self.pref_buffer.add_traj_batch(trajs, labels)
        return trajs, labels, ori_trajs
