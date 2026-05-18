import pathlib
import pickle
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


def kl(p, q):
    assert p.shape == q.shape and len(p.shape) == 1
    return np.sum(p * np.log(p / (q + 1e-8)))


def cosd(p, q):
    assert p.shape == q.shape and len(p.shape) == 1
    num = float(np.dot(p, q))
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    coss = 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    cosd = 1 - coss
    return cosd


def cosine_distance_matrix(p, q, eps=1e-8):
    if len(p.shape) == 1:
        p = p[None]
    if len(q.shape) == 1:
        q = q[None]

    norm_p = np.linalg.norm(p, axis=1, keepdims=True)
    norm_q = np.linalg.norm(q, axis=1)

    norm_p = np.where(norm_p < eps, 1.0, norm_p)
    norm_q = np.where(norm_q < eps, 1.0, norm_q)

    p_norm = p / norm_p
    q_norm = q / norm_q[:, np.newaxis]

    cos_sim = p_norm @ q_norm.T

    cos_sim = np.clip(cos_sim, -1.0, 1.0)

    cos_sim = 0.5 + 0.5 * cos_sim
    d = 1 - cos_sim

    zero_p = (np.linalg.norm(p, axis=1) <= eps)
    zero_q = (np.linalg.norm(q, axis=1) <= eps)
    both_zero = zero_p[:, np.newaxis] & zero_q

    d[both_zero] = 0.0

    return d


def cosine_distance_matrix_torch(p: torch.Tensor,
                                 q: torch.Tensor,
                                 eps: float = 1e-8) -> torch.Tensor:

    if p.dim() == 1:
        p = p.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)

    norm_p = torch.norm(p, dim=1, keepdim=True)
    norm_q = torch.norm(q, dim=1)

    norm_p = torch.where(norm_p < eps, torch.tensor(1.0, device=p.device), norm_p)
    norm_q = torch.where(norm_q < eps, torch.tensor(1.0, device=q.device), norm_q)

    p_norm = p / norm_p
    q_norm = q / norm_q.unsqueeze(1)

    cos_sim = torch.mm(p_norm, q_norm.t())

    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    cos_sim = 0.5 + 0.5 * cos_sim
    d = 1 - cos_sim

    zero_p = (torch.norm(p, dim=1) <= eps)
    zero_q = (torch.norm(q, dim=1) <= eps)
    both_zero = zero_p.unsqueeze(1) & zero_q
    d = torch.where(both_zero,
                    torch.tensor(0.0, device=p.device),
                    d)
    return d


def softmax(x, axis):
    max_val = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - max_val)
    e_x_norm = e_x / np.sum(e_x, axis=axis, keepdims=True)
    return e_x_norm


def tvd(p, q):
    '''
    Total Variation Distance, 1/2*\sum_x|p(x)-q(x)|
    '''
    assert p.shape == q.shape and len(p.shape) == 1
    d = np.sum(np.abs(p - q)) / 2
    return d


def total_variation_distance_matrix(p, q):

    if len(p.shape) == 1:
        p = p[None]
    if len(q.shape) == 1:
        q = q[None]

    abs_diff = np.abs(p[:, None, :] - q[None, :, :])

    d = 0.5 * np.sum(abs_diff, axis=2)

    return d


def total_variation_distance_matrix_torch(p: torch.Tensor,
                                          q: torch.Tensor) -> torch.Tensor:

    if p.dim() == 1:
        p = p.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)

    abs_diff = torch.abs(p.unsqueeze(1) - q.unsqueeze(0))
    d = 0.5 * abs_diff.sum(dim=2)

    return d


def l2_distance_matrix(p, q):

    if len(p.shape) == 1:
        p = p[None]
    if len(q.shape) == 1:
        q = q[None]

    diff = p[:, np.newaxis, :] - q[np.newaxis, :, :]

    d = np.sqrt(np.sum(diff ** 2, axis=2))

    return d


def l2_distance_matrix_torch(p: torch.Tensor,
                             q: torch.Tensor) -> torch.Tensor:

    if p.dim() == 1:
        p = p.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)

    diff = p.unsqueeze(1) - q.unsqueeze(0)
    d = torch.sqrt(torch.sum(diff ** 2, dim=2))

    return d


def get_matd_func(name):
    return {
        'cosd': cosine_distance_matrix,
        'tvd': total_variation_distance_matrix,
        'l2d': l2_distance_matrix,
        'cosd_torch': cosine_distance_matrix_torch,
        'tvd_torch': total_variation_distance_matrix_torch,
        'l2d_torch': l2_distance_matrix_torch,
    }[name]


def compute_state_entropy(obs, full_obs, k, batch_size):
    with torch.no_grad():
        dists = []
        for idx in range(len(full_obs) // batch_size + 1):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            dist = torch.norm(obs[:, None, :] - full_obs[None, start:end, :],
                              dim=-1, p=2)
            dists.append(dist)

        dists = torch.cat(dists, dim=1)
        knn_dists = torch.kthvalue(dists, k=k + 1, dim=1).values
        state_entropy = knn_dists
    return state_entropy.unsqueeze(1)


def save_queries_pickle(path: pathlib.Path,
                        good_states, bad_states, neutral_states,
                        good_infos, bad_infos, neutral_infos):

    data = {
        'good_states': good_states,
        'bad_states': bad_states,
        'neutral_states': neutral_states,
        'good_infos': good_infos,
        'bad_infos': bad_infos,
        'neutral_infos': neutral_infos
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {path}")


def load_queries_pickle(file_name):

    with open(file_name, 'rb') as f:
        return pickle.load(f)


class RandomShiftPixel:
    def __init__(self, max_shift=4, padding_mode='reflect'):
        self.max_shift = max_shift
        self.padding_mode = padding_mode

    def __call__(self, img):

        x_shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        y_shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()

        padding = (self.max_shift, self.max_shift, self.max_shift, self.max_shift)

        img = F.pad(img, padding, padding_mode=self.padding_mode)

        left = self.max_shift + x_shift
        top = self.max_shift + y_shift
        right = left + img.size(-1) - 2 * self.max_shift
        bottom = top + img.size(-2) - 2 * self.max_shift

        img = F.crop(img, top, left, bottom - top, right - left)

        return img
