
import sys
from typing import Any

import gym

import dowel_wrapper
from garaged.src.garage.experiment.local_runner import TrainArgs

assert dowel_wrapper is not None
import dowel

import argparse
import datetime
import functools
import os
import platform
import torch.multiprocessing as mp

if 'mac' in platform.platform():
    pass
else:
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal

from garaged.src.garage.torch.modules import MLPModule

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.q_functions.discrete_mlp_q_function import DiscreteMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex
from garagei.envs.child_policy_env import ChildPolicyEnv
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.torch.modules.gaussian_mlp_module_ex import GaussianMLPTwoHeadedModuleEx
from garagei.torch.modules.categorical_mlp_module_ex import CategoricalMLPModuleEx

from iod.metra import METRA
from iod.metra_raw import METRA_RAW
from iod.metra_sf import MetraSf
from iod.metra_pref import MetraPref
from iod.metra_pref_query import MetraPrefQuery
from iod.dads import DADS
from iod.ppo import PPO
from iod.cic import CIC
from iod.sac import SAC
from iod.utils import get_normalizer_preset, get_gaussian_module_construction, get_log_dir, get_exp_name, get_date_time_string

from utils import save_cmd_args

EXP_DIR = 'exp'
if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'


def make_env(args: argparse.Namespace, max_path_length: int) -> Any:
    """Build environment.

    Args:
        args (argparse.Namespace): argparse arguments
        max_path_length (int): specifies the maximum number of timesteps of a single rollout

    Returns:
        Any: specified environmemt to use
    """
    if args.env == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif args.env == 'cmaze':
        from envs.maze_complex import WrappedComplexMazeEnv
        env = WrappedComplexMazeEnv(
            n=max_path_length,
            maze_config=args.maze_config,
        )  

    elif args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)

    elif args.env == 'ant':
        from envs.mujoco.ant_env import AntEnv
        env = AntEnv(render_hw=100, model_path='ant.xml')

    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder  
        if 'dmc_quadruped' in args.env:
            env = dmc.make('quadruped_run_forward_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        elif 'dmc_humanoid' in args.env:
            env = dmc.make('humanoid_run_color', obs_type='states', frame_stack=1, action_repeat=2, seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
        
        if args.env in ['dmc_quadruped_goal', 'dmc_humanoid_goal']:
            from envs.custom_dmc_tasks.goal_wrappers import GoalWrapper

            env = GoalWrapper(
                env,
                max_path_length=max_path_length,
                goal_range=args.goal_range,
                num_goal_steps=args.downstream_num_goal_steps,
            )
            cp_num_truncate_obs = 2

    elif args.env.startswith('robobin'):
        sys.path.append('lexa')
        from envs.lexa.robobin import MyRoboBinEnv
        if args.env == 'robobin':
            env = MyRoboBinEnv(log_per_goal=True)
        elif args.env == 'robobin_image':
            env = MyRoboBinEnv(obs_type='image', log_per_goal=True)

    elif args.env == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder  
        env = MyKitchenEnv(log_per_goal=True)

    elif args.env.startswith('Safexp'):
        import safety_gym
        env = gym.make(args.env)

    elif args.env.startswith('CSafe'):
        import envs.customized_safety_gym
        env = gym.make(args.env)
        from envs.safety_gym_wrapper import SafetyGymWrapper
        env = SafetyGymWrapper(env)

        args.normalizer_type = "off"
        print("force args.normalizer_type to off")

    elif args.env.startswith('PCSafe'):
        import envs.customized_safety_gym
        env = gym.make(args.env[1:])
        from envs.safety_gym_wrapper import PixelSafetyGymWrapper
        env = PixelSafetyGymWrapper(env)

    elif args.env == 'ant_nav_prime':
        from envs.mujoco.ant_nav_prime_env import AntNavPrimeEnv

        env = AntNavPrimeEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            num_goal_steps=args.downstream_num_goal_steps,
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 2

    elif args.env == 'ant_pref_goal':
        from envs.mujoco.ant_pref_goals_env import AntPrefGoalEnv

        env = AntPrefGoalEnv(
            max_path_length=max_path_length,
            pref_task=args.pref_task,
            goal_range=args.goal_range,
            num_goal_steps=args.downstream_num_goal_steps,
            reward_type=args.downstream_reward_type,
            zero_shot=False,
        )
        cp_num_truncate_obs = 2

    elif args.env == 'ant_pref_goal_zs':
        from envs.mujoco.ant_pref_goals_env import AntPrefGoalEnv

        env = AntPrefGoalEnv(
            max_path_length=max_path_length,
            pref_task=args.pref_task,
            goal_range=args.goal_range,
            num_goal_steps=args.downstream_num_goal_steps,
            reward_type=args.downstream_reward_type,
            zero_shot=True,
        )
        cp_num_truncate_obs = 0

    elif args.env == 'half_cheetah_goal':
        from envs.mujoco.half_cheetah_goal_env import HalfCheetahGoalEnv
        env = HalfCheetahGoalEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 1
    
    elif args.env == 'half_cheetah_goal_notflip':
        from envs.mujoco.half_cheetah_goal_env_notflip import HalfCheetahGoalEnv
        env = HalfCheetahGoalEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            reward_type=args.downstream_reward_type,
            zero_shot=False,
        )
        cp_num_truncate_obs = 1

    elif args.env == 'half_cheetah_goal_notflip_zs':
        from envs.mujoco.half_cheetah_goal_env_notflip import HalfCheetahGoalEnv
        env = HalfCheetahGoalEnv(
            max_path_length=max_path_length,
            goal_range=args.goal_range,
            reward_type=args.downstream_reward_type,
            zero_shot=True,
        )
        cp_num_truncate_obs = 0

    elif args.env == 'half_cheetah_hurdle':
        from envs.mujoco.half_cheetah_hurdle_env import HalfCheetahHurdleEnv

        env = HalfCheetahHurdleEnv(
            reward_type=args.downstream_reward_type,
        )
        cp_num_truncate_obs = 2
    
    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, args.frame_stack)

    
    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        additional_dim = 0
        if args.env in ['ant_nav_prime', 'ant_pref_goal', 'ant_pref_goal_zs']:
            normalizer_name = 'ant'
            additional_dim = cp_num_truncate_obs
        elif args.env in ['half_cheetah_goal', 'half_cheetah_hurdle', 'half_cheetah_goal_notflip', 'half_cheetah_goal_notflip_zs']:
            normalizer_name = 'half_cheetah'
            additional_dim = cp_num_truncate_obs
        else:
            normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        if additional_dim > 0:
            normalizer_mean = np.concatenate([normalizer_mean, np.zeros(additional_dim)])
            normalizer_std = np.concatenate([normalizer_std, np.ones(additional_dim)])
        env = consistent_normalize(env, normalize_obs=True, mean=normalizer_mean, std=normalizer_std, **normalizer_kwargs)

    
    if args.cp_path is not None:
        cp_path = args.cp_path
        if not os.path.exists(cp_path):
            import glob
            cp_path = glob.glob(cp_path)[0]
        cp_dict = torch.load(cp_path, map_location='cpu')

        env = ChildPolicyEnv(
            env,
            cp_dict,
            cp_action_range=1.5,
            cp_unit_length=args.cp_unit_length,
            cp_multi_step=args.cp_multi_step,
            cp_num_truncate_obs=cp_num_truncate_obs,
        )

    return env


class EasyDict:
    def __init__(self, data: dict):
        self.__dict__.update(data)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value


def run(args):
    
    dowel.logger.log('ARGS: ' + str(args))

    
    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    
    set_seed(args.seed)

    
    ctxt = EasyDict(dict(
        snapshot_dir='./exp/Debug/Train_hack_params',
        snapshot_gap=1,
        snapshot_mode='last',
    ))
    runner = OptionLocalRunner(ctxt)

    
    max_path_length = args.max_path_length
    if args.cp_path is not None:
        max_path_length *= args.cp_multi_step
    contextualized_make_env = functools.partial(make_env, args=args, max_path_length=max_path_length)
    env = contextualized_make_env()
    example_ob = env.reset()

    
    if args.encoder:
        if hasattr(env, 'ob_info'):
            if env.ob_info['type'] in ['hybrid', 'pixel']:
                pixel_shape = env.ob_info['pixel_shape']
        else:
            pixel_shape = (64, 64, 3)
    else:
        pixel_shape = None

    
    device = torch.device('cuda' if args.use_gpu else 'cpu')

    
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    
    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None

    
    obs_dim = env.spec.observation_space.flat_dim
    action_dim = env.spec.action_space.flat_dim
    
    
    if args.encoder:
        def make_encoder(**kwargs):
            '''
            make a pixel encoder: pixel -> feature
            '''
            return Encoder(pixel_shape=pixel_shape, **kwargs)

        def with_encoder(module, encoder=None):
            if encoder is None:
                kwargs = {}
                encoder = make_encoder(**kwargs)

            return WithEncoder(encoder=encoder, module=module)

        kwargs = {}
        example_encoder = make_encoder(**kwargs)
        module_obs_dim = example_encoder(torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
    else:
        module_obs_dim = obs_dim
    
    option_info = {
        'dim_option': args.dim_option,
    }

    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )
    module_kwargs = dict(
        hidden_sizes=master_dims,
        layer_normalization=False,
    )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)

    if args.use_discrete_sac:
        module_cls = CategoricalMLPModuleEx
        module_kwargs.update(dict(
            categorical_distribution_cls=OneHotCategorical,
        ))
    elif args.policy_type == 'categorical':
        module_cls = CategoricalMLPModuleEx
        module_kwargs.update(dict(
            categorical_distribution_cls=OneHotCategorical,
        ))
    else:
        module_cls = GaussianMLPTwoHeadedModuleEx
        module_kwargs.update(dict(
            max_std=np.exp(2.),
            normal_distribution_cls=TanhNormal,
            output_w_init=functools.partial(xavier_normal_ex, gain=1.),
            init_std=1.,
        ))

    policy_q_input_dim = module_obs_dim + args.dim_option

    if args.algo in ['sac', 'ppo']:
        policy_q_input_dim = module_obs_dim

    policy_module = module_cls(
        input_dim=policy_q_input_dim,
        output_dim=action_dim,
        **module_kwargs
    )
    if args.encoder:
        policy_module = with_encoder(policy_module)

    policy_kwargs['module'] = policy_module
    option_policy = PolicyEx(**policy_kwargs)
    
    output_dim = args.dim_option
    traj_encoder_obs_dim = module_obs_dim
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=traj_encoder_obs_dim,
        output_dim=output_dim,
        layer_normalization=args.use_layer_norm
    )
    traj_encoder = module_cls(**module_kwargs)
    if args.encoder:
        if args.spectral_normalization:
            te_encoder = make_encoder(spectral_normalization=True)
        else:
            te_encoder = None
        traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)

    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=obs_dim,
        output_dim=obs_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    if args.dual_dist == 's2_from_s':
        dist_predictor = module_cls(**module_kwargs)
    else:
        dist_predictor = None

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    sd_dim_option = args.dim_option
    skill_dynamics_obs_dim = obs_dim
    skill_dynamics_input_dim = skill_dynamics_obs_dim + sd_dim_option
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        const_std=args.sd_const_std,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        input_dim=skill_dynamics_input_dim,
        output_dim=skill_dynamics_obs_dim,
        min_std=0.3,
        max_std=10.0,
    )
    if args.algo == 'dads':
        skill_dynamics = module_cls(**module_kwargs)
    else:
        skill_dynamics = None

    
    if args.algo == 'cic':
        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=args.dim_option * 2,
            output_dim=args.dim_option,
            layer_normalization=False, 
        )
        pred_net = module_cls(**module_kwargs)

        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=args.dim_option,
            output_dim=output_dim,
            layer_normalization=False, 
            spectral_normalization=True,
        )
        z_encoder = module_cls(**module_kwargs)
    else:
        pred_net = None

    def _finalize_lr(lr: float) -> float:
        """Post-process learning rate.

        Args:
            lr (float): initial lr, which can be None

        Returns:
            float: final lr after processing
        """
        if lr is None:
            lr = args.common_lr
        else:
            assert bool(lr), 'To specify a lr of 0, use a negative value'
        if lr < 0.0:
            dowel.logger.log(f'Setting lr to ZERO given {lr}')
            lr = 0.0
        return lr

    if pred_net is not None:
        te_params = list(traj_encoder.parameters()) + list(z_encoder.parameters()) + list(pred_net.parameters())
    else:
        te_params = list(traj_encoder.parameters())

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args.lr_op)},
        ]),
        'traj_encoder': torch.optim.Adam([
            {'params': te_params, 'lr': _finalize_lr(args.lr_te)},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': _finalize_lr(args.dual_lr)},
        ]),
    }

    if skill_dynamics is not None:
        optimizers.update({
            'skill_dynamics': torch.optim.Adam([
                {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })

    if dist_predictor is not None:
        optimizers.update({
            'dist_predictor': torch.optim.Adam([
                {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    replay_buffer = PathBufferEx(
        capacity_in_transitions=int(args.sac_max_buffer_size),
        pixel_shape=pixel_shape,
        sample_goals=(args.algo == 'crl' or 'metra_sf' in args.algo),
        discount=args.sac_discount,
    )

    qf1 = None
    qf2 = None
    log_alpha = None
    if args.algo in ['metra', 'metra_raw', 'metra_pref', 'metra_pref_query', 'metra_test', 'metra_online_test', 'dads', 'sac', 'cic']:
        if args.use_discrete_sac:
            qf1 = DiscreteMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        else:
            qf1 = ContinuousMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        if args.encoder:
            qf1 = with_encoder(qf1)

        if args.use_discrete_sac:
            qf2 = DiscreteMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        else:
            qf2 = ContinuousMLPQFunctionEx(
                obs_dim=policy_q_input_dim,
                action_dim=action_dim,
                hidden_sizes=master_dims,
                hidden_nonlinearity=nonlinearity or torch.relu,
            )
        if args.encoder:
            qf2 = with_encoder(qf2)
        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })
    
    
    
    elif args.algo in ['metra_sf', 'metra_sf_pref']:
        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )
        if args.encoder:
            qf1 = with_encoder(qf1)

        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )
        if args.encoder:
            qf2 = with_encoder(qf2)

        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))
        optimizers.update({
            'qf': torch.optim.Adam([
                {'params': list(qf1.parameters()) + list(qf2.parameters()), 'lr': _finalize_lr(args.sac_lr_q)},
            ]),
            'log_alpha': torch.optim.Adam([
                {'params': log_alpha.parameters(), 'lr': _finalize_lr(args.sac_lr_a)},
            ])
        })

    elif args.algo == 'ppo':
        
        vf = MLPModule(
            input_dim=policy_q_input_dim,
            output_dim=1,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            layer_normalization=False,
        )
        optimizers.update({
            'vf': torch.optim.Adam([
                {'params': vf.parameters(), 'lr': _finalize_lr(args.lr_op)},
            ]),
        })

    
    f_encoder = None
    if args.metra_mlp_rep:
        f_encoder = ContinuousMLPQFunctionEx(
            obs_dim=obs_dim,
            action_dim=obs_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            output_dim=args.dim_option,
        )

        optimizers.update({
            'f_encoder': torch.optim.Adam([
                {'params': list(f_encoder.parameters()), 'lr': _finalize_lr(args.lr_te)},
            ]),
        })

    
    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_kwargs = dict(
        env_name=args.env,
        algo=args.algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        skill_dynamics=skill_dynamics,
        dist_predictor=dist_predictor,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_log=args.n_epochs_per_log,
        n_epochs_per_tb=args.n_epochs_per_log,
        n_epochs_per_save=args.n_epochs_per_save,
        n_epochs_per_pt_save=args.n_epochs_per_pt_save,
        n_epochs_per_pkl_update=args.n_epochs_per_eval if args.n_epochs_per_pkl_update is None else args.n_epochs_per_pkl_update,
        dim_option=args.dim_option,
        num_random_trajectories=args.num_random_trajectories,
        num_video_repeats=args.num_video_repeats,
        eval_record_video=args.eval_record_video if args.env not in ("maze", "cmaze") else False, 
        video_skip_frames=args.video_skip_frames,
        eval_plot_axis=args.eval_plot_axis,
        name=args.algo,
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        sd_batch_norm=args.sd_batch_norm,
        skill_dynamics_obs_dim=skill_dynamics_obs_dim,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,

        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,

        pixel_shape=pixel_shape
    )

    
    
    
    if args.algo == 'metra':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,

            pref_task=args.pref_task,
            pref_coef=args.pref_coef,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = METRA(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'metra_raw': 
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = METRA_RAW(
            **algo_kwargs,
            **skill_common_args,
        )
        
    elif args.algo == 'metra_sf':  
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = MetraSf(  
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'metra_pref': 
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,

            pref_task=args.pref_task,
            pref_coef=args.pref_coef,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = MetraPref(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'metra_pref_query':  
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
            num_zero_shot_goals=args.num_zero_shot_goals,

            pref_task=args.pref_task,
            pref_coef=args.pref_coef,

            pb_capacity=args.pb_capacity,
            labeled_state_capacity=args.labeled_state_capacity,
            query_warmup=args.query_warmup,
            query_freq=args.query_freq,
            query_limit=args.query_limit,
            query_batchsize=args.query_batchsize,
            query_segmentlen=args.query_segmentlen,
            query_method=args.query_method,
            query_large_batch_rate=args.query_large_batch_rate,
            query_state_entropy_batch_size=args.query_state_entropy_batch_size,
            discriminator_batchsize=args.discriminator_batchsize,
            score_model_name=args.score_model_name,
            weight_func=args.weight_func,
            weight_softmax_temp=args.weight_softmax_temp,
            n_sample_times_for_distance=args.n_sample_times_for_distance,
            use_phi_cache=args.use_phi_cache,
            weight_smooth_decay_speed=args.weight_smooth_decay_speed,
        )
        skill_common_args.update(
            inner=args.inner,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )
        algo = MetraPrefQuery(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'cic':
        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )

        algo = CIC(
            **algo_kwargs,
            **skill_common_args,

            pred_net=pred_net,
            z_encoder=z_encoder,
            cic_temp=args.cic_temp,
            cic_alpha=args.cic_alpha,
            knn_k=args.apt_knn_k,
            rms=args.apt_rms,
            alive_reward=args.alive_reward,

            dual_dist_scaling=args.dual_dist_scaling,
            const_scaler=args.const_scaler,
            wdm=args.wdm,
            wdm_cpc=args.wdm_cpc,
            wdm_idz=args.wdm_idz,
            wdm_ids=args.wdm_ids,
            wdm_diff=args.wdm_diff,
            aug=args.aug,
            joint_train=args.joint_train,
        )
    elif args.algo == 'sac':
        algo_kwargs.update(
            use_discrete_sac=args.use_discrete_sac,
        )

        algo = SAC(
            **algo_kwargs,
            **skill_common_args
        )

    elif args.algo == 'dads':
        algo_kwargs.update(
            metra_mlp_rep=args.metra_mlp_rep,
            f_encoder=f_encoder,
            self_normalizing=args.self_normalizing,
            log_sum_exp=args.log_sum_exp,
            add_log_sum_exp_to_rewards=args.add_log_sum_exp_to_rewards,
            fixed_lam=args.fixed_lam,
            add_penalty_to_rewards=args.add_penalty_to_rewards,
            no_diff_in_rep=args.no_diff_in_rep,
            use_discrete_sac=args.use_discrete_sac,
            turn_off_dones=args.turn_off_dones,
            eval_goal_metrics=args.eval_goal_metrics,
            goal_range=args.goal_range,
            frame_stack=args.frame_stack,
            sample_new_z=args.sample_new_z,
            num_negative_z=args.num_negative_z,
            infonce_lam=args.infonce_lam,
            diayn_include_baseline=args.diayn_include_baseline,
            uniform_z=args.uniform_z,
        )

        skill_common_args.update(
            inner=args.inner,
            num_alt_samples=args.num_alt_samples,
            split_group=args.split_group,
            dual_reg=args.dual_reg,
            dual_slack=args.dual_slack,
            dual_dist=args.dual_dist,
        )

        algo = DADS(
            **algo_kwargs,
            **skill_common_args,
        )

    elif args.algo == 'ppo':
        algo = PPO(
            **algo_kwargs,
            vf=vf,
            gae_lambda=0.95,
            ppo_clip=0.2,
        )

    else:
        raise NotImplementedError

    if args.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)

    
    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
    )
    algo.option_policy.to(device)

    runner._train_args = TrainArgs(n_epochs=args.n_epochs,
                                   batch_size=args.traj_batch_size,
                                   plot=False,
                                   store_paths=False,
                                   pause_for_plot=False,
                                   start_epoch=0)

    return runner