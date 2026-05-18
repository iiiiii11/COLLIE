import numpy as np

from .oracle_pref import OracleStatePref


class QueryTrajPref:
    def __init__(self, env_name, pref_task):
        self.env_name = env_name
        self.pref_task = pref_task
        self.state_pref_model = OracleStatePref(env_name, pref_task)

    def get_traj_pref(self, traj):

        states_pref = self.state_pref_model.get_state_pref([traj]) - 1

        min_pref = np.min(states_pref)
        if min_pref < 0:
            return -1
        states_pref_1 = np.maximum(states_pref, 0)
        if np.min(states_pref_1) > 0:
            return 1
        return 0

    def get_trajs_pref(self, trajs):

        if isinstance(trajs, list) or isinstance(trajs, tuple):
            traj_prefs = [self.get_traj_pref(traj) for traj in trajs]
            traj_prefs = np.array(traj_prefs)
        elif isinstance(trajs, np.ndarray):

            traj_prefs = [self.get_traj_pref(trajs[i]) for i in range(trajs.shape[0])]
            traj_prefs = np.array(traj_prefs)

        return traj_prefs
