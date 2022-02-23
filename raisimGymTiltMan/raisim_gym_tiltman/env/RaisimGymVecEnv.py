import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv


class RaisimGymVecEnv(VecEnv):

    def __init__(self, impl):
        self.wrapper = impl
        self.wrapper.init() 
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf,
                                             dtype=np.float32)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1., dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs, len(self._extraInfoNames)], dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]

    def seed(self, seed=None):
        self.wrapper.seed(seed)

    def step(self, action, visualize=False):
        if not visualize:
            self.wrapper.step(action, self._observation, self._reward, self._done, self._extraInfo)
        else:
            self.wrapper.testStep(action, self._observation, self._reward, self._done, self._extraInfo)

        if len(self._extraInfoNames) is not 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if int(self._reward[i]) == -1E10:
                self._reward[i] = 0
                print("done")
                self._done[i] = True

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), self._done.copy(), info.copy()

    def stepRef(self,):
        self.wrapper.stepRef(self._reward)
        for i in range(self.num_envs):
            self._done[i] = False
            self.rewards[i].append(self._reward[i])
            if int(self._reward[i]) == -1E10:
                self._reward[i] = 0
                print("done")
                self._done[i] = True

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                self.rewards[i].clear()

        return self._reward.copy(), self._done.copy()

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def start_recording_video(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_recording_video(self):
        self.wrapper.stopRecordingVideo()

    def curriculum_callback(self, update):
        self.wrapper.curriculumUpdate(update)

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    def show_window(self):
        self.wrapper.showWindow()

    def hide_window(self):
        self.wrapper.hideWindow()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    # Define new functions to connect between Raisim Env and Python
    def set_alien_pose(self, pose):
        self.wrapper.set_alien_pose(pose)

    def get_alien_pose(self, robot_key_joint):
        current = np.zeros(3, dtype=np.float32)
        self.wrapper.get_alien_pose(robot_key_joint, current)
        return current

    def get_robot_key(self,):
        keys = self.wrapper.get_robot_key()
        return keys

    def set_bvh_time(self, time):
        self.wrapper.set_bvh_time(time)
    
    def get_animal_key(self, ):
        keys = self.wrapper.get_animal_key()
        return keys

    def get_target_pose(self, anymal_key_joint):
        current = np.zeros(3, dtype=np.float32)
        self.wrapper.get_target_pose(anymal_key_joint, current)
        return current

    def get_current_q(self, ):
        current = np.zeros(18, dtype=np.float32)
        self.wrapper.get_current_q(current)
        return current

    def get_jacobian(self, joint_name):
        jac = self.wrapper.get_jacobian(joint_name)
        return jac

    def getNumFrames(self,):
        return self.wrapper.getNumFrames();

    def kinect_init(self, ):
        self.wrapper.kinect_init()

    def getTorque(self, ):
        tau = np.zeros(12, dtype=np.float32)
        self.wrapper.get_current_torque(tau)
        return tau

    def stepTest(self, action):
        self.wrapper.stepTest(action, self._observation, self._reward, self._done, self._extraInfo)

        if len(self._extraInfoNames) is not 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if int(self._reward[i]) == -1E10:
                self._reward[i] = 0
                print("done")
                self._done[i] = True

            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        return self._observation.copy(), self._reward.copy(), self._done.copy(), info.copy()

    def getContactStep(self, contact, pose):
        self.wrapper.getContactStep(contact, pose)