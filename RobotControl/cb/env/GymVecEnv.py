import numpy as np
from gym import spaces
from stable_baselines.common.vec_env import VecEnv


class ControlEnv(VecEnv):
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
        self.rewards = [[] for _ in range(self.num_envs)]
        
    @property
    def num_envs(self):
        return 1

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def reset(self):
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    def resetSim(self):
        self.wrapper.resetSim(self._observation)
        return self._observation.copy()

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

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

    def initKinect(self, ):
        self.wrapper.kinectInit()

    def check_delay(self, delay):
        self.wrapper.check_delay(delay)

    def initVis(self, ):
        self.wrapper.visInit()

    def setState(self, reserved_state):
        self.wrapper.setState(reserved_state)

    def setNeutral(self, ):
        self.wrapper.setNeutral()

    def isNeutral(self, ):
        return self.wrapper.isNeutral()

    def setDynamic(self, ):
        self.wrapper.setDynamic()

    def initSim(self, ):
        self.wrapper.simInit()

    def readSeq(self, ):
        self.wrapper.readSeq()

    def startPDControl(self, ):
        self.wrapper.startPD()

    def startControl(self, ):
        self.wrapper.start()

    def control(self, frame):
        self.wrapper.control(frame)

    def getHumanData(self, ):
        return self.wrapper.getHumanData()

    def getHumanSkeleton(self, ):
        return self.wrapper.getHumanSkeleton()

    def getCurrentRobot(self, ):
        return self.wrapper.getCurrentRobot()

    def getCurrentReference(self, ):
        return self.wrapper.getCurrentReference()

    def kinectStop(self, ):
        return self.wrapper.kinectStop()

    def closeDevise(self, ):
        return self.wrapper.kinectStop()

    def modeID(self, mode):
        return self.wrapper.modeID(mode)
        
    def step(self, action):
        self.wrapper.step(action, self._observation)
        return self._observation.copy()

    def stepSim(self, action):
        self.wrapper.stepSim(action, self._observation)
        return self._observation.copy()

    def stepSim2(self, action):
        self.wrapper.stepSim2(action, self._observation)
        return self._observation.copy()

    def updateSimObs(self, ):
        self.wrapper.updateSimObs(self._observation)
        return self._observation.copy()

    def stepTest(self, action):
        self.wrapper.stepTest(action)

    def isMoving(self, ):
        return self.wrapper.isMoving()

    def doSit(self, ):
        return self.wrapper.doSit()

    def doStand(self, ):
        return self.wrapper.doStand()

    def readRecordedHumanParam(self, file):
        self.wrapper.readRecordedHumanParam(file)

    def readRecordedHumanSkel(self, file):
        self.wrapper.readRecordedHumanSkel(file)

    def readHumanData(self, ):
        return self.wrapper.readHumanData()

    def getContactStep(self, contact, pose):
        self.wrapper.getContactStep(contact, pose)

    def getStandStep(self, contact, tilt, man):
        self.wrapper.getStandStep(contact, tilt, man)

    def getWalkStep(self, contact, pose):
        self.wrapper.getWalkStep(contact, pose)

    def getParamLocoDel(self, ori, param):
        self.wrapper.getParamLocoDel(ori, param)

    def getParamLoco(self, ori, param):
        self.wrapper.getParamLoco(ori, param)

    def getSitStep(self, pose):
        self.wrapper.getSitStep( pose)

    def generateRandomSeqLoco(self, ):
        self.wrapper.generateRandomSeqLoco()

    def generateRandomSeqMani(self, ):
        self.wrapper.generateRandomSeqMani()

    def generateRandomSeqSit(self, ):
        self.wrapper.generateRandomSeqSit()
    
    def generateSitPD(self, ):
        self.wrapper.generateSitPD()

    def generateStandPD(self, ):
        self.wrapper.generateStandPD()

    def updateRecorded(self,):
        self.wrapper.updateRecorded()

    def endSim(self, ):
        self.wrapper.endSim()

    def end(self, ):
        self.wrapper.end()
