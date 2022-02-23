from cb.env.GymVecEnv import ControlEnv as Environment
from controlBind import ControlEnv
import time
import os
import argparse
from raisim_gym_lean.algo.ppo2 import PPO2
from new_alg.new_policies import MlpPolicy
from new_alg import action_filter
import argparse
import numpy as np
import torch
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Model_Torch import *
from SampleParser import TripletAllParser2

def deg2rad(deg):
  return deg * np.pi /180

ACTION_FILTER_HIGH_CUT = 5.0
ACTION_FILTER_HIGH_CUT_HW = 3.0

class actionFilterDefault(object): 
    def __init__(self, raw):
        self._action_filter = self._BuildActionFilter()
        if raw :
            self.init_angle = np.array([0, -0.27334326246, 0.08776541961] * 4)
        else:
            self.init_angle = np.array([0., 0.855, -1.728] * 4)
        self._ResetActionFilter()

    def _BuildActionFilter(self):
        sampling_rate = 30
        num_joints = 12
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()
        self._action_filter.init_history(self.init_angle)

    def _FilterAction(self, action):
        filtered_action = np.zeros([1, 12], dtype=np.float32)            
        filtered_action[0,:] = self._action_filter.filter(action[0,:])
        return filtered_action


def simp_result_parser(result):
    epsilon = 1e-6
    ori0_mtx = torch.from_numpy(np.block([[np.identity(4)] ,[np.zeros((12,4))]])).float().cuda()
    pose0_mtx = torch.from_numpy(np.block([[np.zeros((4,12))], [np.identity(12)]])).float().cuda()

    t_scaler =  torch.from_numpy(np.array([0.8030, 2.6180, 0.8905] * 4)).float()
    t_shifter =  torch.from_numpy(np.array([0.0000,  0.7160, -0.0785] * 4)).float()
    real = np.array([ 0., 0.855, -1.728] * 4)
    ori_0 = torch.matmul(result, ori0_mtx)
    norm_quat0 = torch.sqrt(torch.matmul(torch.pow(ori_0 ,2), torch.ones(4,1).cuda()))
    quat_result0 = ori_0 / (norm_quat0 + epsilon) 

    joint_result0 = torch.matmul(result, pose0_mtx)
    scaled_residual0 = torch.add(torch.mul(t_scaler.cuda(), joint_result0), t_shifter.cuda())
    scaled_angle0 = scaled_residual0 + torch.from_numpy(real).float().cuda()

    return quat_result0, scaled_angle0

parser = argparse.ArgumentParser()

parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

weight_path = args.weight
print("Loaded weight from {}\n".format(weight_path))
model_sit = PPO2.load('/home/sonic/Project/Alien_Gesture/data/SittingA1/2021-12-07-11-21-12_Iteration_1050.pkl')
model = PPO2.load(weight_path)

current_dir = os.path.dirname(os.path.realpath(__file__))
save_file = current_dir +'/rsc/map_reach/man_fin.pt'
contact_file = current_dir +'/rsc/map_reach/man_cont.pt'
dim_input = 96
dim_output = 16
mapper_model = MLP_model_tanh(dim_input, dim_output)
device = torch.device('cuda')
mapper_model.load_state_dict(torch.load(save_file, map_location=device))
mapper_model.cuda()
mapper_model.eval()
contact_model = MLP_model_tanh(dim_input, 4)
contact_model.load_state_dict(torch.load(contact_file, map_location=device))
contact_model.cuda()
contact_model.eval()
human_file = current_dir +'/rsc/map_reach/params_reacht.txt'
X = TripletAllParser2(human_file).tensor_coord.to(device)
y_result = mapper_model.forward(X)
quat_t0 , q_t0  = simp_result_parser(y_result)
contact = contact_model.forward(X).cpu().detach().numpy()
kin_motion = np.concatenate((quat_t0.cpu().detach(), q_t0.cpu().detach()), axis = 1)

resetter = 0
i = 0 
af = actionFilterDefault(True)
af_ref = actionFilterDefault(False)
max_demo_step = 30 * 20


env = Environment(ControlEnv('t','a'))
env.generateRandomSeqSit()
obs = env.reset()
env.startControl()
action, _ = model_sit.predict(obs)
# action, _ = model.predict(obs)
current_motor_angle = action
desired_motor_angle = np.array([0., deg2rad(49), deg2rad(-99)] * 4).reshape((1, -1))

env.initSim()
env.initVis()
env.generateSitPD()
env.generateStandPD()
env.generateNeutralSit()

signal = 0
while True:
    if env.isMoving() :
        if signal == 0:
            seq_done = env.doSit()
            if seq_done:
                signal = 1
        elif signal == 1:
            seq_done = env.doStand()
            if seq_done:
                signal = 2
    if signal == 2:
        break
    obs = env.step(action)
    current_motor_angle = obs[:,152:164]


print("done")
while resetter < 200:
    blend_ratio = np.minimum(resetter / 100., 1)
    action_np = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle
    action = action_np.astype(np.float32)
    resetter +=1
    env.stepTest(action)

env.end()
