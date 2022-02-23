from cb.env.GymVecEnv import ControlEnv as Environment
from controlBind import ControlEnv
import time
import os
import argparse
from raisim_gym_lean.algo.ppo2 import PPO2
# from raisim_gym_s.algo.ppo2 import PPO2 as PPO_Sitting
from new_alg.new_policies import MlpPolicy
from new_alg import action_filter
import argparse
import numpy as np
import torch
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Model_Torch import *
from SampleParser import TripletAllParser2
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

weight_path = args.weight
print("Loaded weight from {}\n".format(weight_path))
model = PPO2.load(weight_path)

current_dir = os.path.dirname(os.path.realpath(__file__))
# save_file = current_dir +'/rsc/map_lean/tilt.pt'
# contact_file = current_dir +'/rsc/map_lean/lean_cont.pt'
save_file = current_dir +'/rsc/map_reach/man_fin.pt'
contact_file = current_dir +'/rsc/map_reach/man_cont.pt'
dim_input = 96
dim_output = 16
mapper_model = MLP_model_tanh(dim_input, dim_output)
mapper_model.load_state_dict(torch.load(save_file, map_location=device))
mapper_model.cuda()
mapper_model.eval()
contact_model = MLP_model_tanh(dim_input, 4)
model_init(contact_model, contact_file)
# contact_model.load_state_dict(torch.load(contact_file, map_location=device))
# contact_model.cuda()
# contact_model.eval()

env = Environment(ControlEnv('t','a'))
obs = env.reset()

env.startControl()
env.initVis()
env.initKinect()
action, _ = model.predict(obs)
current_motor_angle = action
desired_motor_angle = np.array([0., deg2rad(49), deg2rad(-99)] * 4).reshape((1, -1))
resetter = 0
af = actionFilterDefault(True)
af_ref = actionFilterDefault(False)
i = 0
# Instead of using keyboard interrupt, hand tune the manual adjustment time
max_demo_step = 30 * 20

while True:
    temp = env.getHumanData().reshape((1,-1))
    # print(temp[:, 64:])
    if env.isMoving() :
        # temp = env.getHumanData().reshape((1,-1))
        # npX = np.hstack((temp[:,0:4], temp[:,14:20]))
        X = torch.from_numpy(temp).float().to(device)
        y_result = mapper_model.forward(X)
        contact = contact_model.forward(X).cpu().detach().numpy().flatten()
        quat_t0 , q_t0  = simp_result_parser(y_result)
        kin_motion = np.concatenate((quat_t0.cpu().detach(), q_t0.cpu().detach()), axis = 1)
        filtered = af_ref._FilterAction(kin_motion[:,4:])
        kin_motion_filtered = kin_motion[0,:]
        kin_motion_filtered[4:] = filtered
        # env.getContactStep(np.array([0,0,0,0], dtype = np.float32), kin_motion[0,:])
        env.getContactStep(contact, kin_motion_filtered)
        # env.getContactStep(contact, kin_motion[0,:])
        i +=1
        action_raw, _ = model.predict(obs)
        action = af._FilterAction(action_raw)
        if i == max_demo_step:
            break
        if env.kinectStop():
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
# env.closeDevice()

env.end()
