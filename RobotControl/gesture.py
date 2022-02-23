from cb.env.GymVecEnv import ControlEnv as Environment
from controlBind import ControlEnv
import math
import time
import os
import argparse
from raisim_gym_lean.algo.ppo2 import PPO2
# from raisim_gym_s.algo.ppo2 import PPO2 as PPO_Sitting
from new_alg.new_policies import MlpPolicy
import argparse
import numpy as np
import torch
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Model_Torch import *
from SampleParser import TripletAllParser2
import time_checker
from utils import *

device = torch.device('cuda')
motion_lables = ('trot', 'manipulate', 'balance', 'lean', 'crabloco', 'jump')

parser = argparse.ArgumentParser()
modelAgile = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionTrot/2021-11-26-10-42-50_Iteration_13800.pkl')

current_dir = os.path.dirname(os.path.realpath(__file__))
dim_input = 96
dim_output = 16

save_file_loco = current_dir +'/rsc/map_gesture/gesture.pt'
contact_file_loco = current_dir +'/rsc/map_gesture/gesture_cont.pt'

mapper_loco = MLP_model_tanh(dim_input, dim_output)
contact_loco = MLP_model_tanh(dim_input, 4)

model_init(mapper_loco, save_file_loco)
model_init(contact_loco, contact_file_loco)

env = Environment(ControlEnv('t','a'))
env.initSim()
env.initVis()
env.initKinect()
obs = env.resetSim()
env.generateSitPD()
env.generateStandPD()

# reading_folder = '/home/sonic/Project/RobotControl/filmed_data/gesture/'
# env.readRecordedHumanParam(reading_folder + 'human_params02.txt')
# env.readRecordedHumanSkel(reading_folder + 'human_skel02.txt')

initFlag= True
# model_current = modelIdle
model_current = modelAgile
mapper_current = mapper_loco
contact_current = contact_loco

desired_motor_angle = np.array([0., deg2rad(49), deg2rad(-99)] * 4).reshape((1, -1))
resetter = 0

af_stand = actionFilterDefault(True)
af_ref_loco = actionFilterDefault(False)
af_ref_ori = actionFilterOrientation()
af_ref_loco_ori = actionFilterOrientation()
af_traj = actionFilterXYZ()
af_current = af_stand
i = 0
# Instead of using keyboard interrupt, hand tune the manual adjustment time
max_demo_step = 30 * 15
human_tensor = np.zeros((HUMAN_HISTORY, 32))
prev_tensor = np.zeros(((HUMAN_HISTORY-1), 32))

buffer = 15
change_flag = False

robot_motion = np.zeros((1, 19))
ref_motion = np.zeros((1, 19))
human_skeleton = np.zeros((1, 96))
human_params = np.zeros((1, 32))
cam_traj = np.zeros((1,3))

current_state = STAND
classified_state = STAND
reserved_state = STAND
reserved_sequence = NOOP

ready_walk = 0
z_recorded = 0

while True:
    temp = env.getHumanData().reshape((1,-1))
    # temp = env.readHumanData().reshape((1,-1))
    if i > buffer:
        if i == max_demo_step + buffer:
            break
        if env.kinectStop():
            break
        if initFlag:
            for i in range(HUMAN_HISTORY):
                human_tensor[i,:] = temp[0,-32:]
            prev_tensor = human_tensor[1:, :]
            initFlag =  False
        else:
            human_tensor[:-1, :] = prev_tensor
            human_tensor[-1,:] = temp[0, -32:]
            prev_tensor = human_tensor[1:, :]

        X = torch.from_numpy(temp).float().to(device)
        y = mapper_loco.forward(X)
        contact = contact_loco.forward(X).cpu().detach().numpy().flatten() 
        quat , q = simp_result_parser(y)
        locomotion = np.concatenate((quat.cpu().detach(), q.cpu().detach()), axis = 1)
        loco_filtered = af_ref_loco._FilterAction(locomotion[:,4:])
        euler = euler_from_quaternion(temp[0, 4:8])
        euler_filtered = af_ref_loco_ori._FilterAction(euler)
        euler_z = np.array([0, 0, euler_filtered[2]])
        quat_filtered = quaternion_from_euler(euler_filtered)
        locomotion_filtered = locomotion[0, :]
        # locomotion_filtered[:4] = np.array([1,0,0,0])
        locomotion_filtered[:4] = quaternion_from_euler(euler_z)
        locomotion_filtered[4:] = loco_filtered
        env.getWalkStep(contact, locomotion_filtered)

        action_raw, _ = model_current.predict(obs)
        action = af_current._FilterAction(action_raw)
        obs = env.stepSim(action)

        current_traj = env.getCurrentRobot()[:3]
        filtered_traj = af_traj._FilterAction(current_traj)
        cam_traj =np.concatenate((cam_traj, filtered_traj.reshape((1,-1))))
        robot_motion = np.concatenate((robot_motion, env.getCurrentRobot().reshape((1,-1))))
        ref_motion = np.concatenate((ref_motion, env.getCurrentReference().reshape((1,-1))))
        human_skeleton = np.concatenate((human_skeleton, env.getHumanSkeleton().reshape((1,-1))))
        human_params = np.concatenate((human_params, human_tensor[-1:,:]))
    i+=1
now = '/home/sonic/Project/RobotControl/filmed_data/' + time_checker.str_datetime()
if not os.path.isdir(now):
    os.mkdir(now)


np.savetxt( now + '/ref_motion.txt', ref_motion[1:,:],fmt='%1.3f')
np.savetxt( now + '/human_skel.txt', human_skeleton[1:,:],fmt='%1.3f')
np.savetxt( now + '/human_params.txt', human_params[1:,:],fmt='%1.3f')
np.savetxt( now + '/robot_motion.txt', robot_motion[1:,:],fmt='%1.3f')
np.savetxt( now + '/cam_traj.txt', cam_traj[1:,:],fmt='%1.3f')

print("done") 
env.endSim()