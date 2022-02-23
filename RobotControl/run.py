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

modelTilt = PPO2.load("/home/sonic/Project/Alien_Gesture/data/Lean2021/2021-11-06-17-10-59_Iteration_17350.pkl")
modelMan = PPO2.load("/home/sonic/Project/Alien_Gesture/data/TiltMan/2022-01-04-10-35-30_Iteration_5900.pkl")
# modelMan = PPO2.load("/home/sonic/Project/Alien_Gesture/data/MappingReachingExp/2021-12-07-22-08-09_Iteration_17500.pkl")
# modelSit = PPO2.load('/home/sonic/Project/Alien_Gesture/data/SittingManipulation/2021-12-15-21-08-12_Iteration_8500.pkl')
modelSit = PPO2.load('/home/sonic/Project/Alien_Gesture/data/SittingManipulation/2022-01-23-11-29-04_Iteration_6500.pkl')
# modelLoco = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionSlowTran/2021-12-27-11-36-01_Iteration_8100.pkl')
modelLoco = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionSlowTran/2022-01-17-15-35-30_Iteration_6000.pkl')
modelAgile = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionTrot/2021-11-26-10-42-50_Iteration_13800.pkl')
modelSitTrans = PPO2.load('/home/sonic/Project/Alien_Gesture/data/SittingA1/2021-12-07-11-21-12_Iteration_1050.pkl')
# modelTiltMan = PPO2.load("/home/sonic/Project/Alien_Gesture/data/TiltMan/2022-01-08-20-27-10_Iteration_19000.pkl")
modelTiltMan = PPO2.load("/home/sonic/Project/Alien_Gesture/data/TiltMan/2022-01-21-12-41-19_Iteration_8200.pkl")
modelWalk = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionWalk/2022-01-18-11-06-53_Iteration_850.pkl')
modelIdle = model_null()

modelList = [modelTiltMan, modelSit, modelAgile, modelSitTrans]

current_dir = os.path.dirname(os.path.realpath(__file__))
dim_input = 96
dim_output = 16

# save_file_tilt = current_dir +'/rsc/map_lean/tilt.pt'
save_file_tilt = current_dir +'/rsc/map_lean/tilt2022.pt'
# contact_file_tilt = current_dir +'/rsc/map_lean/tiltCont.pt'
contact_file_tilt = current_dir +'/rsc/map_lean/tiltcont_old.pt'
# save_file_man = current_dir +'/rsc/map_reach/man_fin.pt'
# save_file_man = current_dir +'/rsc/map_reach/man2022.pt'
save_file_man = current_dir +'/rsc/map_reach/man2022.pt'
contact_file_man = current_dir +'/rsc/map_reach/man2022cont.pt'
save_file_loco = current_dir +'/rsc/map_loco/trotagil.pt'
contact_file_loco = current_dir +'/rsc/map_loco/trotagilcont.pt'
save_file_walk = current_dir +'/rsc/map_walk/walk1.pt'
contact_file_walk = current_dir +'/rsc/map_walk/walk1_cont.pt'
save_file_sit = current_dir +'/rsc/map_sit/sit_man1.pt'

mapper_loco = MLP_model_tanh(dim_input, dim_output)
mapper_tilt = MLP_model_tanh(dim_input, dim_output)
mapper_man = MLP_model_tanh(dim_input, dim_output)
mapper_walk = MLP_model_tanh(dim_input, dim_output)
mapper_sit = MLP_model_tanh(dim_input, dim_output)
contact_loco = MLP_model_tanh(dim_input, 4)
contact_tilt = MLP_model_tanh(dim_input, 4)
contact_man = MLP_model_tanh(dim_input, 4)
contact_walk = MLP_model_tanh(dim_input, 4)

model_init(mapper_loco, save_file_loco)
model_init(mapper_tilt, save_file_tilt)
model_init(mapper_man, save_file_man)
model_init(mapper_walk, save_file_walk)
model_init(mapper_sit, save_file_sit)
model_init(contact_loco, contact_file_loco)
model_init(contact_tilt, contact_file_tilt)
model_init(contact_man, contact_file_man)
model_init(contact_walk, contact_file_walk)

env = Environment(ControlEnv('t','a'))
env.startControl()
env.initVis()
env.initKinect()
obs = env.reset()
env.generateSitPD()
env.generateStandPD()
# reading_folder = '/home/sonic/Project/RobotControl/filmed_data/LocoSteer2/'
# env.readRecordedHumanParam(reading_folder + 'human_params.txt')
# env.readRecordedHumanSkel(reading_folder + 'human_skel.txt')

initFlag= True
# model_current = modelIdle
model_current = modelTiltMan
mapper_current = mapper_tilt
contact_current = contact_tilt

desired_motor_angle = np.array([0., deg2rad(49), deg2rad(-99)] * 4).reshape((1, -1))
resetter = 0

af_stand = actionFilterDefault(True)
af_sit = actionFilterSit()
af_ref_man = actionFilterDefault(False)
af_ref_tilt = actionFilterDefault(False)
af_ref_loco = actionFilterDefault(False)
af_ref_sit = actionFilterSit()
af_ref_ori = actionFilterOrientation()
af_ref_loco_ori = actionFilterOrientation()
af_ref_sit_ori = actionFilterOrientation()
af_traj = actionFilterXYZ()
af_current = af_stand
i = 0
# Instead of using keyboard interrupt, hand tune the manual adjustment time
human_tensor = np.zeros((HUMAN_HISTORY, 32))
prev_tensor = np.zeros(((HUMAN_HISTORY-1), 32))

change_flag = False

robot_motion = np.zeros((1, 19))

current_state = STAND
classified_state = STAND
reserved_state = STAND
reserved_sequence = NOOP

ready_walk = 0
# max_demo_step = 1050
max_demo_step = 450

action = np.array([0, -0.27334326246, 0.08776541961] * 4,dtype=np.float32).reshape((1,-1))
current_motor_angle = np.array([0., deg2rad(49), deg2rad(-99)] * 4).reshape((1, -1))
desired_motor_angle = np.array([0., deg2rad(49), deg2rad(-99)] * 4).reshape((1, -1))

human_skeleton = np.zeros((1, 96))
human_params = np.zeros((1, 32))

try:
    while True:
        TC = time_checker.TimeChecker()
        TC.begin()
        # temp = env.readHumanData().reshape((1,-1))
        temp = env.getHumanData().reshape((1,-1))
        if env.isMoving() :
            TC = time_checker.TimeChecker()
            TC.begin()
            if i == max_demo_step :
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
            if current_state != PROCESSING:
                classified_state = classifyState(current_state, human_tensor)
            
            if current_state != classified_state and current_state != PROCESSING:
                if current_state == STAND and classified_state == SIT:
                    reserved_sequence = STA2SIT
                elif current_state == SIT and classified_state == STAND:
                    reserved_sequence = SIT2STA
                elif current_state == STAND and classified_state == WALK:
                    reserved_sequence = STA2WLK
                current_state = PROCESSING
                reserved_state = classified_state

            if current_state == PROCESSING:
                # (1) If it is in neutral state, do transition
                if env.isNeutral():
                    if reserved_sequence == NOOP:
                        current_state = reserved_state
                        model_current = modelList[current_state]
                        env.setState(reserved_state)
                        env.setDynamic()
                    elif reserved_sequence == STA2SIT:
                        # Here, do the reserved action
                        model_current = modelSitTrans
                        seq_done = env.doSit()
                        if seq_done:
                            af_sit._ResetActionFilter()
                            af_current = af_sit
                            reserved_sequence = NOOP
                    elif reserved_sequence == SIT2STA:
                        model_current = modelSitTrans
                        seq_done = env.doStand()
                        if seq_done:
                            af_stand._ResetActionFilter()
                            af_current = af_stand
                            reserved_sequence = NOOP
                    elif reserved_sequence == STA2WLK:
                        ready_walk +=1
                        if ready_walk == 10:
                            af_stand._ResetActionFilter()
                            af_current = af_stand
                            ready_walk = 0
                            reserved_sequence = NOOP
                else:
                    env.setNeutral()

            elif current_state == STAND: 
                X = torch.from_numpy(temp).float().to(device)
                # Here we want a classifier
                y_tilt = mapper_tilt.forward(X)
                y_man = mapper_man.forward(X)
                contact = contact_man.forward(X).cpu().detach().numpy().flatten()
                quat_tilt , q_tilt  = simp_result_parser(y_tilt)
                quat_man , q_man  = simp_result_parser(y_man)
                tilt_motion = np.concatenate((quat_tilt.cpu().detach(), q_tilt.cpu().detach()), axis = 1)
                man_motion = np.concatenate((quat_man.cpu().detach(), q_man.cpu().detach()), axis = 1)
                tilt_filtered = af_ref_tilt._FilterAction(tilt_motion[:,4:])
                man_filtered = af_ref_man._FilterAction(man_motion[:,4:])
                euler = euler_from_quaternion(tilt_motion[0, :4])
                euler_filtered = af_ref_ori._FilterAction(euler)
                quat_filtered = quaternion_from_euler(euler_filtered)
                tilt_motion_filtered = tilt_motion[0,:]
                tilt_motion_filtered[:4] = quat_filtered
                # tilt_motion_filtered[:4] = np.array([1,0,0,0])
                tilt_motion_filtered[4:] = tilt_filtered
                # tilt_motion_filtered[4:] =  np.array([0., 0.855, -1.728] * 4)
                man_motion_filtered = man_motion[0,:]
                man_motion_filtered[4:] = man_filtered

                # contact = np.array([0,0,0,0], dtype= np.float32)

                env.getStandStep(contact, tilt_motion_filtered, man_motion_filtered)

            elif current_state == WALK:
                X = torch.from_numpy(temp).float().to(device)
                y = mapper_loco.forward(X)
                contact = contact_loco.forward(X).cpu().detach().numpy().flatten() 
                # y = mapper_walk.forward(X)
                # contact = contact_walk.forward(X).cpu().detach().numpy().flatten()
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
                # print(locomotion_filtered)
                env.getWalkStep(contact, locomotion_filtered)
                # env.getWalkStep(contact, locomotion[0,:])

            elif current_state == SIT:
                X = torch.from_numpy(temp).float().to(device)
                y = mapper_sit.forward(X)
                quat , q = simp_result_parser(y)
                sit = np.concatenate((quat.cpu().detach(), q.cpu().detach()), axis = 1)
                sit_ang_filtered = af_ref_sit._FilterAction(sit[:,4:])
                sit_filtered = sit[0, :]
                sit_filtered[4:] = sit_ang_filtered
                env.getSitStep(sit_filtered)

            action_raw, _ = model_current.predict(obs)
            action = af_current._FilterAction(action_raw)
            delay = int(TC.get_time() * 1E6)
            env.check_delay(delay)
        obs = env.step(action)
        human_skeleton = np.concatenate((human_skeleton, env.getHumanSkeleton().reshape((1,-1))))
        human_params = np.concatenate((human_params, human_tensor[-1:,:]))
        current_motor_angle = obs[:,152:164]
        i+=1
except KeyboardInterrupt:
    print("early termination")

print("done") 
now = '/home/sonic/Project/RobotControl/filmed_data/' + time_checker.str_datetime()
if not os.path.isdir(now):
    os.mkdir(now)

np.savetxt( now + '/human_skel.txt', human_skeleton[1:,:],fmt='%1.3f')
np.savetxt( now + '/human_params.txt', human_params[1:,:],fmt='%1.3f')

while resetter < 200:
    blend_ratio = np.minimum(resetter / 100., 1)
    action_np = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle
    action = action_np.astype(np.float32)
    resetter +=1
    env.stepTest(action)
# env.closeDevice()

env.end()
