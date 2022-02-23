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
modelSit = PPO2.load('/home/sonic/Project/Alien_Gesture/data/SittingManipulation/2021-12-15-21-08-12_Iteration_8500.pkl')
# modelLoco = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionSlowTran/2021-12-27-11-36-01_Iteration_8100.pkl')
modelLoco = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionSlowTran/2022-01-20-11-15-15_Iteration_6150.pkl')
modelAgile = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionTrot/2021-11-26-10-42-50_Iteration_13800.pkl')
modelSitTrans = PPO2.load('/home/sonic/Project/Alien_Gesture/data/SittingA1/2021-12-07-11-21-12_Iteration_1050.pkl')
modelTiltMan = PPO2.load("/home/sonic/Project/Alien_Gesture/data/TiltMan/2022-01-08-20-27-10_Iteration_19000.pkl")
modelWalk = PPO2.load('/home/sonic/Project/Alien_Gesture/data/MappingLocomotionWalk/2022-01-18-11-06-53_Iteration_850.pkl')

modelIdle = model_null()


modelList = [modelTiltMan, modelSit, modelAgile, modelSitTrans]

current_dir = os.path.dirname(os.path.realpath(__file__))
dim_input = 96
dim_output = 16

# save_file_classifier = current_dir +'/rsc/class_current.pt'
# classifier = MLP_model_simple(32 * HUMAN_HISTORY, 6)
# model_init(classifier, save_file_classifier)

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
# save_file_loco = current_dir +'/rsc/map_loco/locotrot.pt'
# contact_file_loco = current_dir +'/rsc/map_loco/locotrotcont.pt'
save_file_walk = current_dir +'/rsc/map_walk/walk1.pt'
contact_file_walk = current_dir +'/rsc/map_walk/walk1_cont.pt'
save_file_sit = current_dir +'/rsc/map_sit/sit_man1.pt'
save_file_loco_param = current_dir +'/rsc/map_loco_param/param_loco.pt'
save_file_loco_param_del = current_dir +'/rsc/map_loco_param/param_loco_del.pt'

mapper_loco = MLP_model_tanh(dim_input, dim_output)
mapper_tilt = MLP_model_tanh(dim_input, dim_output)
mapper_man = MLP_model_tanh(dim_input, dim_output)
mapper_walk = MLP_model_tanh(dim_input, dim_output)
mapper_sit = MLP_model_tanh(dim_input, dim_output)
mapper_loco_param = MLP_model_tanh(dim_input, 4)
mapper_loco_param_del = MLP_model_tanh(dim_input, 4)
contact_loco = MLP_model_tanh(dim_input, 4)
contact_tilt = MLP_model_tanh(dim_input, 4)
contact_man = MLP_model_tanh(dim_input, 4)
contact_walk = MLP_model_tanh(dim_input, 4)

model_init(mapper_loco, save_file_loco)
model_init(mapper_tilt, save_file_tilt)
model_init(mapper_man, save_file_man)
model_init(mapper_walk, save_file_walk)
model_init(mapper_sit, save_file_sit)
model_init(mapper_loco_param, save_file_loco_param)
model_init(mapper_loco_param_del, save_file_loco_param_del)
model_init(contact_loco, contact_file_loco)
model_init(contact_tilt, contact_file_tilt)
model_init(contact_man, contact_file_man)
model_init(contact_walk, contact_file_walk)

env = Environment(ControlEnv('t','a'))
env.initSim()
env.initVis()
env.initKinect()
obs = env.resetSim()
env.generateSitPD()
env.generateStandPD()

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
max_demo_step = 30 * 45
human_tensor = np.zeros((HUMAN_HISTORY, 32))
prev_tensor = np.zeros(((HUMAN_HISTORY-1), 32))

buffer = 15
change_flag = False

robot_motion = np.zeros((1, 19))
ref_motion = np.zeros((1, 19))
human_skeleton = np.zeros((1, 96))
human_params = np.zeros((1, 32))
cam_traj = np.zeros((1,3))

# env.readSeq()
# while True:
#     TC = time_checker.TimeChecker()
#     TC.begin()
#     action_raw, _ = model_current.predict(obs)
#     action = af._FilterAction(action_raw)
#     print(action_raw, action)
#     if i == max_demo_step - 5:
#         break
#     TC.print_time()
#     obs = env.stepSim2(action)
#     i +=1

current_state = STAND
classified_state = STAND
reserved_state = STAND
reserved_sequence = NOOP

ready_walk = 0
z_recorded = 0

while True:
    # TC = time_checker.TimeChecker()
    # TC.begin()
    temp = env.getHumanData().reshape((1,-1))
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
            env.getStandStep(contact, tilt_motion_filtered, man_motion_filtered)

        elif current_state == WALK:
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

            # y = mapper_loco_param.forward(X)
            # params = result_parser_loco(y).cpu().detach().numpy().flatten()
            # euler = euler_from_quaternion(temp[0, 4:8])
            # euler_filtered = af_ref_loco_ori._FilterAction(euler)
            # euler_z = np.array([0, 0, euler_filtered[2]], dtype = np.float32)
            # quat_z = np.array(quaternion_from_euler(euler_z), dtype = np.float32)
            # # print(params)
            # env.getParamLoco(quat_z, params)

        elif current_state == SIT:
            X = torch.from_numpy(temp).float().to(device)
            y = mapper_sit.forward(X)
            quat , q = simp_result_parser(y)
            sit = np.concatenate((quat.cpu().detach(), q.cpu().detach()), axis = 1)
            sit_ang_filtered = af_ref_sit._FilterAction(sit[:,4:])
            # euler = euler_from_quaternion(sit_ang_filtered[0, :4])
            # euler_filtered = af_ref_sit_ori._FilterAction(euler)
            # quat_filtered = quaternion_from_euler(euler_filtered)
            
            sit_filtered = sit[0, :]
            # sit_filtered[:4] = quat_filtered
            sit_filtered[4:] = sit_ang_filtered
            env.getSitStep(sit_filtered)

        # obs = env.updateSimObs()
        action_raw, _ = model_current.predict(obs)
        action = af_current._FilterAction(action_raw)
        obs = env.stepSim(action)
        # env.stepSim(action)
        # if current_state == SIT:
        #     print(obs)

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

# while True:
#     temp = env.getHumanData().reshape((1,-1))
#     if i > buffer:
#         if initFlag:
#             for i in range(HUMAN_HISTORY):
#                 human_tensor[:,32 * i: 32 * (i+1)] = temp[:,-32:]
#             prev_tensor = human_tensor
#             initFlag =  False
#         else:
#             human_tensor[:, :-32] = prev_tensor[:, 32:]
#             human_tensor[:,-32:] = temp[:, -32:]
#             prev_tensor = human_tensor
        
#         X_human = torch.from_numpy(human_tensor).float().to(device)
#         y_class = classifier.forward(X_human)
#         _, predicted = torch.max(y_class.data, 1)

#         current_ID = env.modeID(predicted)
#         # current_ID = predicted
#         print(motion_lables[predicted], " and current ", motion_lables[current_ID])
#         # if current_ID - predicted != 0:
#         #     change_flag = True
        
#         # if (current_ID - predicted == 0) and change_flag:
#         #     print("changed")
#         #     obs = env.resetSim()
#         #     change_flag = False

#         if current_ID == 0:
#             model_current = modelTilt
#             mapper_current = mapper_tilt
#             contact_current = contact_tilt
#         elif current_ID == 1: 
#             model_current = modelMan
#             mapper_current = mapper_man
#             contact_current = contact_man
#         elif current_ID == 3: 
#             model_current = modelTilt
#             mapper_current = mapper_tilt
#             contact_current = contact_tilt
#         else:
#             model_current = modelTilt
#             mapper_current = mapper_tilt
#             contact_current = contact_tilt

#         # Here, implement with if-else
#         # Do tilting mapper when it is not clear

#         X = torch.from_numpy(temp).float().to(device)
#         # Here we want a classifier
#         y_result = mapper_current.forward(X)
#         contact = contact_current.forward(X).cpu().detach().numpy().flatten()
#         quat_t0 , q_t0  = simp_result_parser(y_result)
#         kin_motion = np.concatenate((quat_t0.cpu().detach(), q_t0.cpu().detach()), axis = 1)
#         filtered = af_ref._FilterAction(kin_motion[:,4:])
#         kin_motion_filtered = kin_motion[0,:]
#         kin_motion_filtered[4:] = filtered
#         env.getContactStep(contact, kin_motion_filtered)
#         action_raw, _ = model_current.predict(obs)
#         action = af._FilterAction(action_raw)
#         if i == max_demo_step:
#             break
#         if env.kinectStop():
#             break
#     i +=1
#     obs = env.stepSim(action)



        # How to handle this? 
        # if np.all(contact < 0) and np.linalg.norm(euler_filtered) < 0.05: 
        #     print("idle")
        #     model_current = modelIdle
        # elif np.linalg.norm(euler_filtered) < 0.05:
        #     print("manipulate")
        #     model_current = modelMan
        #     tilt_motion_filtered[:4] = np.array([1,0,0,0])
        #     tilt_motion_filtered[4:] =  np.array([0., 0.855, -1.728] * 4)
        # elif np.all(contact < 0) :
        #     print("tilt")
        #     model_current = modelTilt
        # else:
        #     print("tiltman")
        #     model_current = modelTiltMan