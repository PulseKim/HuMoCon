from ruamel.yaml import YAML, dump, RoundTripDumper

from raisim_gym_push_exp.env.bin import env
from raisim_gym_push_exp.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisim_gym_push_exp.helper.raisim_gym_helper import ConfigurationSaver

from raisim_gym_lean.algo.ppo2 import PPO2
from raisim_gym_lean.archi.policies import MlpPolicy
from raisim_gym_lean.algo.action_filter import action_filter

import os
import math
import argparse
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import time_checker

import torch
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from Model_Torch import *
from SampleParser import TripletAllParser
from SampleParser import TripletAllParser2


class Plot(object):
    def __init__(self, joint_keys, file_path):
        self.iteration = []
        self.pd_total = []
        self.pd_gain = {}
        self.file_path = file_path
        for key in joint_keys:
            self.pd_gain[key] = []

    def leg_keys(self, FR, FL, RR, RL):
        self.FR_keys = FR
        self.FL_keys = FL
        self.RR_keys = RR
        self.RL_keys = RL

    def add(self,iteration,pd,pd_total):
        self.iteration.append(iteration)
        for index, key in enumerate(self.pd_gain.keys()):
            self.pd_gain[key].append(pd[index])
        self.pd_total.append(pd_total)

    def draw(self):
        plt.ion()
        plt.figure(1,figsize=(12,12))
        plt.clf()
        plt.title('torque graph')
        ax1 = plt.subplot(5,1,1)
        ax1.set_title("total")
        ax1.plot(self.iteration,self.pd_total,c="r")
        ax2 = plt.subplot(5,1,2)
        ax2.set_title("FR")
        for key in self.FR_keys:
            ax2.plot(self.iteration,self.pd_gain[key],label=str(key))
        ax2.legend(loc=2)
        ax3 = plt.subplot(5,1,3)
        ax3.set_title("FL")
        for key in self.FL_keys:
            ax3.plot(self.iteration,self.pd_gain[key],label=str(key))
        ax3.legend(loc=2)
        ax4 = plt.subplot(5,1,4)
        ax4.set_title("RR")
        for key in self.RR_keys:
            ax4.plot(self.iteration,self.pd_gain[key],label=str(key))
        ax4.legend(loc=2)
        ax5 = plt.subplot(5,1,5)
        ax5.set_title("RL")
        for key in self.RL_keys:
            ax5.plot(self.iteration,self.pd_gain[key],label=str(key))
        ax5.legend(loc=2)
        plt.show()
        plt.pause(4.0)
        # file_path = os.path.dirname(os.getcwd())+"/result/"
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)
        plt.savefig(self.file_path+"/torque_"+time_checker.str_datetime() +".png")


def TensorboardLauncher_local(directory_path):
    from tensorboard import program
    import webbrowser
    # learning visualizer
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', directory_path, '--host', '0.0.0.0'])
    url = tb.launch()
    print("[RAISIM_GYM] Tensorboard session created: "+url)
    webbrowser.open_new(url)
    
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


# configuration
current_dir = os.path.dirname(os.path.realpath(__file__))
rscdir = current_dir+ "/env/rsc"
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default=os.path.abspath(rscdir + "/changed_cfg.yaml"),
                    help='configuration file')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode

# cfg_abs_path = parser.parse_args().cfg
# print(cfg_abs_path)
cfg_abs_path = "/home/sonic/Project/Alien_Gesture/env/A1_Reaching_Expanded/env/cfg.yaml"
cfg = YAML().load(open(cfg_abs_path, 'r'))

# save the configuration and other files
rsg_root = os.path.dirname(os.path.abspath(__file__))+ '/../Alien_Gesture'
log_dir = rsg_root + '/data'
# create environment from the configuration file
if mode == "test":
    cfg['environment']['num_envs'] = 1

if mode == "ref":
    cfg['environment']['num_envs'] = 1

if mode == "gen":
    cfg['environment']['num_envs'] = 1

task_path = os.path.dirname(os.path.realpath(__file__))
rsc_path = "/home/sonic/Project/Alien_Gesture/env/robots"
env = VecEnv(env.RaisimGymEnv(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper)))
# env = Environment(RaisimGymEnv(current_dir+"/env/urdf_aliengo", dump(cfg['environment'], Dumper=RoundTripDumper)))
# Get algorithm
if mode == 'train':
    saver = ConfigurationSaver(log_dir=log_dir+'/MappingReachingExp',
                               save_items=[rsg_root+'/env/A1_Reaching_Expanded/env/Environment.hpp', cfg_abs_path])
    model = PPO2(
        tensorboard_log=saver.data_dir,
        policy=MlpPolicy,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])], act_fun=tf.nn.leaky_relu),
        # policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        env=env,
        gamma=0.95,
        n_steps=math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt']),
        ent_coef=0.0,
        learning_rate=5e-5,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lam=0.95,
        nminibatches=128,
        noptepochs=16,
        cliprange=0.2,
        verbose=1,
    )
    # model.set_update(99)
    # tensorboard
    # Make sure that your chrome browser is already on.
    TensorboardLauncher_local(saver.data_dir + '/PPO2_1')

    # PPO run
    model.learn(total_timesteps=1600000000, eval_every_n=cfg['environment']['eval_every_n'], log_dir=saver.data_dir, record_video=False)

    # Need this line if you want to keep tensorflow alive after training
    input("Press Enter to exit... Tensorboard will be closed after exit\n")

elif mode == 'gen':
    obs = env.reset()

elif mode == 'ref':
    obs = env.reset()
    running_reward = 0.0
    ep_len = 0
    for i in range(1000):
        env.wrapper.showWindow()
        # time.sleep(0.16)
        reward, done = env.stepRef()
        running_reward += reward[0]
        ep_len += 1
        if done[0]:
            print("Episode Reward: {:.2f}".format(running_reward))
            print("Episode Length", ep_len)
            running_reward = 0.0
            ep_len = 0
            break

elif mode == 'retrain':
    weight_path = args.weight
    if weight_path == "":
        print("Can't find trained weight, please provide a trained weight with --weight switch\n")
    else:
        print("Loaded weight from {}\n".format(weight_path))

    ## Add start weight file to saver
    saver = ConfigurationSaver(log_dir=log_dir+'/MappingReachingExp',
                               save_items=[rsg_root+'/env/A1_Reaching_Expanded/env/Environment.hpp', cfg_abs_path])

    model = PPO2.load(
        weight_path,
        tensorboard_log=saver.data_dir,
        policy=MlpPolicy,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])], act_fun=tf.nn.leaky_relu),
        env=env,
        gamma=0.95,
        n_steps=math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt']),
        ent_coef=0.0,
        learning_rate=5e-5,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lam=0.95,
        nminibatches=128,
        noptepochs=16,
        cliprange=0.2,
        verbose=1,
    )

    model.set_env(env)
    model.set_update(5000)

    # tensorboard
    # Make sure that your chrome browser is already on.
    TensorboardLauncher_local(saver.data_dir + '/PPO2_1')
    # PPO run
    model.learn(total_timesteps=1600000000, eval_every_n=cfg['environment']['eval_every_n'], log_dir=saver.data_dir, record_video=False)

    # Need this line if you want to keep tensorflow alive after training
    input("Press Enter to exit... Tensorboard will be closed after exit\n")

else:
    weight_path = args.weight
    if weight_path == "":
        print("Can't find trained weight, please provide a trained weight with --weight switch\n")
    else:
        print("Loaded weight from {}\n".format(weight_path))
        model = PPO2.load(weight_path)
    env.kinect_init()
    obs = env.reset()
    # save_file = current_dir +'/rsc/map_reach/All.pt'
    # contact_file = current_dir +'/rsc/map_reach/Allcont.pt'
    save_file = current_dir +'/rsc/map_reach/man_fin.pt'
    contact_file = current_dir +'/rsc/map_reach/man_cont.pt'
    dim_input = 96
    dim_output = 16
    human_file = current_dir +'/rsc/map_reach/params_reacht.txt'
    X = TripletAllParser2(human_file).tensor_coord.to(device)
    mapper_model = MLP_model_tanh(dim_input, dim_output)
    device = torch.device('cuda')
    mapper_model.load_state_dict(torch.load(save_file, map_location=device))
    mapper_model.cuda()
    mapper_model.eval()
    contact_model = MLP_model_tanh(dim_input, 4)
    contact_model.load_state_dict(torch.load(contact_file, map_location=device))
    contact_model.cuda()
    contact_model.eval()
    y_result = mapper_model.forward(X)
    quat_t0 , q_t0  = simp_result_parser(y_result)
    contact = contact_model.forward(X).cpu().detach().numpy()
    kin_motion = np.concatenate((quat_t0.cpu().detach(), q_t0.cpu().detach()), axis = 1)

    running_reward = 0.0
    ep_len = 0
    env.start_recording_video("Test.mp4")
    joint_keys = ["FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf", "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"]
    plot = Plot(joint_keys, current_dir)
    plot.leg_keys(["FR_hip", "FR_thigh", "FR_calf"], ["FL_hip", "FL_thigh", "FL_calf"], ["RR_hip", "RR_thigh", "RR_calf"], ["RL_hip", "RL_thigh", "RL_calf"])
    af = actionFilterDefault(True)
    af_ref = actionFilterDefault(False)
    for i in range(100000):
        env.wrapper.showWindow()
        action, _ = model.predict(obs)
        action_t = af._FilterAction(action)
        # env.getContactStep(np.array([0,0,0,0], dtype = np.float32), kin_motion[i, :])
        filtered = af_ref._FilterAction(kin_motion[i:i+1,4:])
        kin_motion_filtered = kin_motion[i,:]
        kin_motion_filtered[4:] = filtered
        env.getContactStep(contact[i,:], kin_motion_filtered)
        # env.getContactStep(contact[i,:], kin_motion[i, :])
        obs, reward, done, infos = env.stepTest(action_t)
        torque = env.getTorque()
        running_reward += reward[0]
        total_torque = np.linalg.norm(torque)
        plot.add(i,torque, total_torque)
        ep_len += 1
        if i == kin_motion.shape[0]-1:
            done[0] = True
        if done[0]:
            plot.draw()
            env.stop_recording_video()
            print("Episode Reward: {:.2f}".format(running_reward))
            print("Episode Length", ep_len)
            running_reward = 0.0
            ep_len = 0
            break