import numpy as np
import torch
from new_alg import action_filter
import math

ACTION_FILTER_HIGH_CUT = 5.0
ACTION_FILTER_HIGH_CUT_HW = 3.0
HUMAN_HISTORY = 5
device = torch.device('cuda')
motion_lables = ('trot', 'manipulate', 'balance', 'lean', 'crabloco', 'jump')

PROCESSING = -1
STAND = 0
SIT = 1
WALK = 2

NOOP = -1
STA2SIT = 0
SIT2STA = 1
STA2WLK = 2

def deg2rad(deg):
  return deg * np.pi /180

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

def result_parser_loco(result):
    t_scaler =  torch.from_numpy(np.array([0.0600, 0.2180, 0.0700, 3.1415])).float()
    t_shifter =  torch.from_numpy(np.array([0.2600, 0.2180, 0.1200, 3.1415])).float()
    scaled_result = torch.add(torch.mul(t_scaler.cuda(), result), t_shifter.cuda())
    return scaled_result

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
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,highcut=[ACTION_FILTER_HIGH_CUT_HW],
                                                num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()
        self._action_filter.init_history(self.init_angle)

    def _FilterAction(self, action):
        filtered_action = np.zeros([1, 12], dtype=np.float32)            
        filtered_action[0,:] = self._action_filter.filter(action[0,:])
        return filtered_action

class actionFilterSit(object): 
    def __init__(self, ):
        self._action_filter = self._BuildActionFilter()
        real_angle = np.array([0.27, deg2rad(50), -1.05, -0.27, deg2rad(50), -1.05, 0., deg2rad(120), deg2rad(-120), 0., deg2rad(120), deg2rad(-120)])
        denom = np.array([0.8028, 2.6179, 0.8901] * 4)
        add = np.array([0., np.pi/2, -1.806]*4)
        self.init_angle = (real_angle - add) / denom        
        self._ResetActionFilter()

    def _BuildActionFilter(self):
        sampling_rate = 30
        num_joints = 12
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,highcut=[ACTION_FILTER_HIGH_CUT_HW],
                                                num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()
        self._action_filter.init_history(self.init_angle)

    def _FilterAction(self, action):
        filtered_action = np.zeros([1, 12], dtype=np.float32)            
        filtered_action[0,:] = self._action_filter.filter(action[0,:])
        return filtered_action

class actionFilterOrientation(object): 
    def __init__(self, ):
        self._action_filter = self._BuildActionFilter()
        self.init_angle = np.array([0, 0, 0])
        self._ResetActionFilter()

    def _BuildActionFilter(self):
        sampling_rate = 30
        num_joints = 3
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,highcut=[ACTION_FILTER_HIGH_CUT_HW],
                                                order=10,num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()
        self._action_filter.init_history(self.init_angle)

    def _FilterAction(self, action):
        filtered_action = np.zeros([3], dtype=np.float32)            
        filtered_action[:] = self._action_filter.filter(action[:])
        return filtered_action

class actionFilterXYZ(object): 
    def __init__(self, ):
        self._action_filter = self._BuildActionFilter()
        self.init_angle = np.array([0, 0, 0])
        self._ResetActionFilter()

    def _BuildActionFilter(self):
        sampling_rate = 30
        num_joints = 3
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,highcut=[ACTION_FILTER_HIGH_CUT_HW],
                                                order=20,num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()
        self._action_filter.init_history(self.init_angle)

    def _FilterAction(self, action):
        filtered_action = np.zeros([3], dtype=np.float32)            
        filtered_action[:] = self._action_filter.filter(action[:])
        return filtered_action

def euler_from_quaternion(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
 
    return np.array([roll_x, pitch_y, yaw_z])

def quaternion_from_euler(euler):
    roll, pitch, yaw = round(euler[0], 2), round(euler[1], 2),round(euler[2], 2) 
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qw, qx, qy, qz]

def model_init(model, name):
    model.load_state_dict(torch.load(name, map_location=device))
    model.cuda()
    model.eval()

class model_null():
    def __init__(self, ):
        pass

    def predict(self, obs):
        return np.array([[0, -0.27334326246, 0.08776541961,0, -0.27334326246, 0.08776541961,0, -0.27334326246, 0.08776541961,0, -0.27334326246, 0.08776541961]], dtype=np.float32), 0

def classifyState(state, human_seq):
    human_s = human_seq[0]
    human_m = human_seq[-4]
    human_l = human_seq[-1]
    if state == PROCESSING:
        return PROCESSING
    elif state == STAND:
        diffrightArmx = human_l[8] - human_s[8]
        diffleftArmx = human_l[11] - human_s[11]
        diffrightArmz = human_l[10] - human_s[10]
        diffleftArmz = human_l[13] - human_s[13]
        diffrightKneex = human_l[26] - human_s[26]
        diffleftKneex = human_l[29] - human_s[29]
        diffrightKneez = human_l[28] - human_s[28]
        diffleftKneez = human_l[31] - human_s[31]

        if diffleftArmz > 0.8 and diffrightArmz > 0.8:
            print("sit")
            return SIT
        elif diffrightArmx > 0.3 and diffrightArmz > 0.15 and diffleftKneex > 0.1 and diffleftKneez > 0.06:
        # elif diffleftKneex > 0.1 and diffleftKneez > 0.06:
            print("walk")
            return WALK
        elif diffleftArmx > 0.3 and diffleftArmz > 0.15 and diffrightKneex > 0.1 and diffrightKneez > 0.06:
            print("walk")
            return WALK
        else:
            return STAND

    elif state == SIT:
        diffrightArmz = human_l[10] - human_s[10]
        diffleftArmz = human_l[13] - human_s[13]
        if diffrightArmz < -0.8 and diffleftArmz < -0.8:
            print("stand")
            return STAND
        else:
            return SIT
    elif state == WALK:
        rightarm_settle = human_l[10]<-0.9 and human_s[10]<-0.9
        leftarm_settle = human_l[13]<-0.9 and human_s[13]<-0.9
        rightknee_settle = human_l[26] < 0.1 and human_s[26] < 0.1 
        leftknee_settle = human_l[29] < 0.1 and human_s[29] < 0.1 
        if rightarm_settle and leftarm_settle and rightknee_settle and leftknee_settle:
            print("stand")
            return STAND
        else:
            return WALK

