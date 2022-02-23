import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TripletAllParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        successor = []
        cnt = 0
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            leg_st_idx = 14
            for i in range(0, 32):
                successor.append(float(list_y[i]))
            cnt += 1
            if cnt %3 == 0 :
                self.successor.append(successor)
                successor = []

class TripletAllParser2(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.pp = []
        self.past = []
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        concat1 = np.concatenate((np.array(self.pp), np.array(self.past)), axis = 1)
        self.np_coord = np.concatenate((concat1, np.array(self.successor)), axis = 1)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        
        start = True
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            leg_st_idx = 14
            successor = []
            for i in range(0, 32):
                successor.append(float(list_y[i]))
            if start:
                self.pp.append(successor)
                self.past.append(successor)
                start = False
            else:
                self.pp.append(self.past[-1])
                self.past.append(self.successor[-1])
            self.successor.append(successor)

class TripletLowerParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        successor = []
        cnt = 0
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            leg_st_idx = 14
            for i in range(0, 8):
                successor.append(float(list_y[i]))
            for i in range(leg_st_idx, 20):
                successor.append(float(list_y[i]))
            cnt += 1
            if cnt %3 == 0 :
                self.successor.append(successor)
                successor = []

class TripletLowerParser2(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.pp = []
        self.past = []
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        concat1 = np.concatenate((np.array(self.successor), np.array(self.past)), axis = 1)
        self.np_coord = np.concatenate((concat1, np.array(self.pp)), axis = 1)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        
        start = True
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            leg_st_idx = 14
            successor = []
            for i in range(0, 8):
                successor.append(float(list_y[i]))
            for i in range(leg_st_idx, 20):
                successor.append(float(list_y[i]))
            if start:
                self.pp.append(successor)
                self.past.append(successor)
                start = False
            else:
                self.pp.append(self.past[-1])
                self.past.append(self.successor[-1])
            self.successor.append(successor)

class SimpUpperRootChest(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            successor = []
            for i in range(0, 32):
                successor.append(float(list_y[i]))
            self.successor.append(successor)

class SimpUpperChest(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            successor = []
            leg_st_idx = 14
            for i in range(0, 4):
                successor.append(float(list_y[i]))
            for i in range(8, leg_st_idx):
                successor.append(float(list_y[i]))
            self.successor.append(successor)

class SimpLowerChest(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        # From 20 data, with chest0 / root4/ rarm8/ larm11/ rleg14/ lleg17.
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            successor = []
            leg_st_idx = 14
            for i in range(0, 4):
                successor.append(float(list_y[i]))
            for i in range(leg_st_idx, 20):
                successor.append(float(list_y[i]))
            self.successor.append(successor)


class SampleParserLowerRoot(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.state = []
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_angle = torch.from_numpy(self.np_angle).float()
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_angle = np.array(self.state)
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        len_human_data = 55
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            state = []
            successor = []
            for i in range(0, len_human_data):
                state.append(float(list_y[i]))
            for i in range(len_human_data, len_human_data+4):
                successor.append(float(list_y[i]))
            for i in range(len_human_data+10, len_human_data+16):
                successor.append(float(list_y[i]))
            self.state.append(state)
            self.successor.append(successor)

class SampleParserLower(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.state = []
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_angle = torch.from_numpy(self.np_angle).float()
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_angle = np.array(self.state)
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        len_human_data = 55
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            state = []
            successor = []
            for i in range(0, len_human_data):
                state.append(float(list_y[i]))
            for i in range(len_human_data, len_human_data+4):
                successor.append(float(list_y[i]))
            for i in range(len_human_data+10, len_human_data+16):
                successor.append(float(list_y[i]))
            self.state.append(state)
            self.successor.append(successor)


class SampleParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.state = []
        self.successor = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()
        

    def convert_to_tensor(self, ):
        self.tensor_angle = torch.from_numpy(self.np_angle).float()
        self.tensor_coord = torch.from_numpy(self.np_coord).float()

    def convert_to_numpy(self, ):
        self.np_angle = np.array(self.state)
        self.np_coord = np.array(self.successor)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        len_human_data = 55
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            len_list = len(list_y)
            if len_list < 2 :
                break
            state = []
            successor = []
            for i in range(0, len_human_data):
                state.append(float(list_y[i]))

            for i in range(len_human_data, len_human_data+16):
                successor.append(float(list_y[i]))

            self.state.append(state)
            self.successor.append(successor)

class DataParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor = []
        self.orientation = []
        self.scaler = []
        self.shifter = []
        self.real = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()
        

    def convert_to_tensor(self, ):
        self.tensor_orientation = torch.from_numpy(self.np_orientation).float()
        self.tensor_robot = torch.from_numpy(self.np_robot).float()
        self.tensor_scaler = torch.from_numpy(self.np_scaler).float()
        self.tensor_shifter = torch.from_numpy(self.np_shifter).float()

    def convert_to_numpy(self, ):
        self.np_orientation_raw = np.array(self.orientation)
        epsilon = 1e-6
        norm_ori = np.matmul(np.square(self.np_orientation_raw), np.ones((4,1)))
        self.np_orientation = np.divide(self.np_orientation_raw, norm_ori + epsilon)
        self.np_robot = np.array(self.successor)
        self.np_scaler = np.array(self.scaler)
        self.np_shifter = np.array(self.shifter)
        self.np_real = np.array(self.real)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        len_robot_angles = 12
        len_quat = 4
        info_flag = True

        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            if info_flag:
                for i in range(0, len_robot_angles):
                    self.scaler.append((float(list_y[3*i+1]) - float(list_y[3*i]))/2.)
                    self.shifter.append((float(list_y[3*i+1]) + float(list_y[3*i]))/2. - float(list_y[3*i+2]))
                    self.real.append(float(list_y[3*i+2]))
                info_flag = False
                continue
            len_list = len(list_y)
            if len_list < 2 :
                break
            orientation = []
            successor = []
            for i in range(0, 4):
                orientation.append(float(list_y[i]))
            for i in range(4, 16):
                successor.append(float(list_y[i]))
            self.orientation.append(orientation)
            self.successor.append(successor)

class TripletDataParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.successor0 = []
        self.successor1 = []
        self.successor2 = []
        self.orientation0 = []
        self.orientation1 = []
        self.orientation2 = []
        self.scaler = []
        self.shifter = []
        self.real = []
        self.read_data()
        self.convert_to_numpy()
        self.convert_to_tensor()

    def convert_to_tensor(self, ):
        self.tensor_orientation0 = torch.from_numpy(self.np_orientation0).float()
        self.tensor_orientation1 = torch.from_numpy(self.np_orientation1).float()
        self.tensor_orientation2 = torch.from_numpy(self.np_orientation2).float()
        self.tensor_robot0 = torch.from_numpy(self.np_robot0).float()
        self.tensor_robot1 = torch.from_numpy(self.np_robot1).float()
        self.tensor_robot2 = torch.from_numpy(self.np_robot2).float()
        self.tensor_scaler = torch.from_numpy(self.np_scaler).float()
        self.tensor_shifter = torch.from_numpy(self.np_shifter).float()

    def convert_to_numpy(self, ):
        self.np_orientation_raw0 = np.array(self.orientation0)
        self.np_orientation_raw1 = np.array(self.orientation1)
        self.np_orientation_raw2 = np.array(self.orientation2)
        epsilon = 1e-6
        norm_ori0 = np.matmul(np.square(self.np_orientation_raw0), np.ones((4,1)))
        norm_ori1 = np.matmul(np.square(self.np_orientation_raw1), np.ones((4,1)))
        norm_ori2 = np.matmul(np.square(self.np_orientation_raw2), np.ones((4,1)))
        self.np_orientation0 = np.divide(self.np_orientation_raw0, norm_ori0 + epsilon)
        self.np_orientation1 = np.divide(self.np_orientation_raw1, norm_ori1 + epsilon)
        self.np_orientation2 = np.divide(self.np_orientation_raw2, norm_ori2 + epsilon)
        self.np_robot0 = np.array(self.successor0)
        self.np_robot1 = np.array(self.successor1)
        self.np_robot2 = np.array(self.successor2)
        self.np_scaler = np.array(self.scaler)
        self.np_shifter = np.array(self.shifter)
        self.np_real = np.array(self.real)

    def read_data(self, ):
        file_in = open(self.file_path, 'r')
        len_robot_angles = 12
        len_quat = 4
        info_flag = True

        cnt = 0
        for y in file_in.read().split('\n'):
            list_y = y.split(" ")
            if info_flag:
                for i in range(0, len_robot_angles):
                    self.scaler.append((float(list_y[3*i+1]) - float(list_y[3*i]))/2.)
                    self.shifter.append((float(list_y[3*i+1]) + float(list_y[3*i]))/2. - float(list_y[3*i+2]))
                    self.real.append(float(list_y[3*i+2]))
                info_flag = False
                continue
            len_list = len(list_y)
            if len_list < 2 :
                break
            orientation = []
            successor = []
            for i in range(0, 4):
                orientation.append(float(list_y[i]))
            for i in range(4, 16):
                successor.append(float(list_y[i]))
            if cnt % 3 == 2 : 
                self.orientation0.append(orientation)
                self.successor0.append(successor)
            elif cnt % 3 == 1 : 
                self.orientation1.append(orientation)
                self.successor1.append(successor)
            elif cnt % 3 == 0 : 
                self.orientation2.append(orientation)
                self.successor2.append(successor)
            cnt += 1