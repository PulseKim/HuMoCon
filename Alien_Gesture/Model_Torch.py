import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_model_simple(nn.Module):
    def __init__(self, dimX, dimY):
        super(MLP_model_simple, self).__init__()
        n_hidden_1 = 256
        n_hidden_2 = 128

        self.fc1 = nn.Linear(dimX, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, dimY)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class MLP_model_sigmoid(nn.Module):
    def __init__(self, dimX, dimY):
        super(MLP_model_sigmoid, self).__init__()
        n_hidden_1 = 256
        n_hidden_2 = 128
        n_hidden_3 = 64

        self.fc1 = nn.Linear(dimX, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, dimY)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

class MLP_model_tanh(nn.Module):
    def __init__(self, dimX, dimY):
        super(MLP_model_tanh, self).__init__()
        n_hidden_1 = 256
        n_hidden_2 = 128
        n_hidden_3 = 128
        n_hidden_4 = 128
        n_hidden_5 = 64
        n_hidden_6 = 64

        self.fc1 = nn.Linear(dimX, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, dimY)
        # self.fc5 = nn.Linear(n_hidden_4, n_hidden_5)
        # self.fc6 = nn.Linear(n_hidden_5, n_hidden_6)
        # self.fc7 = nn.Linear(n_hidden_6, dimY)
        # self.fc5 = nn.Linear(n_hidden_4, dimY)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x

class MLP_model_elu(nn.Module):
    def __init__(self, dimX, dimY):
        super(MLP_model_elu, self).__init__()
        n_hidden_1 = 256
        n_hidden_2 = 128
        n_hidden_3 = 128
        n_hidden_4 = 128
        n_hidden_5 = 64
        n_hidden_6 = 64

        self.fc1 = nn.Linear(dimX, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, dimY)
        # self.fc5 = nn.Linear(n_hidden_4, n_hidden_5)
        # self.fc6 = nn.Linear(n_hidden_5, n_hidden_6)
        # self.fc7 = nn.Linear(n_hidden_6, dimY)
        # self.fc5 = nn.Linear(n_hidden_4, dimY)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        # x = F.leaky_relu(self.fc4(x))
        # x = F.leaky_relu(self.fc5(x))
        # x = F.leaky_relu(self.fc6(x))
        # x = F.leaky_relu(self.fc7(x))
        x = F.tanh(self.fc4(x))
        return x


