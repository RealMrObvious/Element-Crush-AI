import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_QNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()
        # Convolutional layers
        """
        Learn patterns in the match-3 board:
            - colors
            - shapes
            - local gem clusters
            - potential matches
            - overall board structure
            - conv1 detects simple edges/patterns
            - conv2 detects bigger patterns (e.g., L-shapes, rows, combos)
        """
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Compute flattened size after convs
        self.flatten_size = 64 * 9 * 7  # 64 channels * height * width
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #Extracts local tile patterns
        x = F.relu(self.conv2(x)) #NExtracts higher-level match opportunities
        x = x.view(x.size(0), -1) #Flattens the board into a feature vector
        x = F.relu(self.fc1(x))   #Learns state value structure
        x = self.fc2(x)           #Outputs Q-values for every possible move
        return x

    def save(self, file_name='cnn_model.pth'):
        import os
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    """
    This class handles:
    - preparing data
    - running Q-learning updates
    - computing the loss
    - updating the neural network
    """
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # inputs to tensors
        # checking if batch or individual
        # state / next_state: [batch, channels, H, W] or signle [channels, H, W]
        if isinstance(state, torch.Tensor) and len(state.shape) == 3:
            state = state.unsqueeze(0)  # single-step -> batch=1
        if isinstance(next_state, torch.Tensor) and len(next_state.shape) == 3:
            next_state = next_state.unsqueeze(0)

        if isinstance(action, (int, float)):
            action = torch.tensor([action], dtype=torch.long)
        elif isinstance(action, list) or isinstance(action, tuple):
            action = torch.tensor(action, dtype=torch.long)

        if isinstance(reward, (int, float)):
            reward = torch.tensor([reward], dtype=torch.float)
        elif isinstance(reward, list) or isinstance(reward, tuple):
            reward = torch.tensor(reward, dtype=torch.float)

        if isinstance(done, bool):
            done = [done]
        elif isinstance(done, list) or isinstance(done, tuple):
            done = list(done)

        #pass forward
        pred = self.model(state) #preidcts Qvals
        target = pred.clone()

        #Bellman Equation
        for idx in range(len(state)):
            Q_new = reward[idx].item()
            if not done[idx]:  #if not terminal state
                ns = next_state[idx].unsqueeze(0)
                Q_new = reward[idx].item() + self.gamma * torch.max(self.model(ns).detach()).item() #actual new qval
            target[idx, action[idx].item()] = Q_new

        #backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()





