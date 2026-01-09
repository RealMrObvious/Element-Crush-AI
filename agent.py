from match3 import match3AI
import torch
import random,sys
import numpy as np
from collections import deque
from model import QTrainer, CNN_QNet
import matplotlib.pyplot as plt
from IPython import display
import game_predictor_agent as gp

MAX_MEMORY = 50_000
BATCH_SIZE = 128
LR = 0.00025





class Agent():
    def __init__(self,game_w,game_h):
        self.game_w = game_w
        self.game_h = game_h
        self.num_games = 0 
        self.epsilon = 0 # randomness
        self.gamma = 0.8 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.state_size = 11
        self.action_size = (game_h*(game_w - 1)) + (game_w*(game_h - 1))
        self.model = CNN_QNet(self.state_size, self.action_size)

        self.model.load_state_dict(torch.load("demo_model.pth"))
        self.model.train()
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    
    def process_state(self,state_dict,num_elements=6):
        board = np.array(state_dict["board"], dtype=np.float32)
        rows, cols = board.shape

        #One-hot encode the board (tile layers 1..6)
        tile_layers = np.zeros((num_elements, rows, cols), dtype=np.float32)    
        for val in range(1, num_elements + 1):
            tile_layers[val - 1] = (board == val).astype(np.float32)

        # Extra channels (broadcasted across entire board)
        turns_left = np.full((1, rows, cols), state_dict["turns_left"] / 100.0, dtype=np.float32)
        score      = np.full((1, rows, cols), state_dict["score"]      / 500.0, dtype=np.float32)
        goal       = np.full((1, rows, cols), state_dict["goal"]       / 500.0, dtype=np.float32)
        combo      = np.full((1, rows, cols), state_dict["combo"]      / 10.0,  dtype=np.float32)
        gameover   = np.full((1, rows, cols), 1.0 if state_dict["gameover"] else 0.0, dtype=np.float32)

        all_layers = np.concatenate([
            tile_layers,   # 6 layers
            turns_left,    # 1 layer
            score,         # 1 layer
            goal,          # 1 layer
            combo,         # 1 layer
            gameover       # 1 layer
        ], axis=0)

        # Convert to torch tensor
        tensor = torch.from_numpy(all_layers).float()
        return tensor


    def pick_action(self, state_tensor, full_random = False):
        # state_tensor: shape (1,C,H,W)
        game_h = self.game_h
        game_w = self.game_w
        horizontal_count = game_h * (game_w - 1)   # 9 * 8 = 72
        vertical_count   = game_w * (game_h - 1)   # 9 * 8 = 72

        self.epsilon = max(0.05, 0.995 ** self.num_games)  # fraction 0..1
        if random.random() < self.epsilon or full_random:
            move =  random.randint(0,self.action_size-1) 
            action_id = move
        else:
            # --- Exploitation ---
            pred = self.model(state_tensor)
            # print(pred)
            action_id = torch.argmax(pred).item()
            # print(action_id)
            move = action_id

        #move = 13 then (5,1) -> (6,1)
        if move < horizontal_count:
            #direction = 0 #"RIGHT"
            x = move % (game_w - 1)
            y = move // (game_w - 1)
            nx = x+1
            ny = y

        else:
            #direction = 1 #"DOWN"
            move -= horizontal_count
            x = move % game_w
            y = move // game_w
            nx = x
            ny = y + 1

        return [x, y, nx, ny], action_id

    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # stack only if not already tensor
        states_tensor = torch.stack(states)   # shape [batch, channels, H, W]
        next_states_tensor = torch.stack(next_states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float)
        dones_tensor = list(dones)

        self.trainer.train_step(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)


    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def reward(self, prev_state,new_state):
        if not prev_state or not new_state:
            return 0

        # Extract fields:
        prev_score = prev_state["score"]
        new_score  = new_state["score"]

        prev_board = prev_state["board"]
        new_board  = new_state["board"]
        reward = 0

        tile_gain = new_score - prev_score
        reward += tile_gain *(20)   # increase weight

        if new_board == prev_board:
            reward -= 0.5           # stronger penalty

        if new_state["gameover"]:
            reward += 250 if new_score >= new_state["goal"] else -25

        return reward

    

