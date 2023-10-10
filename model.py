import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nnFunctional
import os


class LinearQnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self._load()

    def forward(self, x):
        x = nnFunctional.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename: str = "model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, filename)
        torch.save(self.state_dict(), file_name)

    def _load(self, filename="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, filename)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval()
            print("Model Loaded")

class QTrainer:
    def __init__(self, model: LinearQnet, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float)
        action_tensor = torch.tensor(action, dtype=torch.float)
        reward_tensor = torch.tensor(reward, dtype=torch.float)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float)

        if len(state_tensor.shape) == 1:
            state_tensor = torch.unsqueeze(state_tensor, 0)
            action_tensor = torch.unsqueeze(action_tensor, 0)
            reward_tensor = torch.unsqueeze(reward_tensor, 0)
            next_state_tensor = torch.unsqueeze(next_state_tensor, 0)
            done = (done, )


        # 1: predicted q values with the current state
        prediction = self.model(state_tensor)

        target = prediction.clone()

        for index in range(len(done)):
            Q_new = reward_tensor[index]
            if not done[index]:
                Q_new = reward_tensor[index] + self.gamma + torch.max(self.model(next_state_tensor[index]))

            target[index][torch.argmax(action_tensor).item()] = Q_new


        # 2: next predicted q value
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()
        self.model.train()

