import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json


class ImitationLearningModel(nn.Module):
    def __init__(self):
        super(ImitationLearningModel, self).__init__()
        self.fc1 = nn.Linear(3, 128)  
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_dataset(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def prepare_data(dataset):
    inputs = torch.tensor([entry["desired_end_effector_position"] for entry in dataset]).float()
    targets = torch.tensor([entry["joint_angles"] for entry in dataset]).float()
    return inputs, targets

if __name__ == "__main__":
    dataset_path = "/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/datasets/test.json"
    dataset = load_dataset(dataset_path)
    inputs, targets = prepare_data(dataset)

    model = ImitationLearningModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/models/test.pth")
    
    print("Model training complete and saved.")
