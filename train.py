import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from utils import ImitationLearningModel

def load_dataset(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def prepare_data(dataset):
    inputs = torch.tensor([entry["desired_end_effector_position"] for entry in dataset]).float()
    targets = torch.tensor([entry["joint_angles"] for entry in dataset]).float()
    return inputs, targets

if __name__ == "__main__":
    start_time = time.time()  # Start timing
    dataset_path = "/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/datasets/newdataset.json"
    dataset = load_dataset(dataset_path)
    inputs, targets = prepare_data(dataset)

    model = ImitationLearningModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
    torch.save(model.state_dict(), "/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/models/newmodel.pth")
    end_time = time.time()  # End timing
    total_time = end_time - start_time
    print(f"Total training time: {total_time} seconds")
    print("Model training complete and saved.")
