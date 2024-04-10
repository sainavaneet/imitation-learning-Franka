import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

batch_size = 64

class Actor(nn.Module):
    def __init__(self, state_dim=3, action_dim=7):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        return torch.tanh(self.net(x)) * 2

class Discriminator(nn.Module):
    def __init__(self, state_dim=3, action_dim=7):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, state, action):
        assert state.size(0) == action.size(0)
        state_action = torch.cat([state, action], 1)
        return self.net(state_action)

def load_dataset(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def prepare_data(dataset):
    inputs = torch.tensor([entry["desired_end_effector_position"] for entry in dataset]).float()
    targets = torch.tensor([entry["joint_angles"] for entry in dataset]).float()
    return inputs, targets

def train_gail(generator, discriminator, expert_inputs, expert_targets, batch_size=64, num_epochs=10000):
    criterion_disc = nn.BCELoss()
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=1e-4)
    dataset = TensorDataset(expert_inputs, expert_targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        disc_loss_accumulated = 0.0
        gen_loss_accumulated = 0.0
        for expert_states, expert_targets in train_loader:
            generated_actions = generator(expert_states)
            optimizer_disc.zero_grad()
            expert_predictions = discriminator(expert_states, expert_targets)
            generated_predictions = discriminator(expert_states, generated_actions.detach())
            real_labels = torch.ones(expert_predictions.size())
            fake_labels = torch.zeros(generated_predictions.size())
            disc_loss = criterion_disc(expert_predictions, real_labels) + criterion_disc(generated_predictions, fake_labels)
            disc_loss.backward()
            optimizer_disc.step()
            optimizer_gen.zero_grad()
            generated_predictions_for_gen = discriminator(expert_states, generated_actions)
            gen_loss = -criterion_disc(generated_predictions_for_gen, fake_labels)
            gen_loss.backward()
            optimizer_gen.step()
            disc_loss_accumulated += disc_loss.item()
            gen_loss_accumulated += gen_loss.item()
        if epoch % 10 == 0:
            avg_disc_loss = disc_loss_accumulated / len(train_loader)
            avg_gen_loss = gen_loss_accumulated / len(train_loader)
            print(f"Epoch {epoch}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}")
    torch.save(generator.state_dict(), 'generator_model.pth')
    torch.save(discriminator.state_dict(), 'discriminator_model.pth')

dataset_path = "/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/datasets/newdataset.json"
dataset = load_dataset(dataset_path)
inputs, targets = prepare_data(dataset)
generator = Actor()
discriminator = Discriminator()
inputs, targets = prepare_data(load_dataset(dataset_path))
train_dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_gail(generator, discriminator, inputs, targets, batch_size=64, num_epochs=1000)
