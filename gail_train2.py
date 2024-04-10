import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np

# Improved Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Improved Generator (Policy Network)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 7)  # Assuming continuous action space without activation; adjust if needed
        )

    def forward(self, x):
        return self.model(x)

# Adding Gradient Penalty for Discriminator (Optional)
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1), device=real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.shape, device=real_samples.device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def load_data(path):
    with open(path, 'r') as f:
        dataset = json.load(f)
    inputs = torch.tensor([entry["desired_end_effector_position"] for entry in dataset]).float()
    targets = torch.tensor([entry["joint_angles"] for entry in dataset]).float()
    return TensorDataset(inputs, targets)

def train_gail(dataset_path, epochs=100000, batch_size=64, learning_rate=0.0002, gp_lambda=10):
    dataset = load_data(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    loss_fn = nn.BCELoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        for i, (states, actions) in enumerate(dataloader):
            # Preparing input data
            valid = torch.ones((states.size(0), 1), device=states.device)
            fake = torch.zeros((states.size(0), 1), device=states.device)

            # Generator forward pass
            optimizer_G.zero_grad()
            gen_actions = generator(states)
            g_loss = loss_fn(discriminator(gen_actions), valid)
            g_loss.backward()
            optimizer_G.step()

            # Discriminator forward pass
            optimizer_D.zero_grad()
            real_loss = loss_fn(discriminator(actions), valid)
            fake_loss = loss_fn(discriminator(gen_actions.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(discriminator, actions.data, gen_actions.data)
            d_loss += gp_lambda * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

        if d_loss < best_loss:
            best_loss = d_loss.item()
            torch.save(generator.state_dict(), 'best_generator.pth')
            torch.save(discriminator.state_dict(), 'best_discriminator.pth')

        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')


train_gail('/home/navaneet/Desktop/GITHUB/imitation-learning-Franka/datasets/newdataset.json')
