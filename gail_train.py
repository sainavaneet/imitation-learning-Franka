import torch
from PGAIL import GAIL, ExpertTraj

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training configuration
env_name = "Franka"
# solved_reward = 300
random_seed = 0
lr = 0.0002
betas = (0.9, 0.999)
n_epochs = 3000
n_iter = 100
batch_size = 100
directory = "./preTrained"
filename = "GAIL_pretrained"

state_dim = 3
action_dim = 7
max_action = 2


model = GAIL(env_name, state_dim, action_dim, max_action, lr, betas, batch_size)

if random_seed:
    torch.manual_seed(random_seed)

# Training loop
for epoch in range(1, n_epochs + 1):
    model.update(n_iter, batch_size)
    
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}")
        model.save(directory, f"{filename}_{epoch}")

print("Training completed.")
