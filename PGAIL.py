import torch 

import torch.nn as nn 

import torch.nn.functional as F 

from Experttraj import ExpertTraj 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class Actor(nn.Module): 

    def __init__(self, state_dim, action_dim, max_action): 

        super(Actor, self).__init__() 

        self.max_action = max_action 

        self.net = nn.Sequential( 

            nn.Linear(state_dim, 128), 

            nn.ReLU(), 

            nn.Linear(128, 128), 

            nn.ReLU(), 

            nn.Linear(128, 128), 

            nn.ReLU(), 

            nn.Linear(128, action_dim) 

        ) 

    def forward(self, x): 

        return torch.tanh(self.net(x)) * self.max_action 

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.net(state_action)

class GAIL: 

    def __init__(self, env_name, state_dim, action_dim, max_action, lr, betas, batch_size): 

        self.actor = Actor(state_dim, action_dim, max_action).to(device) 

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas) 

        self.discriminator = Discriminator(state_dim, action_dim).to(device) 

        self.optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas) 

        self.max_action = max_action 

        self.expert = ExpertTraj(env_name) 

        self.loss_fn = nn.BCELoss() 

    def select_action(self, state): 

        state = torch.FloatTensor(state.reshape(1, -1)).to(device) 

        return self.actor(state).cpu().data.numpy().flatten() 

    def update(self, n_iter, batch_size=100): 

        for i in range(n_iter): 

            # sample expert transitions 

            exp_state, exp_action = self.expert.sample(batch_size) 

            #exp_next_state = exp_action[:,0:6] 

            exp_state = torch.FloatTensor(exp_state).to(device) 

            exp_action = torch.FloatTensor(exp_action).to(device) 

            #exp_next_state = torch.FloatTensor(exp_next_state).to(device) 

            # sample expert states for actor 

            state, _ = self.expert.sample(batch_size) 

            state = torch.FloatTensor(state).to(device) 

            action = self.actor(state) 

            next_state = action[:,0:3] 

            next_action = self.actor(next_state) 

            ####################### 

            # update discriminator 

            ####################### 

            self.optim_discriminator.zero_grad() 

            # label tensors 

            exp_label = torch.full((batch_size, 1), 1, device=device) 

            policy_label = torch.full((batch_size, 1), 0, device=device) 

            # with expert transitions 

            prob_exp = self.discriminator(exp_state, exp_action) 

            #loss = prob_exp 

            loss = 0.5 * ((prob_exp - 1) ** 2) 

            # with policy transitions 

            prob_policy = self.discriminator(state, action.detach()) 

            #loss -= prob_policy() 

            loss += 0.5 * (prob_policy ** 2) 

            # take gradient step 

            loss.mean().backward() 

            self.optim_discriminator.step() 

            ################ 

            # update policy 

            ################ 

            self.optim_actor.zero_grad() 

            loss_actor = 0.5 * ((self.discriminator(state, action) - 1) ** 2) 

            loss_actor += 0.5 * ((self.discriminator(next_state, next_action) - 1) ** 2) 

            loss_actor.mean().backward() 

            self.optim_actor.step() 

    def save(self, directory='./preTrained', name='GAIL'): 

        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory, name)) 

        torch.save(self.discriminator.state_dict(), '{}/{}_discriminator.pth'.format(directory, name)) 

    def load(self, directory='./preTrained', name='GAIL'): 

        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory, name))) 

        self.discriminator.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory, name))) 