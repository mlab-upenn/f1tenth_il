import torch
from torch import nn, optim


class AgentPolicyMLP(nn.Module):
    """
    The AgentPolicy consists of a two-layer MLP.
    """
    def __init__(self, observ_dim, hidden_dim, action_dim, lr, device):
        super().__init__()

        self.l1 = nn.Linear(observ_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(observ_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        """
        # self.mlp.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr)
        self.loss_function = nn.MSELoss()
        self.device = device
    
    def forward(self, observ_tensor: torch.FloatTensor):
        a = torch.tanh(self.l1(observ_tensor))
        a = torch.tanh(self.l2(a))
        mu = torch.sigmoid(self.mu_head(a))
        # sigma = F.softplus(self.sigma_head(a))
        return mu

    def train(self, observs, actions):
        """
        Trains the agent given a batch of observations and expert-labeled actions.
        
        Returns: the loss.
        """
        observ_tensor = torch.as_tensor(observs, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        self.optimizer.zero_grad()
        pred_action = self(observ_tensor)
        loss = self.loss_function(pred_action, action_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_action(self, observ):
        """
        Predicts an action given an observation.
        """
        observ_tensor = torch.as_tensor(observ, dtype=torch.float32)
        action_tensor = self(observ_tensor)
        return action_tensor.detach().cpu().numpy()