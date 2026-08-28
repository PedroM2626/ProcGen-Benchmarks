import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicCNN(nn.Module):
    """
    CNN Clássica para SAC - Processa 4 frames grayscale 64x64
    
    Arquitetura:
    Input: (4, 64, 64)
    Conv2d(32, 8x8, stride 4) → ReLU
    Conv2d(64, 4x4, stride 2) → ReLU
    Conv2d(64, 3x3, stride 1) → ReLU
    Flatten → FC(512) → ReLU → FC(3)
    """
    
    def __init__(self, action_dim=3, feature_dim=512):
        super(ClassicCNN, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calcular tamanho da saída após as convoluções
        # Input: 4x64x64
        # Conv1: 32x14x14 ( (64-8)/4 + 1 = 14 )
        # Conv2: 64x6x6 ( (14-4)/2 + 1 = 6 )
        # Conv3: 64x4x4 ( (6-3)/1 + 1 = 4 )
        self.flatten_size = 64 * 4 * 4
        
        # Camadas fully connected
        self.fc1 = nn.Linear(self.flatten_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass
        x: (batch_size, 4, 64, 64) - já normalizado para [0, 1]
        """
        # Convoluções
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """
        Retorna features antes da última camada (útil para debug/visualização)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class ClassicCNNWithFeatureDim(nn.Module):
    """
    Versão flexível que permite especificar diferentes dimensões de features
    Útil para o Actor e Critic do SAC
    """
    
    def __init__(self, action_dim=3, feature_dim=512):
        super(ClassicCNNWithFeatureDim, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.flatten_size = 64 * 4 * 4
        
        self.fc1 = nn.Linear(self.flatten_size, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class ClassicCNNActor(nn.Module):
    """
    Actor network para SAC usando CNN clássica
    Output: ação com tanh + log_prob corrigido (SAC)
    """
    
    def __init__(self, action_dim=3, feature_dim=512, action_scale=1.0):
        super(ClassicCNNActor, self).__init__()
        
        self.feature_extractor = ClassicCNNWithFeatureDim(feature_dim=feature_dim)
        self.fc_mean = nn.Linear(feature_dim, action_dim)
        self.fc_log_std = nn.Linear(feature_dim, action_dim)
        self.action_scale = action_scale
        self.action_dim = action_dim
        
    def forward(self, x):
        """
        Forward com reparametrização + tanh squashing
        Retorna: action (B, action_dim), log_prob (B, 1)
        Compatível com SACTrainer.update_*: action, log_prob = actor(state)
        """
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_std = self.fc_log_std(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        # Reparametrização
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # (B, action_dim)
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        # Log prob com correção tanh: log_prob = log N(x_t) - sum log(1 - tanh^2 + eps)
        log_prob = normal.log_prob(x_t)
        # Enforce correction
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        return action, log_prob

    def get_mean_log_std(self, x):
        """Retorna mean e log_std sem sampling (útil para debug)"""
        features = self.feature_extractor(x)
        mean = self.fc_mean(features)
        log_std = torch.clamp(self.fc_log_std(features), -20, 2)
        return mean, log_std
    
    def get_action(self, x, deterministic=False):
        """
        Retorna apenas ação (sem log_prob), usado em select_action/evaluate
        deterministic=True -> tanh(mean)
        """
        if deterministic:
            mean, _ = self.get_mean_log_std(x)
            return torch.tanh(mean) * self.action_scale
        else:
            action, _ = self.forward(x)
            return action

    def evaluate_log_prob(self, x):
        """Helper para compatibilidade: retorna mean, log_std e log_prob se necessário"""
        return self.forward(x)


class ClassicCritic(nn.Module):
    """
    Critic network (Q-function) para SAC usando CNN clássica
    """
    
    def __init__(self, action_dim=3, feature_dim=512):
        super(ClassicCritic, self).__init__()
        
        self.feature_extractor = ClassicCNNWithFeatureDim(feature_dim=feature_dim)
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Q2 network (para target network smoothing)
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x, action):
        features = self.feature_extractor(x)
        
        # Concatenar features com ação
        x_action = torch.cat([features, action], dim=-1)
        
        q1 = self.q1(x_action)
        q2 = self.q2(x_action)
        
        return q1, q2
    
    def q1_forward(self, x, action):
        """Retorna apenas Q1 (útil para cálculo de target)"""
        features = self.feature_extractor(x)
        x_action = torch.cat([features, action], dim=-1)
        return self.q1(x_action)


if __name__ == "__main__":
    # Teste da rede
    model = ClassicCNN(action_dim=3)
    
    # Input batch: (batch_size, 4, 64, 64)
    x = torch.randn(2, 4, 64, 64)
    
    output = model(x)
    print("Output shape:", output.shape)
    print("Teste concluído!")
    
    # Teste do Actor
    actor = ClassicCNNActor(action_dim=3)
    mean, log_std = actor(x)
    print("Actor mean shape:", mean.shape)
    print("Actor log_std shape:", log_std.shape)
    
    # Teste do Critic
    critic = ClassicCritic(action_dim=3)
    action = torch.randn(2, 3)
    q1, q2 = critic(x, action)
    print("Critic Q1 shape:", q1.shape)
    print("Critic Q2 shape:", q2.shape)
