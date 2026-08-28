import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy
import time
from torch.utils.tensorboard import SummaryWriter
import os


class ReplayBuffer:
    """
    Replay Buffer para SAC
    """
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class SACTrainer:
    """
    Implementação do SAC (Soft Actor-Critic)
    
    Componentes:
    - Actor: Política estocástica
    - Critic: Q-functions (Q1, Q2)
    - Target Critic: Q-functions alvo para suavização
    """
    
    def __init__(
        self,
        actor,
        critic,
        env,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=3e-4,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256,
        warmup_steps=1000,
        update_frequency=1,
        target_update_frequency=1,
        log_dir='./logs'
    ):
        self.device = device
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.update_frequency = update_frequency
        self.target_update_frequency = target_update_frequency
        
        # Networks
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic).to(device)
        
        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Learning rate schedulers (criados posteriormente quando conhecemos num_steps)
        self.actor_scheduler = None
        self.critic_scheduler = None
        
        # Entropy coefficient (alpha) - corrigido: log_alpha = log(alpha)
        self.alpha = alpha
        self.target_entropy = -np.prod(env.action_space.shape).item()
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0
        self.episode_count = 0
        
    def select_action(self, state, deterministic=False):
        """
        Seleciona ação usando a política atual
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic=deterministic)
        
        return action.cpu().numpy()[0]
    
    def update_critic(self, state, action, reward, next_state, done):
        """
        Atualiza o Critic usando target Q-value
        """
        # Converter para tensors e normalizar observações para [0,1]
        state = torch.FloatTensor(state).to(self.device) / 255.0
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device) / 255.0
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Calcular target Q-value
        with torch.no_grad():
            # Amostrar ação do actor (Actor retorna action, log_prob já com tanh correction e sum)
            next_action, next_log_prob = self.actor(next_state)
            # next_log_prob já tem shape (B,1), não precisa sum; mantido para compatibilidade se Actor retornar (B, action_dim)
            if next_log_prob.dim() > 1 and next_log_prob.shape[-1] != 1:
                next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)
            
            # Calcular Q-values target
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            
            # Target Q-value
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Calcular Q-values atuais
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Atualizar critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def update_actor(self, state):
        """
        Atualiza o Actor maximizando o Q-value esperado - entropia
        """
        state = torch.FloatTensor(state).to(self.device) / 255.0
        
        # Amostrar ação e calcular log prob (Actor já aplica tanh e correção)
        action, log_prob = self.actor(state)
        if log_prob.dim() > 1 and log_prob.shape[-1] != 1:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Calcular Q-value
        q1, q2 = self.critic(state, action)
        q = torch.min(q1, q2)
        
        # Actor loss: minimizar -(Q - alpha * log_prob)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # Atualizar actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def update_alpha(self, state):
        """
        Atualiza o coeficiente de entropia alpha
        """
        state = torch.FloatTensor(state).to(self.device) / 255.0
        
        with torch.no_grad():
            _, log_prob = self.actor(state)
            if log_prob.dim() > 1 and log_prob.shape[-1] != 1:
                log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Alpha loss corrigido: SAC original usa -(log_alpha * (log_prob + target_entropy))
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        return alpha_loss.item()
    
    def soft_update_target(self):
        """
        Atualiza suavemente o target network
        """
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train_step(self):
        """
        Executa um passo de treinamento
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample do replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Atualizar critic
        critic_loss = self.update_critic(state, action, reward, next_state, done)
        
        # Atualizar actor
        actor_loss = self.update_actor(state)
        
        # Atualizar alpha
        alpha_loss = self.update_alpha(state)
        
        # Atualizar target network
        if self.step_count % self.target_update_frequency == 0:
            self.soft_update_target()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
    
    def train(self, num_steps, eval_frequency=5000, save_frequency=10000):
        """
        Loop principal de treinamento
        """
        # Inicializar schedulers agora que conhecemos num_steps
        if self.actor_scheduler is None:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, T_max=num_steps, eta_min=3e-6
            )
        if self.critic_scheduler is None:
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=num_steps, eta_min=3e-6
            )
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"Iniciando treinamento por {num_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Alpha inicial: {self.alpha:.4f}")
        print(f"Target entropy: {self.target_entropy:.4f}")
        
        start_time = time.time()
        
        for step in range(num_steps):
            self.step_count += 1
            
            # Selecionar ação
            if step < self.warmup_steps:
                # Ação aleatória durante warmup
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state, deterministic=False)
            
            # Step no ambiente
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Armazenar no replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            # Atualizar estado
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Treinar
            if step >= self.warmup_steps and step % self.update_frequency == 0:
                train_info = self.train_step()
                
                if train_info:
                    # Log metrics
                    self.writer.add_scalar('Loss/critic', train_info['critic_loss'], step)
                    self.writer.add_scalar('Loss/actor', train_info['actor_loss'], step)
                    self.writer.add_scalar('Loss/alpha', train_info['alpha_loss'], step)
                    self.writer.add_scalar('Alpha/value', train_info['alpha'], step)
                    
                    # Update learning rate schedulers
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()
                    
                    # Log learning rates
                    self.writer.add_scalar('LR/actor', self.actor_optimizer.param_groups[0]['lr'], step)
                    self.writer.add_scalar('LR/critic', self.critic_optimizer.param_groups[0]['lr'], step)
            
            # Reset se episódio terminou
            if done:
                self.episode_count += 1
                
                # Log episode reward
                self.writer.add_scalar('Reward/episode', episode_reward, self.episode_count)
                self.writer.add_scalar('Length/episode', episode_length, self.episode_count)
                
                print(f"Episode {self.episode_count}: Reward={episode_reward:.2f}, Length={episode_length}")
                
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Avaliação periódica
            if step % eval_frequency == 0 and step > 0:
                eval_reward = self.evaluate(num_episodes=5)
                self.writer.add_scalar('Reward/eval', eval_reward, step)
                print(f"Step {step}: Eval Reward = {eval_reward:.2f}")
            
            # Salvar checkpoint
            if step % save_frequency == 0 and step > 0:
                self.save_checkpoint(step)
        
        total_time = time.time() - start_time
        print(f"\nTreinamento concluído em {total_time/60:.1f} minutos")
        print(f"Total de episódios: {self.episode_count}")
        
        self.writer.close()
    
    def evaluate(self, num_episodes=5, deterministic=True):
        """
        Avalia a política atual
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, deterministic=deterministic)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def save_checkpoint(self, step):
        """
        Salva checkpoint do modelo
        """
        checkpoint_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha.detach().cpu(),
            'log_alpha_value': self.log_alpha.item()
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_{step}.pt'))
        print(f"Checkpoint salvo no step {step}")
     
    def load_checkpoint(self, checkpoint_path):
        """
        Carrega checkpoint do modelo
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        # Restaura log_alpha corretamente mantendo requires_grad e device
        log_alpha_val = checkpoint.get('log_alpha_value', checkpoint['log_alpha'].item() if torch.is_tensor(checkpoint['log_alpha']) else checkpoint['log_alpha'])
        self.log_alpha.data = torch.tensor(log_alpha_val, dtype=torch.float32, device=self.device).data
        # Recria optimizer para garantir referência correta ao novo tensor
        # Se o tensor foi substituído, precisamos atualizar o optimizer param group
        if self.alpha_optimizer.param_groups[0]['params'][0] is not self.log_alpha:
            self.alpha_optimizer.param_groups[0]['params'][0] = self.log_alpha
        
        print(f"Checkpoint carregado do step {checkpoint['step']}")


if __name__ == "__main__":
    # Teste básico
    print("SAC Trainer módulo carregado com sucesso!")
