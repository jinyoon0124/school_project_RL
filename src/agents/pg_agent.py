"""
Policy Gradient (REINFORCE) Agent for Portfolio Management
"""

import collections
import random
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Config import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.config import PG_CONFIG

# ============================================================================
# 하이퍼파라미터 (config.py에서 가져오기)
# ============================================================================
learning_rate = PG_CONFIG['learning_rate']
gamma = PG_CONFIG['gamma']
num_trajectories = PG_CONFIG['num_trajectories']
num_iterations = PG_CONFIG['num_iterations']
hidden_size = PG_CONFIG['network_hidden_size']


# ============================================================================
# Policy Network 클래스
# ============================================================================
class PolicyNet(nn.Module):
    """
    Policy Network (정책 네트워크)
    
    역할:
    - 상태(state)를 입력받아 각 액션의 확률을 출력
    - Softmax를 사용하여 확률 분포 생성
    
    구조:
    - Input: 11차원 (최근 5일 주식/채권 수익률, 현재 주식 비중)
    - Hidden: 128 → 128
    - Output: 41차원 (각 액션의 확률, softmax)
    """
    
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(11, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 41)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 상태 벡터 (11,) 또는 (batch_size, 11)
            
        Returns:
            액션 확률 분포 (41,) 또는 (batch_size, 41)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
    def get_action_and_logp(self, state):
        """
        액션 샘플링 및 log probability 계산
        
        Args:
            state: 현재 상태 (numpy array)
            
        Returns:
            action: 선택된 액션 인덱스 (0~40)
            logp: log probability (gradient 계산용)
        """
        state_tensor = torch.from_numpy(state).float()
        action_prob = self.forward(state_tensor)
        
        # Categorical distribution으로 액션 샘플링
        m = torch.distributions.Categorical(action_prob)
        action = m.sample()
        logp = m.log_prob(action)
        
        return action.item(), logp
    
    def act(self, state):
        """
        액션만 반환 (평가 시 사용)
        
        Args:
            state: 현재 상태 (numpy array)
            
        Returns:
            action: 선택된 액션 인덱스 (0~40)
        """
        action, _ = self.get_action_and_logp(state)
        return action


# ============================================================================
# Value Network 클래스 (Baseline)
# ============================================================================
class ValueNet(nn.Module):
    """
    Value Network (가치 네트워크) - Baseline으로 사용
    
    역할:
    - 상태(state)를 입력받아 가치(value)를 출력
    - Policy gradient의 분산을 줄이기 위한 baseline
    
    구조:
    - Input: 11차원
    - Hidden: 128 → 128
    - Output: 1차원 (상태 가치)
    """
    
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(11, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 상태 벡터 (11,) 또는 (batch_size, 11)
            
        Returns:
            상태 가치 (1,) 또는 (batch_size, 1)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# Trajectory 수집 함수
# ============================================================================
def collect_trajectory(env, policy):
    """
    한 에피소드를 실행하여 궤적(trajectory) 수집
    
    Args:
        env: PortfolioEnv 환경
        policy: PolicyNet 객체
        
    Returns:
        states: 상태 리스트
        actions: 액션 리스트
        rewards: 보상 리스트
        logps: log probability 리스트
    """
    states = []
    actions = []
    rewards = []
    logps = []
    
    state, _ = env.reset()
    done = False
    
    while not done:
        # 액션 선택
        action, logp = policy.get_action_and_logp(state)
        
        # 환경에서 액션 실행
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = (terminated or truncated)
        
        # 데이터 저장
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        logps.append(logp)
        
        state = next_state
    
    return states, actions, rewards, logps


# ============================================================================
# Return 계산 함수
# ============================================================================
def calc_returns(rewards, gamma):
    """
    Discounted returns 계산
    
    Args:
        rewards: 보상 리스트 [r_0, r_1, ..., r_T]
        gamma: Discount factor
        
    Returns:
        returns: Discounted returns 리스트 [G_0, G_1, ..., G_T]
        
    계산 방식:
        G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... + γ^{T-t}*r_T
    """
    # Discounted rewards 계산
    dis_rewards = [gamma**i * r for i, r in enumerate(rewards)]
    
    # Returns 계산 (뒤에서부터 누적)
    returns = []
    for i in range(len(dis_rewards)):
        returns.append(sum(dis_rewards[i:]))
    
    return returns


# ============================================================================
# Policy Gradient 학습 함수
# ============================================================================
def train_pg(policy, value, trajectories, policy_optimizer, value_optimizer):
    """
    Policy Gradient 학습 수행 (Reward-to-go + Baseline)
    
    Args:
        policy: PolicyNet 객체
        value: ValueNet 객체 (baseline)
        trajectories: 수집한 궤적 리스트
        policy_optimizer: Policy network optimizer
        value_optimizer: Value network optimizer
        
    Policy Gradient 공식 (Reward-to-go + Baseline):
        ∇J(θ) = E[∑_t ∇log π(a_t|s_t) * (G_t - V(s_t))]
        여기서:
        - G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ... (reward-to-go)
        - V(s_t) = baseline (분산 감소)
        - A_t = G_t - V(s_t) (advantage)
        
    Loss:
        Policy: L_π = -1/N * ∑_i ∑_t log π(a_t|s_t) * A_t
        Value:  L_V = 1/N * ∑_i ∑_t (V(s_t) - G_t)^2
    """
    policy_loss_terms = []
    value_loss_terms = []
    
    for states, actions, rewards, logps in trajectories:
        # Returns 계산 (각 timestep의 reward-to-go)
        returns = calc_returns(rewards, gamma)
        
        # 각 timestep에 대해 loss term 계산
        for t in range(len(actions)):
            state_tensor = torch.from_numpy(states[t]).float()
            
            # Value prediction (baseline)
            state_value = value(state_tensor)
            
            # Advantage = Return - Baseline
            advantage = returns[t] - state_value.item()
            
            # Policy gradient with baseline: -log π(a|s) * (G_t - V(s_t))
            policy_loss_terms.append(-logps[t] * advantage)
            
            # Value loss: (V(s_t) - G_t)^2
            value_loss_terms.append((state_value - returns[t])**2)
    
    # Policy loss 계산 및 업데이트
    policy_loss = torch.stack(policy_loss_terms).mean()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    # Value loss 계산 및 업데이트
    value_loss = torch.stack(value_loss_terms).mean()
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    return policy_loss.item(), value_loss.item()


# ============================================================================
# 메인 학습 함수
# ============================================================================
def run_pg_training(env, seed=42, verbose=True):
    """
    Policy Gradient 학습 실행
    
    Args:
        env: PortfolioEnv 환경 객체
        seed: Random seed (재현성)
        verbose: 학습 과정 출력 여부
        
    Returns:
        policy: 학습된 Policy network
        training_log: 학습 로그 (iteration별 성능)
    """
    # Random seed 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Policy network 초기화
    policy = PolicyNet()
    
    # Value network 초기화 (baseline)
    value = ValueNet()
    
    # Optimizer 초기화
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    value_optimizer = optim.Adam(value.parameters(), lr=learning_rate)
    
    # 학습 로그
    training_log = []
    print_interval = 10  # 10 iteration마다 출력
    
    if verbose:
        print(f"=== Policy Gradient Training Started (seed={seed}) ===")
        print(f"Total iterations: {num_iterations}")
        print(f"Trajectories per iteration: {num_trajectories}")
        print(f"Total episodes: {num_iterations * num_trajectories}\n")
    
    # 학습 루프
    for iteration in range(num_iterations):
        # 여러 trajectory 수집
        trajectories = []
        episode_returns = []
        
        for _ in range(num_trajectories):
            states, actions, rewards, logps = collect_trajectory(env, policy)
            trajectories.append((states, actions, rewards, logps))
            
            # 에피소드 총 보상 (undiscounted)
            episode_returns.append(sum(rewards))
        
        # Policy gradient 학습 (with baseline)
        policy_loss, value_loss = train_pg(policy, value, trajectories, 
                                            policy_optimizer, value_optimizer)
        
        # 평균 return 계산
        mean_return = np.mean(episode_returns)
        
        # 로그 저장
        training_log.append({
            'iteration': iteration,
            'mean_return': mean_return,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        })
        
        # 출력
        if verbose and (iteration % print_interval == 0 or iteration == num_iterations - 1):
            print(f"Iteration {iteration:3d} | Mean Return: {mean_return:8.4f} | "
                  f"Policy Loss: {policy_loss:8.4f} | Value Loss: {value_loss:8.4f}")
    
    if verbose:
        print("\n=== Training Complete ===\n")
    
    return policy, value, training_log


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == '__main__':
    """
    테스트용 코드
    
    실행 방법:
        python src/agents/pg_agent.py
    """
    print("=" * 70)
    print("Policy Gradient Agent Test")
    print("=" * 70)
    
    # Policy network 생성
    policy = PolicyNet()
    
    print(f"\nPolicy Network 구조:")
    print(f"  Input: 11 (state dimension)")
    print(f"  Hidden: {hidden_size} → {hidden_size}")
    print(f"  Output: 41 (action probabilities)")
    
    # 테스트 입력
    dummy_state = torch.randn(1, 11)
    output = policy(dummy_state)
    
    print(f"\nInput shape: {dummy_state.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be 1.0): {output.sum().item():.6f}")
    print(f"Output (first 5 action probabilities): {output[0, :5]}")
