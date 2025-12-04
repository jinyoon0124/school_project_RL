"""
DQN Agent for Portfolio Management
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

from src.config import DQN_CONFIG

# ============================================================================
# 하이퍼파라미터 (config.py에서 가져오기)
# ============================================================================
learning_rate = DQN_CONFIG['learning_rate']
gamma = DQN_CONFIG['gamma']
buffer_limit = DQN_CONFIG['buffer_limit']
batch_size = DQN_CONFIG['batch_size']
min_buffer_size = DQN_CONFIG['min_buffer_size']

epsilon_start = DQN_CONFIG['epsilon_start']
epsilon_end = DQN_CONFIG['epsilon_end']
epsilon_decay_episodes = DQN_CONFIG['epsilon_decay_episodes']

target_update_freq = DQN_CONFIG['target_update_freq']
train_iterations_per_step = DQN_CONFIG['train_iterations_per_step']
total_episodes = DQN_CONFIG['total_episodes']


# ============================================================================
# ReplayBuffer 클래스
# ============================================================================
class ReplayBuffer():
    """
    Experience Replay Buffer
    
    역할:
    - DQN의 핵심 구성요소로, 과거 경험 (s, a, r, s', done)을 저장
    - 학습 시 랜덤하게 샘플링하여 미니배치 생성
    - 연속된 경험 간의 상관관계를 깨뜨려 학습 안정성 향상
    
    왜 필요한가?
    - RL은 시계열 데이터이므로 연속된 샘플이 매우 유사함
    - 연속된 샘플로만 학습하면 overfitting 발생
    - 랜덤 샘플링으로 i.i.d (독립 동일 분포) 가정에 가깝게 만듦
    """
    def __init__(self):
        # deque: 양방향 큐, maxlen 설정 시 자동으로 오래된 데이터 제거
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        """
        새로운 경험을 buffer에 추가
        
        Args:
            transition: (state, action, reward, next_state, done_mask) 튜플
        """
        self.buffer.append(transition)
    
    def sample(self, n):
        """
        Buffer에서 랜덤하게 n개의 경험을 샘플링
        
        Args:
            n: 샘플링할 경험의 개수 (batch_size)
            
        Returns:
            각각 배치 형태의 텐서:
            - s_lst: 상태 (batch_size, 11)
            - a_lst: 액션 (batch_size, 1)
            - r_lst: 보상 (batch_size, 1)
            - s_prime_lst: 다음 상태 (batch_size, 11)
            - done_mask_lst: 종료 마스크 (batch_size, 1)
        """
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        # 최적화: numpy 배열로 먼저 변환 후 텐서 생성 (20-30% 속도 향상)
        # torch.tensor()보다 torch.from_numpy()가 훨씬 빠름
        s_arr = np.array(s_lst, dtype=np.float32)
        a_arr = np.array(a_lst, dtype=np.int64)
        r_arr = np.array(r_lst, dtype=np.float32)
        s_prime_arr = np.array(s_prime_lst, dtype=np.float32)
        done_mask_arr = np.array(done_mask_lst, dtype=np.float32)
        
        return torch.from_numpy(s_arr), \
               torch.from_numpy(a_arr), \
               torch.from_numpy(r_arr), \
               torch.from_numpy(s_prime_arr), \
               torch.from_numpy(done_mask_arr)
    
    def size(self):
        """현재 buffer에 저장된 경험의 개수 반환"""
        return len(self.buffer)


# ============================================================================
# Q-Network 클래스
# ============================================================================
class Qnet(nn.Module):
    """
    Deep Q-Network
    
    역할:
    - 상태(state)를 입력받아 각 액션의 Q-value를 출력
    - Q-value: 특정 상태에서 특정 액션을 취했을 때의 기대 누적 보상
    
    구조:
    - Input: 11차원 상태 벡터
    - Hidden Layer 1: 128 units + ReLU
    - Hidden Layer 2: 128 units + ReLU
    - Output: 41차원 (각 액션의 Q-value)
    
    왜 이 구조인가?
    - ex011 CartPole과 동일한 구조 (공정한 비교)
    - 128 units: 충분한 표현력, 과적합 방지
    - 2개 hidden layers: 비선형 패턴 학습에 적합
    """
    def __init__(self):
        super(Qnet, self).__init__()
        # 11차원 입력 (최근 5일 주식 수익률 + 5일 채권 수익률 + 현재 주식 비중)
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 128)
        # 41개 액션 출력 (주식 비중 변화량: -1.0 ~ +1.0, 0.05 단위)
        self.fc3 = nn.Linear(128, 41)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 상태 벡터 (batch_size, 11) 또는 (11,)
            
        Returns:
            Q-values (batch_size, 41) 또는 (41,)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 출력층은 활성화 함수 없음 (Q-value는 실수)
        return x
    
    def sample_action(self, obs, epsilon):
        """
        Epsilon-greedy 정책으로 액션 선택
        
        Args:
            obs: 현재 상태 (numpy array 또는 tensor)
            epsilon: 탐험 확률 (0~1)
            
        Returns:
            선택된 액션 인덱스 (0~40)
            
        동작 방식:
        - epsilon 확률로 랜덤 액션 선택 (탐험, exploration)
        - (1-epsilon) 확률로 최대 Q-value 액션 선택 (활용, exploitation)
        """
        out = self.forward(obs)
        coin = random.random()
        
        if coin < epsilon:
            # 탐험: 랜덤 액션 (0~40 중 하나)
            return random.randint(0, 40)
        else:
            # 활용: 최대 Q-value를 가진 액션 선택
            return out.argmax().item()


# ============================================================================
# DQN 학습 함수
# ============================================================================
def train_dqn(q, q_target, memory, optimizer):
    """
    DQN 학습 수행 (1 step당 여러 번 반복)
    
    Args:
        q: 학습할 Q-network (main network)
        q_target: Target Q-network (안정적인 학습을 위한 고정된 네트워크)
        memory: ReplayBuffer 객체
        optimizer: Adam optimizer
    
    DQN 알고리즘 핵심:
    1. Replay buffer에서 미니배치 샘플링
    2. Q(s, a) 계산 (현재 네트워크)
    3. Target 계산: r + γ * max_a' Q_target(s', a')
    4. Loss 계산: MSE(Q(s,a), target)
    5. Backpropagation으로 네트워크 업데이트
    
    왜 Target Network가 필요한가?
    - Q-learning은 자기 자신의 예측으로 자신을 학습 (bootstrapping)
    - 같은 네트워크를 사용하면 target이 계속 변해서 불안정
    - Target network를 고정하여 안정적인 학습 목표 제공
    """
    for i in range(train_iterations_per_step):
        # 1. 미니배치 샘플링
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
        # 2. 현재 Q-value 계산
        q_out = q(s)  # (batch_size, 41)
        q_a = q_out.gather(1, a)  # (batch_size, 1) - 실제 선택한 액션의 Q-value
        
        # 3. Target Q-value 계산
        # max_q_prime: 다음 상태에서 가능한 최대 Q-value
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # (batch_size, 1)
        
        # Bellman equation: Q(s,a) = r + γ * max_a' Q(s', a')
        # done_mask: 에피소드가 끝나면 0, 아니면 1 (종료 시 미래 보상 없음)
        target = r + gamma * max_q_prime * done_mask
        
        # 4. Loss 계산 (Mean Squared Error)
        loss = F.mse_loss(q_a, target)
        
        # 5. Backpropagation
        optimizer.zero_grad()  # 이전 gradient 초기화
        loss.backward()        # Gradient 계산
        optimizer.step()       # 파라미터 업데이트


# ============================================================================
# 메인 학습 함수
# ============================================================================
def run_dqn_training(env, seed=42, verbose=True):
    """
    DQN 학습 실행
    
    Args:
        env: PortfolioEnv 환경 객체
        seed: Random seed (재현성)
        verbose: 학습 과정 출력 여부
        
    Returns:
        q: 학습된 Q-network
        q_target: Target Q-network
        training_log: 학습 로그 (에피소드별 성능)
    """
    # Random seed 설정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 네트워크 초기화
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())  # 초기 가중치 복사
    
    # Replay buffer 초기화
    memory = ReplayBuffer()
    
    # Optimizer 초기화
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    
    # 학습 로그
    training_log = []
    print_interval = 20  # 20 에피소드마다 출력
    score = 0.0
    
    if verbose:
        print(f"=== DQN Training Started (seed={seed}) ===")
        print(f"Total episodes: {total_episodes}")
        print(f"Epsilon decay: {epsilon_start} → {epsilon_end} over {epsilon_decay_episodes} episodes")
        print(f"Target update frequency: every {target_update_freq} episodes\n")
    
    # 학습 루프
    for n_epi in range(total_episodes):
        # Epsilon 선형 감소 (Linear annealing)
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (n_epi / epsilon_decay_episodes))
        
        # 에피소드 시작
        s, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        # 에피소드 진행
        while not done:
            # Action 선택 (epsilon-greedy)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            
            # 환경에서 액션 실행
            s_prime, r, terminated, truncated, info = env.step(a)
            done = (terminated or truncated)
            done_mask = 0.0 if done else 1.0
            
            # Replay buffer에 경험 저장
            memory.put((s, a, r, s_prime, done_mask))
            
            s = s_prime
            episode_reward += r
            
            # 학습 수행 (buffer에 충분한 데이터가 쌓였을 때)
            if memory.size() > min_buffer_size:
                train_dqn(q, q_target, memory, optimizer)
        
        score += episode_reward
        
        # Target network 업데이트 (Hard update)
        if n_epi % target_update_freq == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
        
        # 진행 상황 출력
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            if verbose:
                print(f"Episode {n_epi:4d} | "
                      f"Avg Reward: {avg_score:7.4f} | "
                      f"Buffer: {memory.size():5d} | "
                      f"Epsilon: {epsilon*100:4.1f}%")
            
            training_log.append({
                'episode': n_epi,
                'avg_reward': avg_score,
                'epsilon': epsilon,
                'buffer_size': memory.size()
            })
            score = 0.0
    
    if verbose:
        print("\n=== Training Complete ===")
    
    return q, q_target, training_log


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == '__main__':
    """
    테스트용 코드
    실제 사용 시에는 train_dqn.py에서 이 모듈을 import하여 사용
    """
    print("DQN Agent module loaded successfully!")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Batch size: {batch_size}")
    print(f"  Buffer size: {buffer_limit}")
    print(f"  Epsilon: {epsilon_start} → {epsilon_end} (decay over {epsilon_decay_episodes} episodes)")
    print(f"  Target update: every {target_update_freq} episodes")
    print(f"  Total episodes: {total_episodes}")
    
    # 네트워크 구조 테스트
    print(f"\nQ-Network structure:")
    q = Qnet()
    print(q)
    
    # 더미 입력으로 forward pass 테스트
    dummy_state = torch.randn(1, 11)  # (batch_size=1, state_dim=11)
    output = q(dummy_state)
    print(f"\nInput shape: {dummy_state.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (Q-values for 41 actions): {output}")
