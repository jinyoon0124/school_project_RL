import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class PortfolioEnv(gym.Env):
    """
    강화학습을 위한 포트폴리오 관리 환경
    
    State (11차원):
        - 최근 5일 주식 수익률 (5)
        - 최근 5일 채권 수익률 (5)
        - 현재 주식 비중 (1)
        
    Action (Discrete 41):
        - 주식 비중 변화량 (Delta w)
        - -1.0 ~ +1.0 (0.05 단위)
        
    Reward:
        - Sharpe Proxy: r - lambda * r^2
    """
    
    def __init__(self, df, lambda_risk=1.0, episode_years=10):
        """
        초기화
        
        Args:
            df (pd.DataFrame): 전처리된 데이터 (Index: Date, Columns: ['r_stock', 'r_bond'])
            lambda_risk (float): 위험 회피 계수 (기본값: 1.0)
            episode_years (int): 에피소드 길이 (년 단위, 기본값: 10)
        """
        super(PortfolioEnv, self).__init__()
        
        self.df = df
        self.lambda_risk = lambda_risk
        self.episode_length = episode_years * 252  # 대략적인 거래일 수
        
        # Action Space: 41개 이산형 액션
        # -1.0, -0.95, ..., 0, ..., 0.95, 1.0
        self.action_space = spaces.Discrete(41)
        self.action_values = np.linspace(-1.0, 1.0, 41)
        
        # Observation Space: 11차원 연속형 벡터
        # [r_stock_t-4...t, r_bond_t-4...t, w_stock_t]
        # 수익률은 대략 -0.5 ~ 0.5 범위, 비중은 0~1 범위
        low = np.array([-1.0] * 10 + [0.0])
        high = np.array([1.0] * 10 + [1.0])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 내부 변수 초기화
        self.current_step = 0
        self.current_idx = 0
        self.start_idx = 0
        self.w_stock = 0.5
        self.w_bond = 0.5
        self.portfolio_value = 1.0
        
        print(f"PortfolioEnv initialized with lambda={lambda_risk}, episode_years={episode_years}")
    
    def _get_state(self):
        """
        현재 시점의 상태 벡터 생성
        
        Returns:
            np.array: 11차원 상태 벡터
                [r_stock_{t-4}, ..., r_stock_t,  # 5개
                 r_bond_{t-4}, ..., r_bond_t,    # 5개
                 w_stock]                         # 1개
        """
        # 과거 5일 수익률 추출 (current_idx-4 ~ current_idx)
        start = self.current_idx - 4
        end = self.current_idx + 1
        
        r_stock_history = self.df['r_stock'].iloc[start:end].values  # 5개
        r_bond_history = self.df['r_bond'].iloc[start:end].values    # 5개
        
        # 상태 벡터 구성
        state = np.concatenate([
            r_stock_history,  # 5개
            r_bond_history,   # 5개
            [self.w_stock]    # 1개
        ]).astype(np.float32)
        
        return state
    
    def reset(self, seed=None, options=None):
        """
        에피소드 시작 시 환경 초기화
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options
            
        Returns:
            tuple: (state, info)
                - state: 11차원 초기 상태 벡터
                - info: 추가 정보 딕셔너리
        """
        # Random seed 설정
        super().reset(seed=seed)
        
        # 가능한 시작 인덱스 범위 계산
        # - 최소: 5 (과거 5일 데이터 필요)
        # - 최대: len(df) - episode_length (에피소드가 끝까지 갈 수 있어야 함)
        min_start = 5
        max_start = len(self.df) - self.episode_length
        
        # 29개 구간 중 랜덤 선택 (1년 단위 sliding)
        # 1년 ≈ 252 거래일
        num_windows = (max_start - min_start) // 252
        window_idx = self.np_random.integers(0, num_windows)
        self.start_idx = min_start + window_idx * 252
        
        # 내부 변수 초기화
        self.current_idx = self.start_idx
        self.current_step = 0
        self.w_stock = 0.5
        self.w_bond = 0.5
        self.portfolio_value = 1.0
        
        # 초기 상태 생성
        state = self._get_state()
        
        info = {
            'start_date': self.df.index[self.start_idx],
            'window_idx': window_idx
        }
        
        return state, info
    
    def _is_rebalance_day(self):
        """
        현재 날짜가 월초(리밸런싱 날짜)인지 확인
        
        Returns:
            bool: 월초면 True, 아니면 False
        """
        if self.current_idx == self.start_idx:
            # 에피소드 첫 날은 리밸런싱 안 함
            return False
        
        current_date = self.df.index[self.current_idx]
        prev_date = self.df.index[self.current_idx - 1]
        
        # 월이 바뀌었는지 확인
        return current_date.month != prev_date.month
    
    def _calculate_reward(self, portfolio_return):
        """
        Sharpe proxy reward 계산
        
        Args:
            portfolio_return (float): 포트폴리오 일간 수익률
            
        Returns:
            float: reward = r - lambda * r^2
        """
        return portfolio_return - self.lambda_risk * (portfolio_return ** 2)
    
    def step(self, action):
        """
        액션 실행 및 다음 상태로 전환
        
        Args:
            action (int): 0~40 사이의 정수 (액션 인덱스)
            
        Returns:
            tuple: (next_state, reward, terminated, truncated, info)
        """
        # 1. 액션을 Delta w로 변환
        delta_w = self.action_values[action]
        
        # 2. 리밸런싱 (월초만)
        if self._is_rebalance_day():
            # 주식 비중 업데이트 (0~1 범위로 클리핑)
            self.w_stock = np.clip(self.w_stock + delta_w, 0.0, 1.0)
            self.w_bond = 1.0 - self.w_stock
        
        # 3. 포트폴리오 수익률 계산
        r_stock = self.df['r_stock'].iloc[self.current_idx]
        r_bond = self.df['r_bond'].iloc[self.current_idx]
        portfolio_return = self.w_stock * r_stock + self.w_bond * r_bond
        
        # 4. 포트폴리오 가치 업데이트
        self.portfolio_value *= (1.0 + portfolio_return)
        
        # 5. Reward 계산
        reward = self._calculate_reward(portfolio_return)
        
        # 6. Step 및 인덱스 증가
        self.current_step += 1
        self.current_idx += 1
        
        # 7. 종료 조건 확인
        terminated = (self.current_step >= self.episode_length)
        truncated = False  # 시간 제한 없음
        
        # 8. 다음 상태 생성 (종료되지 않았을 때만)
        if not terminated:
            next_state = self._get_state()
        else:
            # 종료 시 더미 상태 (사용되지 않음)
            next_state = np.zeros(11, dtype=np.float32)
        
        # 9. 추가 정보
        info = {
            'portfolio_value': self.portfolio_value,
            'w_stock': self.w_stock,
            'w_bond': self.w_bond,
            'portfolio_return': portfolio_return,
            'current_date': self.df.index[self.current_idx - 1]
        }
        
        return next_state, reward, terminated, truncated, info
