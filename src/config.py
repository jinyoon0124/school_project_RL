"""
Global Configuration for Portfolio RL Project

모든 하이퍼파라미터와 설정을 중앙에서 관리합니다.
DQN, Policy Gradient 모두 이 파일을 참조합니다.
"""

# ============================================================================
# 데이터 설정
# ============================================================================
DATA_START_DATE = '1962-01-01'  # 데이터 시작 날짜
DATA_END_DATE = '2025-11-30'    # 데이터 종료 날짜

# Train/Val/Test 분할 기준
TRAIN_END_DATE = '1999-12-31'
VAL_START_DATE = '2000-01-01'
VAL_END_DATE = '2007-12-31'
TEST_START_DATE = '2008-01-01'


# ============================================================================
# 환경 설정 (PortfolioEnv)
# ============================================================================
LAMBDA_RISK = 1.0          # Sharpe proxy 위험 회피 계수
EPISODE_YEARS = 10         # 에피소드 길이 (년 단위)


# ============================================================================
# 실험 설정
# ============================================================================
RANDOM_SEEDS = [42, 123, 456, 789, 1024]  # 실험용 random seeds (design.md 6.2)
DEFAULT_SEED = 42                          # 기본 seed


# ============================================================================
# DQN 하이퍼파라미터 (design.md 5.1 참조)
# ============================================================================
DQN_CONFIG = {
    'learning_rate': 0.0005,
    'gamma': 0.98,
    'batch_size': 32,
    'buffer_limit': 50000,
    'min_buffer_size': 2000,
    'epsilon_start': 0.08,
    'epsilon_end': 0.01,
    'epsilon_decay_episodes': 100,
    'target_update_freq': 10,
    'train_iterations_per_step': 10,
    'total_episodes': 1000
}


# ============================================================================
# Policy Gradient 하이퍼파라미터 (design.md 5.2 참조)
# ============================================================================
PG_CONFIG = {
    'learning_rate': 0.0005,
    'gamma': 0.98,
    'num_trajectories': 5,      # Batch size (trajectories per update)
    'num_iterations': 1000,     # Total iterations
    'network_hidden_size': 128  # Hidden layer size
}


# ============================================================================
# 경로 설정
# ============================================================================
import os

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 데이터 경로
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 결과 경로
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


# ============================================================================
# 유틸리티 함수
# ============================================================================
def create_result_directories():
    """결과 저장 디렉토리 생성"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def get_model_filename(algorithm, seed, timestamp=None):
    """
    모델 파일명 생성 (design.md 8.2 참조)
    
    Args:
        algorithm: 'dqn' or 'pg'
        seed: random seed
        timestamp: 타임스탬프 (None이면 자동 생성)
    
    Returns:
        파일명 (예: 'dqn_lr0.0005_bs32_seed42_20241204_151018.pt')
    """
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if algorithm == 'dqn':
        lr = DQN_CONFIG['learning_rate']
        bs = DQN_CONFIG['batch_size']
        return f"dqn_lr{lr}_bs{bs}_seed{seed}_{timestamp}.pt"
    
    elif algorithm == 'pg':
        lr = PG_CONFIG['learning_rate']
        batch = PG_CONFIG['num_trajectories']
        return f"pg_lr{lr}_batch{batch}_seed{seed}_{timestamp}.pt"
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_log_filename(algorithm, seed, timestamp=None):
    """
    로그 파일명 생성
    
    Args:
        algorithm: 'dqn' or 'pg'
        seed: random seed
        timestamp: 타임스탬프 (None이면 자동 생성)
    
    Returns:
        파일명 (예: 'dqn_training_log_seed42_20241204_151018.json')
    """
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{algorithm}_training_log_seed{seed}_{timestamp}.json"


# ============================================================================
# 설정 출력 (디버깅용)
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("Portfolio RL Project Configuration")
    print("=" * 70)
    
    print("\n[Data Settings]")
    print(f"  Start date: {DATA_START_DATE}")
    print(f"  End date:   {DATA_END_DATE or 'Latest available'}")
    print(f"  Train:      ~ {TRAIN_END_DATE}")
    print(f"  Val:        {VAL_START_DATE} ~ {VAL_END_DATE}")
    print(f"  Test:       {TEST_START_DATE} ~")
    
    print("\n[Environment Settings]")
    print(f"  Lambda risk:     {LAMBDA_RISK}")
    print(f"  Episode years:   {EPISODE_YEARS}")
    
    print("\n[Experiment Settings]")
    print(f"  Random seeds:    {RANDOM_SEEDS}")
    print(f"  Default seed:    {DEFAULT_SEED}")
    
    print("\n[DQN Hyperparameters]")
    for key, value in DQN_CONFIG.items():
        print(f"  {key:30s}: {value}")
    
    print("\n[PG Hyperparameters]")
    for key, value in PG_CONFIG.items():
        print(f"  {key:30s}: {value}")
    
    print("\n[Paths]")
    print(f"  Project root:    {PROJECT_ROOT}")
    print(f"  Data dir:        {DATA_DIR}")
    print(f"  Results dir:     {RESULTS_DIR}")
    
    print("\n" + "=" * 70)
