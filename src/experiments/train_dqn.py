"""
DQN Training Script for Portfolio Management

이 스크립트는 DQN 에이전트를 학습시키고 결과를 저장합니다.
ex011의 main() 함수와 유사한 역할을 합니다.
"""

import os
import sys
import json
import torch

# 프로젝트 루트를 Python path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Config import
from src.config import (
    DATA_START_DATE, DATA_END_DATE,
    LAMBDA_RISK, EPISODE_YEARS,
    DEFAULT_SEED,
    create_result_directories,
    get_model_filename, get_log_filename,
    MODELS_DIR, LOGS_DIR
)

# 모듈 import
from src.agents.dqn_agent import run_dqn_training
from src.env.portfolio_env import PortfolioEnv
from src.utils.data_loader import load_sp500_data, load_dgs10_data, preprocess_data, split_data


def main():
    """
    DQN 학습 메인 함수
    
    역할:
    1. 데이터 로딩 및 전처리
    2. 환경 생성
    3. 학습 실행
    4. 모델 및 로그 저장
    """
    
    print("=" * 70)
    print("DQN Training for Portfolio Management")
    print("=" * 70)
    
    # ========================================================================
    # 1. 데이터 로딩 및 전처리
    # ========================================================================
    print("\n[Step 1/4] Loading and preprocessing data...")
    print("-" * 70)
    
    # Config에서 날짜 가져오기
    sp500 = load_sp500_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    dgs10 = load_dgs10_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    
    # 수익률 계산 및 병합
    data = preprocess_data(sp500, dgs10)
    
    # Train/Val/Test 분할
    train_df, val_df, test_df = split_data(data)
    
    print(f"✓ Data loaded successfully!")
    print(f"  - Train: {len(train_df)} days")
    print(f"  - Val:   {len(val_df)} days")
    print(f"  - Test:  {len(test_df)} days")
    
    # ========================================================================
    # 2. 환경 생성
    # ========================================================================
    print("\n[Step 2/4] Creating training environment...")
    print("-" * 70)
    
    # Config에서 환경 파라미터 가져오기
    train_env = PortfolioEnv(
        df=train_df,
        lambda_risk=LAMBDA_RISK,
        episode_years=EPISODE_YEARS
    )
    
    print(f"✓ Environment created!")
    print(f"  - State space: {train_env.observation_space.shape}")
    print(f"  - Action space: {train_env.action_space.n} actions")
    print(f"  - Episode length: ~{train_env.episode_length} steps")
    print(f"  - Lambda risk: {LAMBDA_RISK}")
    
    # ========================================================================
    # 3. 학습 실행
    # ========================================================================
    print("\n[Step 3/4] Starting DQN training...")
    print("-" * 70)
    
    # Config에서 seed 가져오기
    seed = DEFAULT_SEED
    print(f"Using random seed: {seed}")
    
    # DQN 학습 실행
    q, q_target, training_log = run_dqn_training(
        env=train_env,
        seed=seed,
        verbose=True
    )
    
    print("\n✓ Training completed!")
    
    # ========================================================================
    # 4. 모델 및 로그 저장
    # ========================================================================
    print("\n[Step 4/4] Saving model and training log...")
    print("-" * 70)
    
    # 결과 디렉토리 생성 (config 함수 사용)
    create_result_directories()
    
    # 타임스탬프 생성
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Config 함수로 파일명 생성
    model_filename = get_model_filename('dqn', seed, timestamp)
    log_filename = get_log_filename('dqn', seed, timestamp)
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    log_path = os.path.join(LOGS_DIR, log_filename)
    
    # 모델 저장 (체크포인트 형식)
    from src.agents.dqn_agent import learning_rate, batch_size
    checkpoint = {
        'model_state_dict': q.state_dict(),
        'target_state_dict': q_target.state_dict(),
        'config': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'lambda_risk': LAMBDA_RISK,
            'episode_years': EPISODE_YEARS
        },
        'seed': seed,
        'timestamp': timestamp,
        'training_log': training_log
    }
    
    torch.save(checkpoint, model_path)
    print(f"✓ Model saved: {model_filename}")
    
    # 학습 로그 JSON 저장 (하이퍼파라미터 포함)
    from src.agents.dqn_agent import (
        gamma, buffer_limit, min_buffer_size,
        epsilon_start, epsilon_end, epsilon_decay_episodes,
        target_update_freq, train_iterations_per_step, total_episodes
    )
    
    log_data = {
        'hyperparameters': {
            'algorithm': 'DQN',
            'learning_rate': learning_rate,
            'gamma': gamma,
            'batch_size': batch_size,
            'buffer_limit': buffer_limit,
            'min_buffer_size': min_buffer_size,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay_episodes': epsilon_decay_episodes,
            'target_update_freq': target_update_freq,
            'train_iterations_per_step': train_iterations_per_step,
            'total_episodes': total_episodes,
            'lambda_risk': LAMBDA_RISK,
            'episode_years': EPISODE_YEARS,
            'seed': seed,
            'timestamp': timestamp
        },
        'training_history': training_log
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"✓ Training log saved: {log_filename}")
    
    # ========================================================================
    # 완료 메시지
    # ========================================================================
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - Model: {model_path}")
    print(f"  - Log:   {log_path}")
    print("\nNext steps:")
    print("  1. Evaluate the model on validation/test data")
    print("  2. Compare with baseline strategies")
    print("  3. Visualize learning curves and portfolio performance")
    print("=" * 70)


if __name__ == '__main__':
    """
    스크립트 실행 진입점
    
    실행 방법:
        python src/experiments/train_dqn.py
    
    또는 프로젝트 루트에서:
        python -m src.experiments.train_dqn
    """
    main()
