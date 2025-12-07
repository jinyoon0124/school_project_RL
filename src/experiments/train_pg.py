"""
Policy Gradient Training Script for Portfolio Management
"""

import os
import sys
import json
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.utils.data_loader import load_sp500_data, load_dgs10_data, preprocess_data, split_data
from src.env.portfolio_env import PortfolioEnv
from src.agents.pg_agent import run_pg_training
from src.config import (
    DATA_START_DATE, DATA_END_DATE,
    TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE, TEST_START_DATE,
    LAMBDA_RISK, EPISODE_YEARS,
    DEFAULT_SEED,
    create_result_directories, get_model_filename, get_log_filename
)

import torch


def main():
    """
    Policy Gradient 학습 메인 함수
    
    단계:
    1. 데이터 로딩 및 전처리
    2. 학습 환경 생성
    3. PG 학습 실행
    4. 모델 및 로그 저장
    """
    print("=" * 70)
    print("Policy Gradient Training for Portfolio Management")
    print("=" * 70)
    
    # ========================================================================
    # Step 1: 데이터 로딩 및 전처리
    # ========================================================================
    print("\n[Step 1/4] Loading and preprocessing data...")
    print("-" * 70)
    
    # S&P500 데이터 다운로드
    sp500_df = load_sp500_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    
    # DGS10 (10년 국채 금리) 데이터 로드
    dgs10_df = load_dgs10_data()
    
    # 데이터 전처리 (수익률 계산 및 병합)
    df = preprocess_data(sp500_df, dgs10_df)
    
    # Train/Val/Test 분할
    train_df, val_df, test_df = split_data(df)
    
    print(f"✓ Data loaded successfully!")
    print(f"  - Train: {len(train_df)} days")
    print(f"  - Val:   {len(val_df)} days")
    print(f"  - Test:  {len(test_df)} days")
    
    # ========================================================================
    # Step 2: 학습 환경 생성
    # ========================================================================
    print("\n[Step 2/4] Creating training environment...")
    print("-" * 70)
    
    train_env = PortfolioEnv(
        df=train_df,
        lambda_risk=LAMBDA_RISK,
        episode_years=EPISODE_YEARS
    )
    
    print(f"✓ Environment created!")
    print(f"  - State space: {train_env.observation_space.shape}")
    print(f"  - Action space: {train_env.action_space.n} actions")
    print(f"  - Episode length: ~{int(EPISODE_YEARS * 252)} steps")
    print(f"  - Lambda risk: {LAMBDA_RISK}")
    
    # ========================================================================
    # Step 3: Policy Gradient 학습
    # ========================================================================
    print("\n[Step 3/4] Starting Policy Gradient training...")
    print("-" * 70)
    
    # Random seed 설정
    seed = DEFAULT_SEED
    print(f"Using random seed: {seed}")
    
    # PG 학습 실행
    policy, value, training_log = run_pg_training(
        env=train_env,
        seed=seed,
        verbose=True
    )
    
    print("\n✓ Training completed!")
    
    # ========================================================================
    # Step 4: 모델 및 로그 저장
    # ========================================================================
    print("\n[Step 4/4] Saving model and training log...")
    print("-" * 70)
    
    # 결과 디렉토리 생성
    create_result_directories()
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 모델 파일명 생성
    model_filename = get_model_filename('pg', seed, timestamp)
    model_path = os.path.join(project_root, 'results', 'models', model_filename)
    
    # 모델 저장 (Policy와 Value network 모두)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value.state_dict(),
        'config': {
            'seed': seed,
            'lambda_risk': LAMBDA_RISK,
            'episode_years': EPISODE_YEARS,
            'timestamp': timestamp
        }
    }, model_path)
    
    print(f"✓ Model saved: {model_filename}")
    
    # 로그 파일명 생성
    log_filename = get_log_filename('pg', seed, timestamp)
    log_path = os.path.join(project_root, 'results', 'logs', log_filename)
    
    # 학습 로그 저장 (하이퍼파라미터 포함)
    from src.agents.pg_agent import learning_rate, gamma, num_trajectories, num_iterations, hidden_size
    
    log_data = {
        'hyperparameters': {
            'algorithm': 'Policy Gradient (REINFORCE)',
            'learning_rate': learning_rate,
            'gamma': gamma,
            'num_trajectories': num_trajectories,
            'num_iterations': num_iterations,
            'network_hidden_size': hidden_size,
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
    print(f"\nNext steps:")
    print(f"  1. Evaluate the model on validation/test data")
    print(f"  2. Compare with DQN and baseline strategies")
    print(f"  3. Visualize learning curves and portfolio performance")
    print("=" * 70)


if __name__ == '__main__':
    main()
