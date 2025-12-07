"""
Hyperparameter Evaluation Utilities

Validation 데이터를 사용한 모델 평가 및 비교 도구
"""

import os
import sys
import numpy as np
import torch

# 프로젝트 루트를 path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.agents.dqn_agent import Qnet
from src.agents.pg_agent import PolicyNet
from src.utils.metrics import calculate_all_metrics


def evaluate_model_on_dataset(model, df, model_type='dqn'):
    """
    학습된 모델을 특정 데이터셋에서 평가 (환경 없이 직접 계산)
    
    Args:
        model: 학습된 모델 (Qnet 또는 PolicyNet)
        df: 평가할 데이터프레임 (train/val/test)
        model_type: 'dqn' 또는 'pg'
        
    Returns:
        dict: {
            'sharpe_ratio': float,
            'annualized_return': float,
            'volatility': float,
            'max_drawdown': float,
            'cumulative_return': float,
            'returns': np.array (일별 수익률)
        }
    """
    model.eval()  # 평가 모드
    
    # 초기 상태
    w_stock = 0.5
    w_bond = 0.5
    returns = []
    prev_month = None
    
    # 액션 값 배열 (환경과 동일)
    action_values = np.linspace(-1.0, 1.0, 41)
    
    # 최소 5일 이후부터 시작 (과거 5일 데이터 필요)
    with torch.no_grad():
        for i in range(5, len(df)):
            # 1. 상태 생성
            r_stock_history = df['r_stock'].iloc[i-4:i+1].values  # 5개
            r_bond_history = df['r_bond'].iloc[i-4:i+1].values    # 5개
            
            state = np.concatenate([
                r_stock_history,  # 5개
                r_bond_history,   # 5개
                [w_stock]         # 1개
            ]).astype(np.float32)
            
            # 2. 모델로 액션 선택
            if model_type == 'dqn':
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            else:  # pg
                # PG: 확률이 가장 높은 액션 선택 (deterministic evaluation)
                state_tensor = torch.from_numpy(state).float()
                action_probs = model(state_tensor)
                action = action_probs.argmax().item()
            
            # 3. 액션을 delta_w로 변환
            delta_w = action_values[action]
            
            # 4. 월초 리밸런싱
            current_month = df.index[i].month
            if prev_month is not None and current_month != prev_month:
                # 월이 바뀜 → 리밸런싱
                w_stock = np.clip(w_stock + delta_w, 0.0, 1.0)
                w_bond = 1.0 - w_stock
            prev_month = current_month
            
            # 5. 포트폴리오 수익률 계산
            r_portfolio = w_stock * df['r_stock'].iloc[i] + w_bond * df['r_bond'].iloc[i]
            returns.append(r_portfolio)
    
    # 6. 성능 지표 계산
    metrics = calculate_all_metrics(returns)
    
    # 7. 수익률 배열 추가
    metrics['returns'] = np.array(returns)
    
    return metrics


def load_model(model_path, model_type='dqn'):
    """
    모델 파일 로드
    
    Args:
        model_path: 모델 파일 경로
        model_type: 'dqn' 또는 'pg'
        
    Returns:
        model: 로드된 모델
    """
    checkpoint = torch.load(model_path, weights_only=False)
    
    if model_type == 'dqn':
        model = Qnet()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # pg
        model = PolicyNet()
        model.load_state_dict(checkpoint['policy_state_dict'])
    
    model.eval()
    return model


def compare_models_on_validation(model_paths, val_df, model_type='dqn'):
    """
    여러 모델을 Validation 데이터에서 비교
    
    Args:
        model_paths: 모델 파일 경로 리스트 또는 딕셔너리 {이름: 경로}
        val_df: Validation 데이터프레임
        model_type: 'dqn' 또는 'pg'
        
    Returns:
        results: {모델명: metrics} 딕셔너리
        best_model: (모델명, 경로, Sharpe ratio) 튜플
    """
    results = {}
    
    # 리스트를 딕셔너리로 변환
    if isinstance(model_paths, list):
        model_paths = {os.path.basename(path): path for path in model_paths}
    
    print("=" * 70)
    print(f"Comparing {len(model_paths)} models on Validation set")
    print("=" * 70)
    
    for model_name, model_path in model_paths.items():
        print(f"\nEvaluating: {model_name}")
        print("-" * 70)
        
        # 모델 로드
        model = load_model(model_path, model_type)
        
        # Validation 평가
        metrics = evaluate_model_on_dataset(model, val_df, model_type)
        
        # 결과 저장
        results[model_name] = metrics
        
        # 주요 지표 출력
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>8.4f}")
        print(f"  Annualized Return:  {metrics['annualized_return']:>8.2%}")
        print(f"  Volatility:         {metrics['volatility']:>8.2%}")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:>8.2%}")
    
    # 최고 성능 모델 찾기
    best_model_name = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    best_model_path = model_paths[best_model_name[0]]
    
    print("\n" + "=" * 70)
    print(f"🏆 Best Model: {best_model_name[0]}")
    print(f"   Sharpe Ratio: {best_model_name[1]['sharpe_ratio']:.4f}")
    print("=" * 70)
    
    return results, (best_model_name[0], best_model_path, best_model_name[1]['sharpe_ratio'])


def print_comparison_table(results, title="Model Comparison"):
    """
    모델 비교 테이블 출력
    
    Args:
        results: {모델명: metrics} 딕셔너리
        title: 테이블 제목
    """
    print("\n" + "=" * 90)
    print(f"{title}")
    print("=" * 90)
    
    # 헤더
    print(f"{'Model':<30} {'Sharpe':<10} {'Ann.Ret':<10} {'Vol':<10} {'MaxDD':<10}")
    print("-" * 90)
    
    # 각 모델 출력 (Sharpe ratio 기준 정렬)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<30} "
              f"{metrics['sharpe_ratio']:<10.4f} "
              f"{metrics['annualized_return']:<10.2%} "
              f"{metrics['volatility']:<10.2%} "
              f"{metrics['max_drawdown']:<10.2%}")
    
    print("=" * 90)


if __name__ == '__main__':
    """
    테스트 코드
    """
    from src.utils.data_loader import load_sp500_data, load_dgs10_data, preprocess_data, split_data
    from src.config import DATA_START_DATE, DATA_END_DATE
    
    print("Testing eval_hyperparams.py")
    print("=" * 70)
    
    # 데이터 로딩
    print("\nLoading data...")
    sp500_df = load_sp500_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    dgs10_df = load_dgs10_data()
    df = preprocess_data(sp500_df, dgs10_df)
    train_df, val_df, test_df = split_data(df)
    
    print(f"✓ Validation: {len(val_df)} days ({val_df.index[0].date()} ~ {val_df.index[-1].date()})")
    
    # 모델 찾기
    models_dir = os.path.join(project_root, 'results', 'models')
    
    if os.path.exists(models_dir):
        dqn_models = [f for f in os.listdir(models_dir) if f.startswith('dqn_') and f.endswith('.pt')]
        pg_models = [f for f in os.listdir(models_dir) if f.startswith('pg_') and f.endswith('.pt')]
        
        print(f"\nFound {len(dqn_models)} DQN models and {len(pg_models)} PG models")
        
        if dqn_models:
            print("\nTesting with first DQN model...")
            model_path = os.path.join(models_dir, dqn_models[0])
            model = load_model(model_path, 'dqn')
            metrics = evaluate_model_on_dataset(model, val_df, 'dqn')
            print(f"✓ Validation Sharpe: {metrics['sharpe_ratio']:.4f}")
    else:
        print(f"\n⚠️  No models directory found at {models_dir}")
        print("   Train some models first!")
