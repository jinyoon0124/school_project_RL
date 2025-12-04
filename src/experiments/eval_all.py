"""
All Strategies Evaluation

Baseline, DQN, PG 전략들의 성능을 평가하고 비교
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.utils.data_loader import load_sp500_data, load_dgs10_data, preprocess_data, split_data
from src.env.portfolio_env import PortfolioEnv
from src.utils.metrics import calculate_all_metrics, print_metrics
from src.agents.dqn_agent import Qnet
from src.agents.pg_agent import PolicyNet, ValueNet
from src.config import (
    DATA_START_DATE, DATA_END_DATE,
    LAMBDA_RISK, EPISODE_YEARS,
    create_result_directories
)

import torch


class BaselineStrategy:
    """
    Baseline 전략 클래스
    
    고정 비중 전략 (주식 100%, 채권 100%, 60/40 등)
    """
    
    def __init__(self, stock_weight, rebalance=False):
        """
        Args:
            stock_weight (float): 주식 비중 (0.0~1.0)
            rebalance (bool): 월초 리밸런싱 여부
        """
        self.target_stock_weight = stock_weight
        self.rebalance = rebalance
        
        # 목표 비중에 해당하는 액션 찾기
        # action_values = [-1.0, -0.95, ..., 0, ..., 0.95, 1.0]
        # 초기 w_stock = 0.5이므로, target - 0.5 = delta_w
        delta_w = stock_weight - 0.5
        
        # 가장 가까운 액션 찾기
        action_values = np.linspace(-1.0, 1.0, 41)
        self.action = np.argmin(np.abs(action_values - delta_w))
    
    def get_action(self, state):
        """
        액션 선택
        
        Args:
            state: 현재 상태 (사용하지 않음)
            
        Returns:
            int: 선택된 액션
        """
        if self.rebalance:
            # 리밸런싱: 항상 목표 비중으로 조정
            return self.action
        else:
            # 리밸런싱 없음: 액션 0 (변화 없음)
            return 20  # action_values[20] = 0.0


def evaluate_baseline_simple(stock_weight, df, rebalance=False):
    """
    Baseline 전략을 직접 계산 (환경 없이)
    
    Args:
        stock_weight (float): 주식 비중 (0.0~1.0)
        df (pd.DataFrame): 평가 데이터
        rebalance (bool): 월초 리밸런싱 여부
        
    Returns:
        dict: 성능 지표
    """
    w_stock = stock_weight
    w_bond = 1.0 - stock_weight
    
    returns = []
    prev_month = None
    
    for i in range(len(df)):
        # 포트폴리오 수익률 계산
        r_portfolio = w_stock * df['r_stock'].iloc[i] + w_bond * df['r_bond'].iloc[i]
        returns.append(r_portfolio)
        
        # 월초 리밸런싱 (월이 바뀌었을 때)
        if rebalance:
            current_month = df.index[i].month
            if prev_month is not None and current_month != prev_month:
                # 월이 바뀜 → 리밸런싱
                w_stock = stock_weight
                w_bond = 1.0 - stock_weight
            prev_month = current_month
    
    # 성능 지표 계산
    metrics = calculate_all_metrics(returns)
    
    return metrics


def compare_baselines(test_df):
    """
    Test 기간에서 Baseline 전략들 비교
    
    Args:
        test_df: Test 데이터
        
    Returns:
        dict: 각 전략의 성능 지표
    """
    print("=" * 70)
    print("Baseline Strategy Evaluation (Test Period)")
    print("=" * 70)
    print(f"Period: {test_df.index[0].date()} ~ {test_df.index[-1].date()}")
    print(f"Days: {len(test_df)}")
    print("=" * 70)
    
    # Baseline 전략 정의
    strategies = {
        '100% Stock': {'stock_weight': 1.0, 'rebalance': False},
        '100% Bond': {'stock_weight': 0.0, 'rebalance': False},
        '60/40 (Monthly Rebalance)': {'stock_weight': 0.6, 'rebalance': True}
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\n[{strategy_name}]")
        print("-" * 70)
        
        # 전략 평가
        metrics = evaluate_baseline_simple(
            stock_weight=params['stock_weight'],
            df=test_df,
            rebalance=params['rebalance']
        )
        
        # 결과 출력
        print_metrics(metrics, title=strategy_name)
        
        # 결과 저장
        results[strategy_name] = {
            'cumulative_return': metrics['cumulative_return'],
            'annualized_return': metrics['annualized_return'],
            'volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'num_periods': metrics['num_periods']
        }
    
    return results


def evaluate_model(model, test_df, model_type='dqn'):
    """
    학습된 모델을 Test 데이터에서 평가 (환경 없이 직접 계산)
    
    Args:
        model: 학습된 모델 (Qnet 또는 PolicyNet)
        test_df: Test 데이터프레임
        model_type: 'dqn' 또는 'pg'
        
    Returns:
        dict: 성능 지표
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
        for i in range(5, len(test_df)):
            # 1. 상태 생성 (환경의 _get_state()와 동일)
            r_stock_history = test_df['r_stock'].iloc[i-4:i+1].values  # 5개
            r_bond_history = test_df['r_bond'].iloc[i-4:i+1].values    # 5개
            
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
                action = model.act(state)
            
            # 3. 액션을 delta_w로 변환
            delta_w = action_values[action]
            
            # 4. 월초 리밸런싱 (환경과 동일)
            current_month = test_df.index[i].month
            if prev_month is not None and current_month != prev_month:
                # 월이 바뀜 → 리밸런싱
                w_stock = np.clip(w_stock + delta_w, 0.0, 1.0)
                w_bond = 1.0 - w_stock
            prev_month = current_month
            
            # 5. 포트폴리오 수익률 계산
            r_portfolio = w_stock * test_df['r_stock'].iloc[i] + w_bond * test_df['r_bond'].iloc[i]
            returns.append(r_portfolio)
    
    # 6. 성능 지표 계산
    metrics = calculate_all_metrics(returns)
    return metrics


def load_and_evaluate_dqn(model_path, test_df):
    """
    DQN 모델 로드 및 평가
    
    Args:
        model_path: 모델 파일 경로
        test_df: Test 데이터
        
    Returns:
        dict: 성능 지표
    """
    print("\n[DQN Model]")
    print("-" * 70)
    print(f"Loading model from: {model_path}")
    
    # 모델 로드
    checkpoint = torch.load(model_path)
    model = Qnet()
    model.load_state_dict(checkpoint['q_state_dict'])
    
    # 평가 (환경 없이 직접 계산)
    metrics = evaluate_model(model, test_df, model_type='dqn')
    
    # 결과 출력
    print_metrics(metrics, title="DQN Model")
    
    return {
        'cumulative_return': metrics['cumulative_return'],
        'annualized_return': metrics['annualized_return'],
        'volatility': metrics['volatility'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'num_periods': metrics['num_periods']
    }


def load_and_evaluate_pg(model_path, test_df):
    """
    PG 모델 로드 및 평가
    
    Args:
        model_path: 모델 파일 경로
        test_df: Test 데이터
        
    Returns:
        dict: 성능 지표
    """
    print("\n[Policy Gradient Model]")
    print("-" * 70)
    print(f"Loading model from: {model_path}")
    
    # 모델 로드
    checkpoint = torch.load(model_path)
    model = PolicyNet()
    model.load_state_dict(checkpoint['policy_state_dict'])
    
    # 평가 (환경 없이 직접 계산)
    metrics = evaluate_model(model, test_df, model_type='pg')
    
    # 결과 출력
    print_metrics(metrics, title="Policy Gradient Model")
    
    return {
        'cumulative_return': metrics['cumulative_return'],
        'annualized_return': metrics['annualized_return'],
        'volatility': metrics['volatility'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'num_periods': metrics['num_periods']
    }


def compare_all_strategies(baseline_results, dqn_results, pg_results):
    """
    모든 전략 비교 테이블 출력
    
    Args:
        baseline_results: Baseline 전략 결과
        dqn_results: DQN 결과
        pg_results: PG 결과
    """
    print("\n" + "=" * 70)
    print("All Strategies Comparison (Test Period)")
    print("=" * 70)
    
    # 모든 결과 합치기
    all_results = {**baseline_results, 'DQN': dqn_results, 'Policy Gradient': pg_results}
    
    # 테이블 헤더
    print(f"\n{'Strategy':<30} {'Cum.Ret':<12} {'Ann.Ret':<10} {'Vol':<10} {'Sharpe':<10} {'MaxDD':<10}")
    print("-" * 90)
    
    # 각 전략 출력
    for strategy_name, results in all_results.items():
        print(f"{strategy_name:<30} "
              f"{results['cumulative_return']:>10.2%}  "
              f"{results['annualized_return']:>8.2%}  "
              f"{results['volatility']:>8.2%}  "
              f"{results['sharpe_ratio']:>8.4f}  "
              f"{results['max_drawdown']:>8.2%}")
    
    print("=" * 90)
    
    # 최고 Sharpe ratio 찾기
    best_strategy = max(all_results.items(), key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n🏆 Best Strategy (Sharpe Ratio): {best_strategy[0]} ({best_strategy[1]['sharpe_ratio']:.4f})")
    
    return all_results


def find_best_model(model_type='dqn'):
    """
    학습 로그를 분석하여 가장 성능이 좋은 모델 찾기
    
    Args:
        model_type: 'dqn' 또는 'pg'
        
    Returns:
        str: 최고 성능 모델 파일명
    """
    logs_dir = os.path.join(project_root, 'results', 'logs')
    models_dir = os.path.join(project_root, 'results', 'models')
    
    # 로그 파일 찾기
    log_files = [f for f in os.listdir(logs_dir) if f.startswith(f'{model_type}_training_log')]
    
    if not log_files:
        raise FileNotFoundError(f"No {model_type.upper()} training logs found")
    
    best_model = None
    best_performance = -float('inf')
    
    for log_file in log_files:
        log_path = os.path.join(logs_dir, log_file)
        
        with open(log_path, 'r') as f:
            training_log = json.load(f)
        
        # 마지막 N개 에피소드의 평균 reward 계산
        if model_type == 'dqn':
            # DQN: 마지막 50 에피소드 평균
            recent_rewards = [episode['avg_reward'] for episode in training_log[-50:]]
        else:  # pg
            # PG: 마지막 10 iteration 평균
            recent_rewards = [iteration['mean_return'] for iteration in training_log[-10:]]
        
        avg_performance = np.mean(recent_rewards)
        
        # 최고 성능 모델 찾기
        if avg_performance > best_performance:
            best_performance = avg_performance
            # 로그 파일명에서 타임스탬프 추출
            timestamp = log_file.split('_')[-1].replace('.json', '')
            # 해당하는 모델 파일 찾기
            model_pattern = f"{model_type}_*_{timestamp}.pt"
            matching_models = [f for f in os.listdir(models_dir) if f.endswith(f'_{timestamp}.pt') and f.startswith(model_type)]
            if matching_models:
                best_model = matching_models[0]
    
    if best_model is None:
        raise FileNotFoundError(f"No matching {model_type.upper()} model found")
    
    print(f"Best {model_type.upper()} model: {best_model}")
    print(f"Average performance (last episodes): {best_performance:.4f}")
    
    return best_model


def save_results(results, filename='baseline_results.json'):
    """
    결과를 JSON 파일로 저장
    
    Args:
        results: 평가 결과 딕셔너리
        filename: 저장할 파일명
    """
    # 결과 디렉토리 생성
    create_result_directories()
    
    # 파일 경로
    filepath = os.path.join(project_root, 'results', 'logs', filename)
    
    # JSON 저장
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {filepath}")


def main():
    """
    메인 함수
    """
    print("=" * 70)
    print("All Strategies Evaluation (Baseline + DQN + PG)")
    print("=" * 70)
    
    # 데이터 로딩
    print("\n[Step 1/5] Loading data...")
    print("-" * 70)
    
    sp500_df = load_sp500_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    dgs10_df = load_dgs10_data()
    df = preprocess_data(sp500_df, dgs10_df)
    train_df, val_df, test_df = split_data(df)
    
    print("✓ Data loaded successfully!")
    print(f"  - Test: {len(test_df)} days ({test_df.index[0].date()} ~ {test_df.index[-1].date()})")
    
    # Baseline 평가 (Test 기간만)
    print("\n[Step 2/5] Evaluating baseline strategies on Test set...")
    print("-" * 70)
    
    baseline_results = compare_baselines(test_df)
    
    # DQN 평가
    print("\n[Step 3/5] Evaluating DQN model on Test set...")
    print("-" * 70)
    
    # 최고 성능 DQN 모델 찾기
    best_dqn_model = find_best_model('dqn')
    dqn_model_path = os.path.join(project_root, 'results', 'models', best_dqn_model)
    dqn_results = load_and_evaluate_dqn(dqn_model_path, test_df)
    
    # PG 평가
    print("\n[Step 4/5] Evaluating PG model on Test set...")
    print("-" * 70)
    
    # 최고 성능 PG 모델 찾기
    best_pg_model = find_best_model('pg')
    pg_model_path = os.path.join(project_root, 'results', 'models', best_pg_model)
    pg_results = load_and_evaluate_pg(pg_model_path, test_df)
    
    # 모든 전략 비교
    print("\n[Step 5/5] Comparing all strategies...")
    print("-" * 70)
    
    all_results = compare_all_strategies(baseline_results, dqn_results, pg_results)
    
    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(all_results, filename=f'all_strategies_results_{timestamp}.json')
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
