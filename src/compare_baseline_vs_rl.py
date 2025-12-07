"""
Baseline 전략 vs RL 모델 종합 비교

Test 데이터에서 전통적인 Baseline 전략들과 
RL 모델들(최고 성능, 평균 앙상블)을 비교 분석합니다.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.experiments.eval_hyperparams import evaluate_model_on_dataset, load_model
from src.utils.data_loader import load_sp500_data, load_dgs10_data, preprocess_data, split_data
from src.utils.metrics import calculate_all_metrics
from src.config import DATA_START_DATE, DATA_END_DATE, PLOTS_DIR


def evaluate_baseline(df, weights, rebalance_monthly=False):
    """
    Baseline 전략 평가
    
    Args:
        df: 데이터프레임
        weights: (stock_weight, bond_weight) 튜플
        rebalance_monthly: 월별 리밸런싱 여부
    """
    stock_weight, bond_weight = weights
    returns = []
    prev_month = None
    
    # 리밸런싱이 필요한 경우 현재 비중 추적
    if rebalance_monthly:
        current_stock_weight = stock_weight
        current_bond_weight = bond_weight
    
    for i in range(len(df)):
        if rebalance_monthly:
            # 월초 리밸런싱
            current_month = df.index[i].month
            if prev_month is not None and current_month != prev_month:
                current_stock_weight = stock_weight
                current_bond_weight = bond_weight
            prev_month = current_month
            
            # 포트폴리오 수익률
            r_portfolio = current_stock_weight * df['r_stock'].iloc[i] + \
                         current_bond_weight * df['r_bond'].iloc[i]
            
            # 수익률 반영 후 비중 자동 변화 (리밸런싱 전까지)
            portfolio_value = 1.0
            stock_value = current_stock_weight * (1 + df['r_stock'].iloc[i])
            bond_value = current_bond_weight * (1 + df['r_bond'].iloc[i])
            new_portfolio_value = stock_value + bond_value
            
            current_stock_weight = stock_value / new_portfolio_value
            current_bond_weight = bond_value / new_portfolio_value
        else:
            # Buy-and-hold (리밸런싱 없음)
            r_portfolio = stock_weight * df['r_stock'].iloc[i] + \
                         bond_weight * df['r_bond'].iloc[i]
        
        returns.append(r_portfolio)
    
    metrics = calculate_all_metrics(returns)
    metrics['returns'] = np.array(returns)
    return metrics


def calculate_cumulative_returns(returns):
    """누적 수익률 계산"""
    cumulative = np.zeros(len(returns) + 1)
    cumulative[0] = 0.0
    
    for i, r in enumerate(returns):
        cumulative[i + 1] = (1 + cumulative[i] / 100) * (1 + r) - 1
        cumulative[i + 1] *= 100
    
    return cumulative


def create_ensemble_model(model_paths, model_type, test_df):
    """
    앙상블 모델 평가 (여러 모델의 평균 예측)
    
    Args:
        model_paths: 모델 경로 리스트
        model_type: 'dqn' 또는 'pg'
        test_df: 테스트 데이터
        
    Returns:
        앙상블 모델의 성능 지표
    """
    import torch
    
    # 모든 모델 로드
    models = [load_model(path, model_type) for path in model_paths]
    
    # 초기 상태
    w_stock = 0.5
    w_bond = 0.5
    returns = []
    prev_month = None
    
    action_values = np.linspace(-1.0, 1.0, 41)
    
    with torch.no_grad():
        for i in range(5, len(test_df)):
            # 상태 생성
            r_stock_history = test_df['r_stock'].iloc[i-4:i+1].values
            r_bond_history = test_df['r_bond'].iloc[i-4:i+1].values
            
            state = np.concatenate([
                r_stock_history,
                r_bond_history,
                [w_stock]
            ]).astype(np.float32)
            
            # 각 모델의 예측
            delta_w_predictions = []
            
            for model in models:
                if model_type == 'dqn':
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                else:  # pg
                    state_tensor = torch.from_numpy(state).float()
                    action_probs = model(state_tensor)
                    action = action_probs.argmax().item()
                
                delta_w = action_values[action]
                delta_w_predictions.append(delta_w)
            
            # 앙상블: 평균
            avg_delta_w = np.mean(delta_w_predictions)
            
            # 월초 리밸런싱
            current_month = test_df.index[i].month
            if prev_month is not None and current_month != prev_month:
                w_stock = np.clip(w_stock + avg_delta_w, 0.0, 1.0)
                w_bond = 1.0 - w_stock
            prev_month = current_month
            
            # 포트폴리오 수익률
            r_portfolio = w_stock * test_df['r_stock'].iloc[i] + \
                         w_bond * test_df['r_bond'].iloc[i]
            returns.append(r_portfolio)
    
    metrics = calculate_all_metrics(returns)
    metrics['returns'] = np.array(returns)
    return metrics


def main(dqn_pattern='dqn_lambda10.0_eps50_seed*.pt',
         pg_pattern='pg_lambda10.0_traj5_seed*.pt'):
    """
    메인 함수
    
    Args:
        dqn_pattern: DQN 모델 파일명 패턴 (기본값: 최적 하이퍼파라미터)
        pg_pattern: PG 모델 파일명 패턴 (기본값: 최적 하이퍼파라미터)
    """
    print("=" * 70)
    print("Baseline vs RL Models - Comprehensive Comparison")
    print("=" * 70)
    
    # ========================================================================
    # 데이터 로딩
    # ========================================================================
    print("\n[Step 1/4] Loading data...")
    print("-" * 70)
    
    sp500_df = load_sp500_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    dgs10_df = load_dgs10_data()
    df = preprocess_data(sp500_df, dgs10_df)
    train_df, val_df, test_df = split_data(df)
    
    print(f"✓ Test: {len(test_df)} days ({test_df.index[0].date()} ~ {test_df.index[-1].date()})")
    
    # ========================================================================
    # Baseline 전략 평가
    # ========================================================================
    print("\n[Step 2/4] Evaluating Baseline strategies...")
    print("-" * 70)
    
    baselines = {
        '100% Stock': evaluate_baseline(test_df, (1.0, 0.0)),
        '100% Bond': evaluate_baseline(test_df, (0.0, 1.0)),
        '60/40 (Monthly Rebal)': evaluate_baseline(test_df, (0.6, 0.4), rebalance_monthly=True),
    }
    
    for name, metrics in baselines.items():
        print(f"  {name:25s}: Sharpe={metrics['sharpe_ratio']:.4f}, "
              f"Return={metrics['annualized_return']:.2%}")
    
    # ========================================================================
    # RL 모델 평가
    # ========================================================================
    print("\n[Step 3/4] Evaluating RL models...")
    print("-" * 70)
    
    # DQN 모델들
    dqn_dir = os.path.join(project_root, 'results', 'models', 'dqn_randomseed')
    dqn_models = sorted(glob.glob(os.path.join(dqn_dir, dqn_pattern)))
    
    dqn_results = {}
    for model_path in dqn_models:
        seed = int(os.path.basename(model_path).split('seed')[1].split('_')[0])
        model = load_model(model_path, 'dqn')
        metrics = evaluate_model_on_dataset(model, test_df, 'dqn')
        dqn_results[seed] = metrics
    
    # DQN 최고 모델
    best_dqn_seed = max(dqn_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
    best_dqn_metrics = dqn_results[best_dqn_seed]
    print(f"  DQN Best (seed {best_dqn_seed}):    Sharpe={best_dqn_metrics['sharpe_ratio']:.4f}")
    
    # DQN 앙상블
    print("  DQN Ensemble:          Computing...")
    dqn_ensemble_metrics = create_ensemble_model(dqn_models, 'dqn', test_df)
    print(f"                         Sharpe={dqn_ensemble_metrics['sharpe_ratio']:.4f}")
    
    # PG 모델들
    pg_dir = os.path.join(project_root, 'results', 'models', 'pg_randomseed')
    pg_models = sorted(glob.glob(os.path.join(pg_dir, pg_pattern)))
    
    pg_results = {}
    for model_path in pg_models:
        seed = int(os.path.basename(model_path).split('seed')[1].split('_')[0])
        model = load_model(model_path, 'pg')
        metrics = evaluate_model_on_dataset(model, test_df, 'pg')
        pg_results[seed] = metrics
    
    # PG 최고 모델
    best_pg_seed = max(pg_results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
    best_pg_metrics = pg_results[best_pg_seed]
    print(f"  PG Best (seed {best_pg_seed}):      Sharpe={best_pg_metrics['sharpe_ratio']:.4f}")
    
    # PG 앙상블
    print("  PG Ensemble:           Computing...")
    pg_ensemble_metrics = create_ensemble_model(pg_models, 'pg', test_df)
    print(f"                         Sharpe={pg_ensemble_metrics['sharpe_ratio']:.4f}")
    
    # ========================================================================
    # 결과 정리
    # ========================================================================
    print("\n[Step 4/4] Creating visualizations...")
    print("-" * 70)
    
    all_strategies = {
        # Baselines
        '100% Stock': baselines['100% Stock'],
        '100% Bond': baselines['100% Bond'],
        '60/40 (Monthly Rebal)': baselines['60/40 (Monthly Rebal)'],
        # RL Models
        f'DQN Best (seed {best_dqn_seed})': best_dqn_metrics,
        'DQN Ensemble': dqn_ensemble_metrics,
        f'PG Best (seed {best_pg_seed})': best_pg_metrics,
        'PG Ensemble': pg_ensemble_metrics,
    }
    
    # 누적 수익률 그래프
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colors = {
        '100% Stock': '#e74c3c',
        '100% Bond': '#3498db',
        '60/40 (Monthly Rebal)': '#2ecc71',
        f'DQN Best (seed {best_dqn_seed})': '#f39c12',
        'DQN Ensemble': '#d35400',
        f'PG Best (seed {best_pg_seed})': '#9b59b6',
        'PG Ensemble': '#8e44ad',
    }
    
    linestyles = {
        '100% Stock': '-',
        '100% Bond': '-',
        '60/40 (Monthly Rebal)': '-',
        f'DQN Best (seed {best_dqn_seed})': '--',
        'DQN Ensemble': '-',
        f'PG Best (seed {best_pg_seed})': '--',
        'PG Ensemble': '-',
    }
    
    linewidths = {
        '100% Stock': 2,
        '100% Bond': 2,
        '60/40 (Monthly Rebal)': 3,
        f'DQN Best (seed {best_dqn_seed})': 2,
        'DQN Ensemble': 3,
        f'PG Best (seed {best_pg_seed})': 2,
        'PG Ensemble': 3,
    }
    
    for name, metrics in all_strategies.items():
        cumulative = calculate_cumulative_returns(metrics['returns'])
        dates = [test_df.index[0]] + list(test_df.index[:len(cumulative)-1])
        
        sharpe = metrics['sharpe_ratio']
        final_return = cumulative[-1]
        
        label = f"{name} (Sharpe={sharpe:.3f}, Final={final_return:.1f}%)"
        
        ax.plot(dates, cumulative,
                label=label,
                color=colors.get(name, 'gray'),
                linestyle=linestyles.get(name, '-'),
                linewidth=linewidths.get(name, 2),
                alpha=0.8)
    
    # 그래프 꾸미기
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=13, fontweight='bold')
    ax.set_title('Baseline Strategies vs RL Models - Test Period (2008-2025)',
                fontsize=15, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filepath = os.path.join(PLOTS_DIR, 'baseline_vs_rl_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: baseline_vs_rl_comparison.png")
    
    # ========================================================================
    # 결과 반환
    # ========================================================================
    return {
        'baselines': baselines,
        'dqn_best': (best_dqn_seed, best_dqn_metrics),
        'dqn_ensemble': dqn_ensemble_metrics,
        'pg_best': (best_pg_seed, best_pg_metrics),
        'pg_ensemble': pg_ensemble_metrics,
        'all_strategies': all_strategies
    }


if __name__ == '__main__':
    results = main()
    
    print("\n" + "=" * 70)
    print("Comparison Complete!")
    print("=" * 70)
