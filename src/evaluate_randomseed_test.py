"""
Test 데이터에서 Random Seed 모델들의 최종 성능 평가

DQN과 PG 모델들을 각각 다른 random seed로 학습한 모델들을 
Test 데이터셋에서 평가하고 성능 지표 및 신뢰구간을 계산합니다.
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime
from scipy import stats

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.experiments.eval_hyperparams import evaluate_model_on_dataset, load_model
from src.utils.data_loader import load_sp500_data, load_dgs10_data, preprocess_data, split_data
from src.config import DATA_START_DATE, DATA_END_DATE


def calculate_confidence_interval(values, confidence=0.95):
    """
    신뢰구간 계산 (t-분포 사용)
    
    Args:
        values: 측정값 리스트
        confidence: 신뢰수준 (기본값: 95%)
        
    Returns:
        (mean, lower_bound, upper_bound, std)
    """
    n = len(values)
    if n == 0:
        return 0, 0, 0, 0
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    if n == 1:
        return mean, mean, mean, 0
    
    # t-분포 사용 (샘플 크기가 작을 때)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * std / np.sqrt(n)
    
    return mean, mean - margin, mean + margin, std


def print_seed_results(seed_results, model_name):
    """
    Seed별 상세 결과 출력
    
    Args:
        seed_results: {seed: metrics} 딕셔너리
        model_name: 모델 이름 (DQN 또는 PG)
    """
    print(f"\n{model_name} Results by Seed:")
    print(f"{'Seed':<8} {'Sharpe':<10} {'Ann.Return':<12} {'Volatility':<12} {'Max DD':<10}")
    print("-" * 70)
    
    for seed in sorted(seed_results.keys()):
        r = seed_results[seed]
        print(f"{seed:<8} {r['sharpe']:<10.4f} {r['return']:<12.2%} "
              f"{r['volatility']:<12.2%} {r['max_dd']:<10.2%}")


def print_summary_statistics(sharpe_list, return_list, vol_list, dd_list, model_name):
    """
    요약 통계 출력
    
    Args:
        sharpe_list: Sharpe ratio 리스트
        return_list: 수익률 리스트
        vol_list: 변동성 리스트
        dd_list: Max Drawdown 리스트
        model_name: 모델 이름
    """
    if not sharpe_list:
        print(f"\n⚠️  No {model_name} models found!")
        return
    
    # 통계 계산
    sharpe_mean, sharpe_lower, sharpe_upper, sharpe_std = calculate_confidence_interval(sharpe_list)
    return_mean, return_lower, return_upper, return_std = calculate_confidence_interval(return_list)
    vol_mean, vol_lower, vol_upper, vol_std = calculate_confidence_interval(vol_list)
    dd_mean, dd_lower, dd_upper, dd_std = calculate_confidence_interval(dd_list)
    
    print(f"\n{model_name} Summary Statistics ({len(sharpe_list)} models):")
    print("=" * 70)
    print(f"Sharpe Ratio:")
    print(f"  Mean:  {sharpe_mean:>8.4f}")
    print(f"  Std:   {sharpe_std:>8.4f}")
    print(f"  95% CI: [{sharpe_lower:.4f}, {sharpe_upper:.4f}]")
    
    print(f"\nAnnualized Return:")
    print(f"  Mean:  {return_mean:>8.2%}")
    print(f"  Std:   {return_std:>8.2%}")
    print(f"  95% CI: [{return_lower:.2%}, {return_upper:.2%}]")
    
    print(f"\nVolatility:")
    print(f"  Mean:  {vol_mean:>8.2%}")
    print(f"  Std:   {vol_std:>8.2%}")
    print(f"  95% CI: [{vol_lower:.2%}, {vol_upper:.2%}]")
    
    print(f"\nMax Drawdown:")
    print(f"  Mean:  {dd_mean:>8.2%}")
    print(f"  Std:   {dd_std:>8.2%}")
    print(f"  95% CI: [{dd_lower:.2%}, {dd_upper:.2%}]")


def main(dqn_pattern='dqn_lambda10.0_eps50_seed*.pt',
         pg_pattern='pg_lambda10.0_traj5_seed*.pt'):
    """
    Test 평가 메인 함수
    
    Args:
        dqn_pattern: DQN 모델 파일명 패턴 (기본값: 최적 하이퍼파라미터)
        pg_pattern: PG 모델 파일명 패턴 (기본값: 최적 하이퍼파라미터)
    """
    print("=" * 70)
    print("Random Seed Models - Test Dataset Evaluation")
    print("=" * 70)
    
    # ========================================================================
    # 데이터 로딩
    # ========================================================================
    print("\n[Step 1/3] Loading data...")
    print("-" * 70)
    
    sp500_df = load_sp500_data(start_date=DATA_START_DATE, end_date=DATA_END_DATE)
    dgs10_df = load_dgs10_data()
    df = preprocess_data(sp500_df, dgs10_df)
    train_df, val_df, test_df = split_data(df)
    
    print(f"✓ Data loaded successfully!")
    print(f"  - Test: {len(test_df)} days ({test_df.index[0].date()} ~ {test_df.index[-1].date()})")
    
    # ========================================================================
    # DQN 모델 평가
    # ========================================================================
    print("\n[Step 2/3] Evaluating DQN models on Test data...")
    print("-" * 70)
    
    dqn_dir = os.path.join(project_root, 'results', 'models', 'dqn_randomseed')
    dqn_models = sorted(glob.glob(os.path.join(dqn_dir, dqn_pattern)))
    
    if not dqn_models:
        print(f"⚠️  No DQN models found in {dqn_dir}")
        dqn_sharpe_list = []
        dqn_return_list = []
        dqn_vol_list = []
        dqn_dd_list = []
        dqn_seed_results = {}
    else:
        print(f"Found {len(dqn_models)} DQN models")
        
        dqn_sharpe_list = []
        dqn_return_list = []
        dqn_vol_list = []
        dqn_dd_list = []
        dqn_seed_results = {}
        
        for model_path in dqn_models:
            model_name = os.path.basename(model_path)
            # Extract seed from filename (e.g., dqn_lambda10.0_eps50_seed42_timestamp.pt)
            seed = int(model_name.split('seed')[1].split('_')[0])
            
            print(f"  Evaluating seed={seed}...")
            
            model = load_model(model_path, 'dqn')
            metrics = evaluate_model_on_dataset(model, test_df, 'dqn')
            
            dqn_sharpe_list.append(metrics['sharpe_ratio'])
            dqn_return_list.append(metrics['annualized_return'])
            dqn_vol_list.append(metrics['volatility'])
            dqn_dd_list.append(metrics['max_drawdown'])
            
            dqn_seed_results[seed] = {
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['annualized_return'],
                'volatility': metrics['volatility'],
                'max_dd': metrics['max_drawdown']
            }
            
            print(f"    Sharpe: {metrics['sharpe_ratio']:.4f}, "
                  f"Return: {metrics['annualized_return']:.2%}, "
                  f"MaxDD: {metrics['max_drawdown']:.2%}")
    
    # ========================================================================
    # PG 모델 평가
    # ========================================================================
    print("\n[Step 3/3] Evaluating PG models on Test data...")
    print("-" * 70)
    
    pg_dir = os.path.join(project_root, 'results', 'models', 'pg_randomseed')
    pg_models = sorted(glob.glob(os.path.join(pg_dir, pg_pattern)))
    
    if not pg_models:
        print(f"⚠️  No PG models found in {pg_dir}")
        pg_sharpe_list = []
        pg_return_list = []
        pg_vol_list = []
        pg_dd_list = []
        pg_seed_results = {}
    else:
        print(f"Found {len(pg_models)} PG models")
        
        pg_sharpe_list = []
        pg_return_list = []
        pg_vol_list = []
        pg_dd_list = []
        pg_seed_results = {}
        
        for model_path in pg_models:
            model_name = os.path.basename(model_path)
            # Extract seed from filename
            seed = int(model_name.split('seed')[1].split('_')[0])
            
            print(f"  Evaluating seed={seed}...")
            
            model = load_model(model_path, 'pg')
            metrics = evaluate_model_on_dataset(model, test_df, 'pg')
            
            pg_sharpe_list.append(metrics['sharpe_ratio'])
            pg_return_list.append(metrics['annualized_return'])
            pg_vol_list.append(metrics['volatility'])
            pg_dd_list.append(metrics['max_drawdown'])
            
            pg_seed_results[seed] = {
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['annualized_return'],
                'volatility': metrics['volatility'],
                'max_dd': metrics['max_drawdown']
            }
            
            print(f"    Sharpe: {metrics['sharpe_ratio']:.4f}, "
                  f"Return: {metrics['annualized_return']:.2%}, "
                  f"MaxDD: {metrics['max_drawdown']:.2%}")
    
    # ========================================================================
    # 결과 요약
    # ========================================================================
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    
    # DQN 통계
    if dqn_sharpe_list:
        print_summary_statistics(dqn_sharpe_list, dqn_return_list, dqn_vol_list, 
                                dqn_dd_list, "DQN (λ=10.0, ε=50)")
    
    # PG 통계
    if pg_sharpe_list:
        print_summary_statistics(pg_sharpe_list, pg_return_list, pg_vol_list, 
                                pg_dd_list, "PG (λ=10.0, n=5)")
    
    # ========================================================================
    # Seed별 상세 결과
    # ========================================================================
    print("\n" + "=" * 70)
    print("Detailed Results by Seed")
    print("=" * 70)
    
    if dqn_seed_results:
        print_seed_results(dqn_seed_results, "DQN")
    
    if pg_seed_results:
        print_seed_results(pg_seed_results, "PG")
    
    # ========================================================================
    # 비교 테이블
    # ========================================================================
    print("\n" + "=" * 70)
    print("Final Comparison Table")
    print("=" * 70)
    
    print(f"\n{'Strategy':<25} {'Sharpe Ratio':<20} {'Ann. Return':<20} {'Max DD':<20}")
    print("-" * 85)
    
    if dqn_sharpe_list:
        dqn_sharpe_mean, _, _, dqn_sharpe_std = calculate_confidence_interval(dqn_sharpe_list)
        dqn_return_mean, _, _, dqn_return_std = calculate_confidence_interval(dqn_return_list)
        dqn_dd_mean, _, _, dqn_dd_std = calculate_confidence_interval(dqn_dd_list)
        
        print(f"{'DQN (λ=10.0, ε=50)':<25} "
              f"{dqn_sharpe_mean:.4f} ± {dqn_sharpe_std:.4f}   "
              f"{dqn_return_mean:.2%} ± {dqn_return_std:.2%}   "
              f"{dqn_dd_mean:.2%} ± {dqn_dd_std:.2%}")
    
    if pg_sharpe_list:
        pg_sharpe_mean, _, _, pg_sharpe_std = calculate_confidence_interval(pg_sharpe_list)
        pg_return_mean, _, _, pg_return_std = calculate_confidence_interval(pg_return_list)
        pg_dd_mean, _, _, pg_dd_std = calculate_confidence_interval(pg_dd_list)
        
        print(f"{'PG (λ=10.0, n=5)':<25} "
              f"{pg_sharpe_mean:.4f} ± {pg_sharpe_std:.4f}   "
              f"{pg_return_mean:.2%} ± {pg_return_std:.2%}   "
              f"{pg_dd_mean:.2%} ± {pg_dd_std:.2%}")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    
    # 결과 반환 (다음 단계에서 사용 가능)
    return {
        'dqn': {
            'seed_results': dqn_seed_results,
            'sharpe_list': dqn_sharpe_list,
            'return_list': dqn_return_list,
            'vol_list': dqn_vol_list,
            'dd_list': dqn_dd_list
        },
        'pg': {
            'seed_results': pg_seed_results,
            'sharpe_list': pg_sharpe_list,
            'return_list': pg_return_list,
            'vol_list': pg_vol_list,
            'dd_list': pg_dd_list
        }
    }


if __name__ == '__main__':
    main()
