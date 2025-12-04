"""
Performance Metrics for Portfolio Evaluation

포트폴리오 성능 평가를 위한 지표 계산 함수들
"""

import numpy as np
import pandas as pd


def calculate_cumulative_return(returns):
    """
    누적 수익률 계산
    
    Args:
        returns (array-like): 일일 수익률 시계열
        
    Returns:
        float: 누적 수익률 (예: 0.5 = 50% 수익)
    """
    return (1 + np.array(returns)).prod() - 1


def calculate_annualized_return(returns, periods_per_year=252):
    """
    연환산 수익률 계산
    
    Args:
        returns (array-like): 일일 수익률 시계열
        periods_per_year (int): 연간 거래일 수 (기본값: 252)
        
    Returns:
        float: 연환산 수익률
    """
    cum_return = calculate_cumulative_return(returns)
    n_periods = len(returns)
    
    if n_periods == 0:
        return 0.0
    
    annualized = (1 + cum_return) ** (periods_per_year / n_periods) - 1
    return annualized


def calculate_volatility(returns, periods_per_year=252):
    """
    연환산 변동성 계산
    
    Args:
        returns (array-like): 일일 수익률 시계열
        periods_per_year (int): 연간 거래일 수 (기본값: 252)
        
    Returns:
        float: 연환산 변동성 (표준편차)
    """
    return np.std(returns) * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Sharpe Ratio 계산
    
    Args:
        returns (array-like): 일일 수익률 시계열
        risk_free_rate (float): 무위험 이자율 (연환산)
        periods_per_year (int): 연간 거래일 수 (기본값: 252)
        
    Returns:
        float: Sharpe ratio
    """
    ann_return = calculate_annualized_return(returns, periods_per_year)
    volatility = calculate_volatility(returns, periods_per_year)
    
    if volatility == 0:
        return 0.0
    
    sharpe = (ann_return - risk_free_rate) / volatility
    return sharpe


def calculate_max_drawdown(returns):
    """
    Maximum Drawdown 계산
    
    Args:
        returns (array-like): 일일 수익률 시계열
        
    Returns:
        float: Maximum drawdown (음수, 예: -0.2 = -20% 하락)
    """
    # 누적 수익률 계산
    cumulative = (1 + np.array(returns)).cumprod()
    
    # Running maximum 계산
    running_max = np.maximum.accumulate(cumulative)
    
    # Drawdown 계산
    drawdown = (cumulative - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    
    return max_dd


def calculate_all_metrics(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    모든 성능 지표를 한 번에 계산
    
    Args:
        returns (array-like): 일일 수익률 시계열
        risk_free_rate (float): 무위험 이자율 (연환산)
        periods_per_year (int): 연간 거래일 수 (기본값: 252)
        
    Returns:
        dict: 모든 성능 지표를 담은 딕셔너리
    """
    metrics = {
        'cumulative_return': calculate_cumulative_return(returns),
        'annualized_return': calculate_annualized_return(returns, periods_per_year),
        'volatility': calculate_volatility(returns, periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'num_periods': len(returns)
    }
    
    return metrics


def print_metrics(metrics, title="Performance Metrics"):
    """
    성능 지표를 보기 좋게 출력
    
    Args:
        metrics (dict): calculate_all_metrics()의 반환값
        title (str): 출력 제목
    """
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Cumulative Return:    {metrics['cumulative_return']:>10.2%}")
    print(f"Annualized Return:    {metrics['annualized_return']:>10.2%}")
    print(f"Volatility:           {metrics['volatility']:>10.2%}")
    print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>10.4f}")
    print(f"Max Drawdown:         {metrics['max_drawdown']:>10.2%}")
    print(f"Number of Periods:    {metrics['num_periods']:>10d}")
    print("=" * 70)


# ============================================================================
# 테스트 코드
# ============================================================================
if __name__ == '__main__':
    """
    테스트용 코드
    
    실행 방법:
        python src/utils/metrics.py
    """
    print("Testing metrics.py...")
    print()
    
    # 테스트 데이터 생성 (랜덤 수익률)
    np.random.seed(42)
    test_returns = np.random.normal(0.0005, 0.01, 252)  # 1년치 데이터
    
    # 모든 지표 계산
    metrics = calculate_all_metrics(test_returns)
    
    # 출력
    print_metrics(metrics, "Test Portfolio Performance")
    
    print("\n✓ All tests passed!")
