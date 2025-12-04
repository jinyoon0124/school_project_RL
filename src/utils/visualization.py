"""
Visualization utilities for RL Portfolio Management

모든 시각화 함수를 제공합니다:
1. 성능 비교 (누적 수익률, 지표 테이블)
2. 학습 과정 (학습 곡선 비교)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
sns.set_style("whitegrid")
COLORS = {
    '100% Stock': '#e74c3c',
    '100% Bond': '#3498db',
    '60/40 Rebalance': '#95a5a6',
    'DQN': '#2ecc71',
    'PG': '#f39c12'
}


# ============================================================================
# 1. 성능 비교 시각화
# ============================================================================

def plot_cumulative_returns(
    returns_dict: Dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    save_path: str = None,
    title: str = "Cumulative Returns Comparison"
):
    """
    누적 수익률 비교 그래프
    
    Args:
        returns_dict: {전략명: 일간 수익률 배열}
        dates: 날짜 인덱스
        save_path: 저장 경로 (None이면 표시만)
        title: 그래프 제목
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for strategy_name, returns in returns_dict.items():
        # 누적 수익률 계산
        cumulative = (1 + returns).cumprod()
        
        # 색상 선택
        color = COLORS.get(strategy_name, None)
        
        # 플롯
        ax.plot(dates, cumulative, label=strategy_name, 
                linewidth=2, color=color, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (1 = 100%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Y축 로그 스케일 (선택사항)
    # ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_performance_table(
    metrics_dict: Dict[str, Dict[str, Any]],
    save_path: str = None
):
    """
    성능 지표 비교 테이블
    
    Args:
        metrics_dict: {전략명: {지표명: 값}}
                     값은 숫자 또는 문자열 (예: "0.65 ± 0.03")
        save_path: 저장 경로
    """
    # DataFrame 생성
    df = pd.DataFrame(metrics_dict).T
    
    # 컬럼 순서 정렬
    column_order = ['Sharpe Ratio', 'CAGR (%)', 'Volatility (%)', 'Max Drawdown (%)']
    df = df[column_order]
    
    # 셀 텍스트 준비 (숫자는 반올림, 문자열은 그대로)
    cell_text = []
    for idx, row in df.iterrows():
        row_text = []
        for val in row:
            if isinstance(val, str):
                # 이미 문자열이면 그대로 (예: "0.65 ± 0.03")
                row_text.append(val)
            else:
                # 숫자면 반올림
                row_text.append(f"{val:.2f}")
        cell_text.append(row_text)
    
    # 테이블 시각화
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # 테이블 생성
    table = ax.table(
        cellText=cell_text,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.25] * len(df.columns)  # 약간 넓게
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 행 레이블 스타일
    for i in range(len(df)):
        table[(i+1, -1)].set_facecolor('#ecf0f1')
        table[(i+1, -1)].set_text_props(weight='bold')
    
    plt.title('Performance Metrics Comparison', 
              fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# 2. 학습 과정 시각화
# ============================================================================

def plot_learning_curves_comparison(
    logs_dict: Dict[str, List[Dict]],
    algorithm: str = "DQN",
    save_path: str = None,
    title: str = None
):
    """
    여러 실험의 학습 곡선 비교
    
    Args:
        logs_dict: {실험명: 학습 로그 리스트}
                  예: {'seed 42': log1, 'seed 123': log2, 'λ=5.0': log3}
        algorithm: 'DQN' or 'PG'
        save_path: 저장 경로
        title: 그래프 제목 (None이면 자동 생성)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 색상 팔레트
    colors = plt.cm.Set2(np.linspace(0, 1, len(logs_dict)))
    
    for idx, (exp_name, training_log) in enumerate(logs_dict.items()):
        if algorithm == "DQN":
            episodes = [log['episode'] for log in training_log]
            rewards = [log['avg_reward'] for log in training_log]
            
            ax.plot(episodes, rewards, label=exp_name,
                   linewidth=2, color=colors[idx], alpha=0.8)
            
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Average Reward', fontsize=12)
            
        elif algorithm == "PG":
            iterations = [log['iteration'] for log in training_log]
            returns = [log['mean_return'] for log in training_log]
            
            ax.plot(iterations, returns, label=exp_name,
                   linewidth=2, color=colors[idx], alpha=0.8)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Mean Return', fontsize=12)
    
    # 제목 설정
    if title is None:
        title = f'{algorithm} Learning Curves Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()



# ============================================================================
# 유틸리티 함수
# ============================================================================

def create_plots_directory():
    """결과 플롯 디렉토리 생성"""
    plots_dir = 'results/plots'
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


if __name__ == '__main__':
    """
    테스트 코드
    """
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  1. plot_cumulative_returns()")
    print("  2. plot_performance_table()")
    print("  3. plot_learning_curves_comparison()")
