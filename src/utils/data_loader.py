"""
Data Loading and Preprocessing

This module handles:
- Downloading S&P500 and DGS10 data
- Calculating returns
- Train/Val/Test split
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred


def load_sp500_data(start_date='1962-01-01', end_date='2025-11-30'):
    """
    S&P500 지수 데이터 다운로드
    
    Args:
        start_date (str): 시작 날짜 (YYYY-MM-DD)
        end_date (str): 종료 날짜 (YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: S&P500 종가 데이터
            Index: Date
            Columns: ['SP500']
    """
    print(f"Downloading S&P500 data from {start_date} to {end_date}...")
    
    # ^GSPC = S&P500 지수 티커
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    
    # Close 가격만 추출하고 컬럼명 변경
    sp500 = sp500[['Close']].rename(columns={'Close': 'SP500'})
    
    print(f"Downloaded {len(sp500)} days of S&P500 data")
    
    return sp500


def load_dgs10_data(start_date='1962-01-01', end_date='2025-11-30', csv_path='../../data/raw/DGS10.csv'):
    """
    로컬 CSV 파일에서 10년 국채 금리(DGS10) 데이터 로드
    
    Args:
        start_date (str): 시작 날짜 (YYYY-MM-DD)
        end_date (str): 종료 날짜 (YYYY-MM-DD)
        csv_path (str): DGS10 CSV 파일 경로 (기본: data/raw/DGS10.csv)
    
    Returns:
        pd.DataFrame: DGS10 금리 데이터 (% 단위)
            Index: Date
            Columns: ['DGS10']
    
    Note:
        CSV 파일 형식:
        - 첫 번째 컬럼: observation_date (YYYY-MM-DD)
        - 두 번째 컬럼: DGS10 (금리 %, 예: 4.5)
    """
    import os
    
    print(f"Loading DGS10 data from {csv_path}...")
    
    # 상대 경로를 절대 경로로 변환 (현재 파일 기준)
    if not os.path.isabs(csv_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, csv_path)
    
    # CSV 파일 존재 확인
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"DGS10 CSV file not found at: {csv_path}\n"
            f"Please download DGS10 data from FRED and save it to data/raw/DGS10.csv\n"
            f"Download link: https://fred.stlouisfed.org/series/DGS10"
        )
    
    # CSV 파일 읽기 (observation_date 컬럼을 인덱스로)
    dgs10 = pd.read_csv(csv_path, parse_dates=['observation_date'], index_col='observation_date')
    
    # 날짜 범위 필터링
    dgs10 = dgs10.loc[start_date:end_date]
    
    # 결측치 제거 (주말/공휴일 등)
    dgs10 = dgs10.dropna()
    
    # 인덱스 이름 설정
    dgs10.index.name = 'Date'
    
    print(f"Loaded {len(dgs10)} days of DGS10 data")
    
    return dgs10


# 테스트 코드
if __name__ == '__main__':
    # S&P500 데이터 다운로드 테스트
    sp500 = load_sp500_data()
    
    print("\n=== S&P500 Data ===")
    print(f"Shape: {sp500.shape}")
    print(f"\nFirst 5 rows:")
    print(sp500.head())
    print(f"\nLast 5 rows:")
    print(sp500.tail())
    print(f"\nData info:")
    print(sp500.info())
    
    # DGS10 데이터 로드 테스트
    dgs10 = load_dgs10_data()
    
    print("\n\n=== DGS10 Data ===")
    print(f"Shape: {dgs10.shape}")
    print(f"\nFirst 5 rows:")
    print(dgs10.head())
    print(f"\nLast 5 rows:")
    print(dgs10.tail())
    print(f"\nData info:")
    print(dgs10.info())
    print(f"\nMissing values: {dgs10.isnull().sum().sum()}")
