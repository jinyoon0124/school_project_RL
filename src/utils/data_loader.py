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

def split_data(df):
    """
    데이터를 Train/Val/Test로 분할
    
    Args:
        df (pd.DataFrame): 전처리된 데이터 (Index: Date)
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Train: ~ 1999-12-31
    train_df = df.loc[:'1999-12-31']
    
    # Val: 2000-01-01 ~ 2007-12-31
    val_df = df.loc['2000-01-01':'2007-12-31']
    
    # Test: 2008-01-01 ~
    test_df = df.loc['2008-01-01':]
    
    print(f"Data Split Results:")
    print(f"Train: {len(train_df)} days ({train_df.index[0].date()} ~ {train_df.index[-1].date()})")
    print(f"Val  : {len(val_df)} days ({val_df.index[0].date()} ~ {val_df.index[-1].date()})")
    print(f"Test : {len(test_df)} days ({test_df.index[0].date()} ~ {test_df.index[-1].date()})")
    
    return train_df, val_df, test_df



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


def preprocess_data(sp500_df, dgs10_df):
    """
    주식/채권 수익률 계산 및 데이터 병합
    
    Args:
        sp500_df (pd.DataFrame): S&P500 가격 데이터
            Index: Date
            Columns: ['SP500']
        dgs10_df (pd.DataFrame): DGS10 금리 데이터 (% 단위)
            Index: Date
            Columns: ['DGS10']
    
    Returns:
        pd.DataFrame: 일간 수익률 데이터
            Index: Date (공통 거래일만)
            Columns: ['r_stock', 'r_bond']
    
    Note:
        - 주식 수익률: r_stock = P_t / P_{t-1} - 1
        - 채권 수익률: 제로쿠폰 채권 가격 공식 사용
            P_t = 100 / (1 + y_t/100)^10
            r_bond = (P_t / P_{t-1}) - 1
        - Inner join: 주식과 채권 모두 데이터가 있는 날만 사용
    """
    print("Preprocessing data...")
    
    # 1. S&P500 가격 데이터 준비
    # yfinance는 MultiIndex를 반환할 수 있으므로 단일 컬럼으로 변환
    if isinstance(sp500_df.columns, pd.MultiIndex):
        sp500_df = sp500_df.droplevel(1, axis=1)  # Ticker 레벨 제거
    
    # 컬럼명 통일
    sp500_df.columns = ['SP500']
    
    # 2. Inner join으로 공통 날짜만 추출
    data = sp500_df.join(dgs10_df, how='inner')
    
    print(f"Common dates after inner join: {len(data)} days")
    
    # 3. 주식 일간 수익률 계산
    data['r_stock'] = data['SP500'].pct_change()
    
    # 4. 채권 수익률 계산 (제로쿠폰 채권 가격 공식)
    # P_t = 100 / (1 + y_t/100)^10
    data['bond_price'] = 100 / (1 + data['DGS10'] / 100) ** 10
    
    # r_bond = (P_t / P_{t-1}) - 1
    data['r_bond'] = data['bond_price'].pct_change()
    
    # 5. 필요한 컬럼만 선택 (r_stock, r_bond)
    result = data[['r_stock', 'r_bond']].copy()
    
    # 6. 첫 번째 행 제거 (pct_change로 인한 NaN)
    result = result.dropna()
    
    print(f"Final data shape: {result.shape}")
    print(f"Date range: {result.index[0]} to {result.index[-1]}")
    print(f"Missing values: {result.isnull().sum().sum()}")
    
    return result


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
    
    # 데이터 전처리 테스트
    print("\n" + "="*50)
    data = preprocess_data(sp500, dgs10)
    
    print("\n=== Preprocessed Data ===")
    print(f"Shape: {data.shape}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nLast 5 rows:")
    print(data.tail())
    print(f"\nData statistics:")
    print(data.describe())
    print(f"\nData info:")
    print(data.info())

    # 데이터 분할 테스트
    print("\n" + "="*50)
    train, val, test = split_data(data)
