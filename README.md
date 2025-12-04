# Portfolio Management with Reinforcement Learning

이 프로젝트는 강화학습(DQN, Policy Gradient)을 사용하여 주식(S&P 500)과 채권(10년 만기 국채)으로 구성된 포트폴리오의 자산 배분 전략을 학습하는 프로젝트입니다.

## 1. 환경 설정 (Installation)

### 필수 요구 사항
- Python 3.8 이상
- Virtual Environment (권장)

### 설치 방법

1. **저장소 클론 및 디렉토리 이동**
   ```bash
   git clone <repository_url>
   cd school_project_RL
   ```

2. **가상환경 생성 및 활성화**
   ```bash
   # Mac/Linux
   python -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **의존성 라이브러리 설치**
   ```bash
   pip install -r requirements.txt
   ```

## 2. 데이터 준비 (Data Preparation)

이 프로젝트는 **S&P 500 지수**와 **10년 만기 미국 국채 금리(DGS10)** 데이터를 사용합니다.
필요한 원본 데이터는 `data/raw/` 디렉토리에 이미 포함되어 있습니다.

### 데이터 구성
- **`data/raw/SP500.csv`**: S&P 500 지수 데이터 (Source: Yahoo Finance)
- **`data/raw/DGS10.csv`**: 10년 만기 미국 국채 금리 데이터 (Source: FRED)

별도의 다운로드 없이 바로 학습 및 평가가 가능합니다. 만약 최신 데이터로 업데이트하고 싶다면 `src/utils/data_loader.py`가 자동으로 `yfinance`를 통해 S&P 500 데이터를 갱신하거나 캐시를 사용합니다.

## 3. 모델 학습 (Training)

두 가지 강화학습 알고리즘(DQN, Policy Gradient)을 학습시킬 수 있습니다. 학습된 모델과 로그는 `results/` 디렉토리에 저장됩니다.

### DQN (Deep Q-Network) 학습
```bash
python src/experiments/train_dqn.py
```
- **설정**: `src/config.py`의 `DQN_CONFIG`에서 하이퍼파라미터 조정 가능
- **출력**: `results/models/dqn_*.pt` (모델), `results/logs/dqn_*.json` (학습 로그)

### Policy Gradient (REINFORCE) 학습
```bash
python src/experiments/train_pg.py
```
- **설정**: `src/config.py`의 `PG_CONFIG`에서 하이퍼파라미터 조정 가능
- **출력**: `results/models/pg_*.pt` (모델), `results/logs/pg_*.json` (학습 로그)

## 4. 학습된 모델 (Pre-trained Models)

다양한 실험 설정으로 학습된 모델을 제공합니다. 아래 링크에서 다운로드하여 `results/models/` 디렉토리에 위치시키면 바로 평가가 가능합니다.

### 모델 명명 규칙
파일명은 다음과 같은 규칙을 따릅니다:
- **DQN**: `dqn_lr{learning_rate}_bs{batch_size}_seed{seed}_{timestamp}.pt`
- **PG**: `pg_lr{learning_rate}_batch{num_trajectories}_seed{seed}_{timestamp}.pt`

### 다운로드 링크

#### 1. Random Seed Experiments (Lambda = 1.0)
| Algorithm | Seed | Description | Link |
|-----------|------|-------------|------|
| **DQN** | 42 | Baseline (Completed) | [Download](#) |
| **PG** | 42 | Baseline (Completed) | [Download](#) |
| **DQN** | 123 | Seed 123 (Running) | [Download](#) |
| **PG** | 123 | Seed 123 (Running) | [Download](#) |
| **DQN** | 456 | Seed 456 (Planned) | [Download](#) |
| **PG** | 456 | Seed 456 (Planned) | [Download](#) |
| **DQN** | 789 | Seed 789 (Planned) | [Download](#) |
| **PG** | 789 | Seed 789 (Planned) | [Download](#) |
| **DQN** | 1024 | Seed 1024 (Planned) | [Download](#) |
| **PG** | 1024 | Seed 1024 (Planned) | [Download](#) |

#### 2. Risk Aversion (Lambda) Experiments (Seed = 42)
| Algorithm | Lambda | Description | Link |
|-----------|--------|-------------|------|
| **DQN** | 5.0 | Balanced Risk | [Download](#) |
| **PG** | 5.0 | Balanced Risk | [Download](#) |
| **DQN** | 10.0 | High Safety | [Download](#) |
| **PG** | 10.0 | High Safety | [Download](#) |

#### 3. Hyperparameter Tuning (Seed = 42, Lambda = 1.0)
| Algorithm | Parameter | Value | Description | Link |
|-----------|-----------|-------|-------------|------|
| **DQN** | Epsilon Decay | 50 | Fast Exploration | [Download](#) |
| **DQN** | Epsilon Decay | 200 | Slow Exploration | [Download](#) |
| **PG** | Trajectories | 3 | Small Batch | [Download](#) |
| **PG** | Trajectories | 10 | Large Batch | [Download](#) |

> **Note**: 위 모델들은 `results/models/` 폴더에 저장해야 `eval_all.py` 스크립트가 인식할 수 있습니다.

## 5. 평가 및 시각화 (Evaluation)

학습된 모델을 평가하고 Baseline 전략(주식 100%, 채권 100%, 60/40 전략)과 비교합니다.

```bash
python src/experiments/eval_all.py
```
- Test 기간(2008년~현재)에 대한 수익률, Sharpe Ratio, MDD 등을 계산합니다.
- 결과 그래프는 `results/plots/`에 저장됩니다.

## 6. 프로젝트 구조 (Project Structure)

```
school_project_RL/
├── data/                   # 데이터 디렉토리
│   ├── raw/                # 원본 데이터 (SP500.csv, DGS10.csv)
│   └── processed/          # 전처리된 데이터
├── results/                # 실험 결과
│   ├── models/             # 학습된 모델 (.pt)
│   ├── logs/               # 학습 로그 (.json)
│   └── plots/              # 결과 그래프 (.png)
├── src/                    # 소스 코드
│   ├── agents/             # RL 에이전트 (dqn_agent.py, pg_agent.py)
│   ├── env/                # 강화학습 환경 (portfolio_env.py)
│   ├── experiments/        # 학습 및 평가 스크립트
│   ├── utils/              # 데이터 로더 및 유틸리티
│   └── config.py           # 전체 설정 관리
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 설명
```

## 7. 주요 설정 (Configuration)

`src/config.py` 파일에서 프로젝트의 모든 주요 설정을 관리합니다.

- **데이터 기간**: `DATA_START_DATE`, `DATA_END_DATE`
- **위험 회피 성향**: `LAMBDA_RISK` (기본값: 1.0)
- **학습 파라미터**: Learning rate, Batch size, Episodes 등
