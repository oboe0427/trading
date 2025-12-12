# src/models/har_rv.py
"""
HAR / log-HAR for USDJPY 5min

프로젝트 구조:
trading-project/
    data/
        usdjpym5_indicators_0529-1204.csv
    src/
        models/
            har_rv.py

설정 예:
- H = 1,3,6,12 (다음 H개 5분 RV → 5,15,30,60분)
- S = 12 (1시간)
- M = 48 (4시간)
- L = 288 (24시간)

CSV 전제:
- ';' 구분자
- 컬럼: Date, Time, Open, High, Low, Close, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------------------------------------------------------------------
# 설정값 객체
# ---------------------------------------------------------------------
@dataclass
class HARConfig:
    """HAR-RV 윈도우 설정 및 기본 옵션"""
    H: int = 1          # 예측 horizon (다음 H개 5분 RV)
    S: int = 12         # short window (1시간 = 12 * 5분봉)
    M: int = 48         # medium window (4시간 = 48 * 5분봉)
    L: int = 288        # long window (24시간 = 288 * 5분봉)
    price_col: str = "Close"


# ---------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------
def load_usdjpy_indicators(
    csv_path: str | Path,
    src_utc_offset: int = 2,   # 원본 CSV 시간대 (UTC+2)
    dst_utc_offset: int = 9,   # 변환하고 싶은 시간대 (UTC+9, 한국)
) -> pd.DataFrame:
    """
    usdjpym5_indicators_0529-1204.csv 전용 로더

    - ';' 구분자
    - Date, Time -> timestamp 인덱스
    - CSV는 UTC+2 기준으로 기록되어 있고,
      반환되는 timestamp / Date / Time 은 UTC+9 기준으로 변환된다.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=";")

    # 1) CSV의 Date+Time을 naive datetime으로 파싱 (원본: UTC+2)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"])

    # 2) UTC+2 -> UTC+9 로 시프트 (기본: +7시간)
    diff_hours = dst_utc_offset - src_utc_offset
    ts_local = ts + pd.to_timedelta(diff_hours, unit="h")

    # 3) 변환된 timestamp를 인덱스로 사용
    df["timestamp"] = ts_local
    df = df.sort_values("timestamp").set_index("timestamp")

    # 🔹 4) Date / Time 컬럼도 UTC+9 기준으로 다시 생성
    # (필요하면 포맷은 여기서 바꾸면 됨)
    df["Date"] = df.index.strftime("%Y.%m.%d")   # 예: 2025.12.09
    df["Time"] = df.index.strftime("%H:%M")      # 예: 13:25  (5분봉이라 보통 분 단위면 충분)

    return df


# ---------------------------------------------------------------------
# RV / log-RV 기본 처리
# ---------------------------------------------------------------------
def compute_rv(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    5분 realized variance (RV) 계산.

    rv_t = (log(C_t) - log(C_{t-1}))^2

    Returns:
        'log_price', 'log_ret', 'rv' 컬럼이 추가된 복사본
    """
    df = df.copy()
    df["log_price"] = np.log(df[price_col].astype(float))
    df["log_ret"] = df["log_price"].diff()
    df["rv"] = df["log_ret"] ** 2
    return df


def prepare_base_rv_df(
    df: pd.DataFrame,
    price_col: str = "Close",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    RV, log-RV 모두 가지는 베이스 df 생성.

    - rv 없으면 compute_rv로 계산
    - log_rv 없으면 log(rv + eps) 계산
    """
    if "rv" not in df.columns:
        df = compute_rv(df, price_col=price_col)
    else:
        df = df.copy()

    if "log_rv" not in df.columns:
        df["log_rv"] = np.log(df["rv"] + eps)

    return df


# ---------------------------------------------------------------------
# (1) Level HAR-RV (그냥 rv 기준)
# ---------------------------------------------------------------------
def add_har_features(df: pd.DataFrame, cfg: HARConfig) -> pd.DataFrame:
    """
    HAR-RV용 S/M/L 윈도우 피처 + 미래 RV(H-step) 타겟 생성. (level rv 버전)

    - rv_s_t : 최근 S개 5분봉 rv의 평균
    - rv_m_t : 최근 M개 5분봉 rv의 평균
    - rv_l_t : 최근 L개 5분봉 rv의 평균
    - rv_future_t : rv_{t+H}
    """
    df = prepare_base_rv_df(df, price_col=cfg.price_col)

    # HAR 윈도우 피처 (short / medium / long)
    df["rv_s"] = df["rv"].rolling(cfg.S).mean()
    df["rv_m"] = df["rv"].rolling(cfg.M).mean()
    df["rv_l"] = df["rv"].rolling(cfg.L).mean()

    # H-step ahead 타겟
    df["rv_future"] = df["rv"].shift(-cfg.H)

    # 결측 제거
    df = df.dropna(subset=["rv_s", "rv_m", "rv_l", "rv_future"]).copy()

    return df


def prepare_har_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Level HAR-RV OLS용 X, y 생성.

    X = [1, rv_s, rv_m, rv_l]
    y = rv_future
    """
    X = df[["rv_s", "rv_m", "rv_l"]]
    X = sm.add_constant(X)
    y = df["rv_future"]
    return X, y


def fit_har_ols(
    X: pd.DataFrame,
    y: pd.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Level HAR-RV OLS 피팅."""
    model = sm.OLS(y, X, missing="drop")
    res = model.fit()
    return res


# ---------------------------------------------------------------------
# (2) log-HAR-RV
# ---------------------------------------------------------------------
def add_log_har_features(
    df: pd.DataFrame,
    cfg: HARConfig,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    log-HAR-RV용 S/M/L 윈도우 피처 + log-RV 타겟 생성.

    - log_rv = log(rv + eps)
    - log_rv_s: 최근 S개 log_rv 평균
    - log_rv_m: 최근 M개 log_rv 평균
    - log_rv_l: 최근 L개 log_rv 평균
    - log_rv_future: log_rv_{t+H}
    """
    df = prepare_base_rv_df(df, price_col=cfg.price_col, eps=eps)

    df["log_rv_s"] = df["log_rv"].rolling(cfg.S).mean()
    df["log_rv_m"] = df["log_rv"].rolling(cfg.M).mean()
    df["log_rv_l"] = df["log_rv"].rolling(cfg.L).mean()

    df["log_rv_future"] = df["log_rv"].shift(-cfg.H)

    df = df.dropna(
        subset=["log_rv_s", "log_rv_m", "log_rv_l", "log_rv_future"]
    ).copy()

    return df


def prepare_log_har_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    log-HAR-RV OLS용 X, y 생성.

    X = [1, log_rv_s, log_rv_m, log_rv_l]
    y = log_rv_future
    """
    X = df[["log_rv_s", "log_rv_m", "log_rv_l"]]
    X = sm.add_constant(X)
    y = df["log_rv_future"]
    return X, y


def fit_log_har_ols(
    X: pd.DataFrame,
    y: pd.Series,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """log-HAR-RV OLS 피팅."""
    model = sm.OLS(y, X, missing="drop")
    res = model.fit()
    return res


def fit_log_har_for_horizons(
    df_raw: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 6, 12),
    S: int = 12,
    M: int = 48,
    L: int = 288,
    price_col: str = "Close",
    eps: float = 1e-12,
    train_ratio: float = 0.8,
) -> Dict[int, dict]:
    """
    여러 H(1,3,6,12 등)에 대해 log-HAR을 한 번에 피팅하고
    결과를 dict로 반환.

    반환 구조 (예: results[1]):
        {
            "cfg": HARConfig(...),
            "df_har": df_har,                 # log-HAR 피처 포함 df
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "model": res,                     # statsmodels 결과 객체
            "split_idx": split_idx,           # train/test 경계 인덱스
            "y_pred_test_log": y_pred_log,    # 로그 스케일 예측
            "y_pred_test_level": y_pred_lvl,  # level (exp) 예측
            "metrics": {
                "R2_train": ...,
                "R2_adj_train": ...,
                "MSE_test_log": ...,
                "MAE_test_log": ...,
                "MSE_test_level": ...,
                "MAE_test_level": ...,
            },
        }
    """
    # 공통 베이스 (rv, log_rv 미리 만들어둔다)
    base_df = prepare_base_rv_df(df_raw, price_col=price_col, eps=eps)

    results: Dict[int, dict] = {}

    for H in horizons:
        cfg = HARConfig(H=H, S=S, M=M, L=L, price_col=price_col)

        df_har = add_log_har_features(base_df, cfg, eps=eps)
        X, y = prepare_log_har_matrix(df_har)

        n = len(df_har)
        split_idx = int(n * train_ratio)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        res = fit_log_har_ols(X_train, y_train)

        # 예측 (log 스케일)
        y_pred_log = res.predict(X_test)

        # level 스케일 (실제 RV 유사한 스케일)
        y_test_lvl = np.exp(y_test)
        y_pred_lvl = np.exp(y_pred_log)

        # 간단 메트릭
        mse_test_log = float(((y_test - y_pred_log) ** 2).mean())
        mae_test_log = float((y_test - y_pred_log).abs().mean())

        mse_test_lvl = float(((y_test_lvl - y_pred_lvl) ** 2).mean())
        mae_test_lvl = float((y_test_lvl - y_pred_lvl).abs().mean())

        results[H] = {
            "cfg": cfg,
            "df_har": df_har,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "model": res,
            "split_idx": split_idx,
            "y_pred_test_log": y_pred_log,
            "y_pred_test_level": y_pred_lvl,
            "metrics": {
                "R2_train": float(res.rsquared),
                "R2_adj_train": float(res.rsquared_adj),
                "MSE_test_log": mse_test_log,
                "MAE_test_log": mae_test_log,
                "MSE_test_level": mse_test_lvl,
                "MAE_test_level": mae_test_lvl,
            },
        }

    return results

def grid_search_log_har_sml(
    df_raw: pd.DataFrame,
    H: int = 1,
    S_candidates: Iterable[int] = (6, 12, 18, 24),            # 0.5h, 1h, 1.5h, 2h
    M_candidates: Iterable[int] = (24, 48, 72, 96),           # 2h, 4h, 6h, 8h
    L_candidates: Iterable[int] = (144, 288, 432, 576),       # 12h, 24h, 36h, 48h
    price_col: str = "Close",
    eps: float = 1e-12,
    train_ratio: float = 0.8,
    metric: str = "MAE_test_level",   # 최적화 기준
    minimize: bool = True,            # True면 metric 최소화, False면 최대화
) -> Dict[str, object]:
    """
    H를 고정하고 S, M, L 후보를 자동으로 탐색하는 그리드 서치.

    예)
        search_res = grid_search_log_har_sml(df_raw, H=1)
        best_key = search_res["best_key"]          # (S,M,L) 튜플
        best_summary = search_res["summary_df"].head()

    반환:
        {
            "summary_df":  각 (S,M,L)별 성능 요약 DataFrame (MultiIndex),
            "best_key":    (S_best, M_best, L_best),
            "best_result": best_result_dict (fit_log_har_for_horizons에서 쓰던 구조와 거의 동일),
            "all_results": {(S,M,L): result_dict, ...},
            "metric":      metric 이름,
        }
    """
    # 공통 베이스 (rv, log_rv 미리 생성)
    base_df = prepare_base_rv_df(df_raw, price_col=price_col, eps=eps)

    all_results: Dict[tuple, dict] = {}
    summary_rows = []

    for S in S_candidates:
        for M in M_candidates:
            if M < S:
                continue  # 중기 윈도우는 단기보다 길어야 함
            for L in L_candidates:
                if L < M:
                    continue  # 장기 윈도우는 중기보다 길어야 함

                cfg = HARConfig(H=H, S=S, M=M, L=L, price_col=price_col)

                # log-HAR 피처 생성
                df_har = add_log_har_features(base_df, cfg, eps=eps)

                # 데이터가 너무 적으면 스킵
                if len(df_har) < 500:  # 필요하면 조정
                    continue

                X, y = prepare_log_har_matrix(df_har)

                n = len(df_har)
                split_idx = int(n * train_ratio)

                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                res = fit_log_har_ols(X_train, y_train)

                # 예측 (log 스케일)
                y_pred_log = res.predict(X_test)

                # level 스케일 (실제 RV 스케일)
                y_test_lvl = np.exp(y_test)
                y_pred_lvl = np.exp(y_pred_log)

                mse_test_log = float(((y_test - y_pred_log) ** 2).mean())
                mae_test_log = float((y_test - y_pred_log).abs().mean())

                mse_test_lvl = float(((y_test_lvl - y_pred_lvl) ** 2).mean())
                mae_test_lvl = float((y_test_lvl - y_pred_lvl).abs().mean())

                key = (S, M, L)

                metrics_dict = {
                    "R2_train": float(res.rsquared),
                    "R2_adj_train": float(res.rsquared_adj),
                    "MSE_test_log": mse_test_log,
                    "MAE_test_log": mae_test_log,
                    "MSE_test_level": mse_test_lvl,
                    "MAE_test_level": mae_test_lvl,
                }

                all_results[key] = {
                    "cfg": cfg,
                    "df_har": df_har,
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                    "y_test": y_test,
                    "model": res,
                    "split_idx": split_idx,
                    "y_pred_test_log": y_pred_log,
                    "y_pred_test_level": y_pred_lvl,
                    "metrics": metrics_dict,
                }

                summary_rows.append(
                    {
                        "S": S,
                        "M": M,
                        "L": L,
                        **metrics_dict,
                    }
                )

    if not summary_rows:
        raise ValueError("유효한 (S,M,L) 조합에서 결과가 생성되지 않았습니다.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index(["S", "M", "L"])

    if metric not in summary_df.columns:
        raise ValueError(f"metric='{metric}' 이(가) summary_df 컬럼에 없습니다: {summary_df.columns.tolist()}")

    # 최적 조합 선택
    if minimize:
        best_key = summary_df[metric].idxmin()
    else:
        best_key = summary_df[metric].idxmax()

    best_result = all_results[best_key]

    # metric 기준으로 정렬된 summary_df
    ascending = minimize
    summary_df = summary_df.sort_values(metric, ascending=ascending)

    return {
        "summary_df": summary_df,
        "best_key": best_key,
        "best_result": best_result,
        "all_results": all_results,
        "metric": metric,
    }


# ---------------------------------------------------------------------
# 모듈 단독 실행 예시 (옵션)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]  # .../trading-project
    csv_path = ROOT / "data" / "usdjpym5_indicators_0529-1204.csv"

    df_raw = load_usdjpy_indicators(csv_path)

    # 예시: log-HAR H={1,3,6,12}
    results = fit_log_har_for_horizons(df_raw, horizons=(1, 3, 6, 12))

    for H, r in sorted(results.items()):
        print(f"\n==== H = {H} ====")
        print("R2_train:", r["metrics"]["R2_train"])
        print("MSE_test_log:", r["metrics"]["MSE_test_log"])
