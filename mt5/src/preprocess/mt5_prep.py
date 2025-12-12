# trading-project/src/preprocess/mt5_prep.py

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]
eps = 1e-6  # 캔들 피처 계산에서 0으로 나누기 방지용


# =========================================================
# 1) CSV 로드
# =========================================================
def load_mt5_csv(path: PathLike, sep: str = ";") -> pd.DataFrame:
    """
    MT5에서 추출한 세미콜론 구분 CSV 로드.
    예: usdjpym5_indicators_0529-1204.csv
    """
    path = Path(path)
    df = pd.read_csv(path, sep=sep)
    return df


# =========================================================
# 2) UTC+2 → UTC+9 변환
# =========================================================
def convert_to_utc9(df: pd.DataFrame) -> pd.DataFrame:
    """
    Date, Time이 UTC+2 기준일 때 UTC+9로 변환.
    - datetime_utc9 컬럼 생성
    - Date, Time 컬럼을 UTC+9 기준으로 다시 세팅
    """
    out = df.copy()

    dt_utc2 = pd.to_datetime(
        out["Date"] + " " + out["Time"],
        format="%Y.%m.%d %H:%M",
    )
    dt_utc9 = dt_utc2 + pd.Timedelta(hours=7)

    out["datetime_utc9"] = dt_utc9
    out["Date"] = dt_utc9.dt.strftime("%Y.%m.%d")
    out["Time"] = dt_utc9.dt.strftime("%H:%M")

    return out


# =========================================================
# 3) 세션 컬럼 추가 (asia / eu / us_early / us_late + session_cat 0~3)
# =========================================================
def _get_session(ts: pd.Timestamp) -> str:
    """
    UTC+9 기준 세션 라벨링.

    0 - asia     : 09:00 ~ 16:59
    1 - eu       : 17:00 ~ 23:29
    2 - us_early : 23:30 ~ 01:59
    3 - us_late  : 02:00 ~ 08:59
    """
    h = ts.hour
    m = ts.minute

    # 0) asia: 09:00 ~ 16:59
    if 9 <= h < 17:
        return "asia"

    # 1) eu: 17:00 ~ 23:29
    #   - 17:00~22:59 전체
    #   - 23:00~23:29
    if (17 <= h < 23) or (h == 23 and m < 30):
        return "eu"

    # 2) us_early: 23:30 ~ 01:59
    #   - 23:30~23:59
    #   - 00:00~01:59
    if (h == 23 and m >= 30) or (0 <= h < 2):
        return "us_early"

    # 3) 나머지는 us_late: 02:00 ~ 08:59
    return "us_late"


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    datetime_utc9 기준으로
    - session (str: asia / eu / us_early / us_late)
    - session_cat (int: 0 / 1 / 2 / 3)
    컬럼 추가.
    """
    if "datetime_utc9" not in df.columns:
        raise ValueError("datetime_utc9 컬럼이 없습니다. convert_to_utc9 먼저 호출하세요.")

    out = df.copy()
    out["session"] = out["datetime_utc9"].apply(_get_session)

    # 0: asia, 1: eu, 2: us_early, 3: us_late
    session_map = {
        "asia": 0,
        "eu": 1,
        "us_early": 2,
        "us_late": 3,
    }
    out["session_cat"] = out["session"].map(session_map).astype("Int64")

    return out


# =========================================================
# 4) ktr = 각 세션의 '첫 봉' (High - Low)
# =========================================================
def add_ktr_first_bar_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    ktr = 각 날짜(Date) & 세션(session) 별 첫 봉의 (High - Low).
    같은 (날짜, 세션) 내 모든 봉에 해당 값을 ffill로 채워 넣음.

    여기서 날짜는 UTC+9 기준 Calendar Date (YYYY-MM-DD).
    """
    required_cols = {"datetime_utc9", "High", "Low", "session"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"ktr 계산에 필요한 컬럼이 없습니다: {missing}")

    out = df.copy()

    # 세션별 날짜
    out["session_date"] = out["datetime_utc9"].dt.date

    # 각 (session, session_date) 그룹에서 첫 행 마킹
    out["is_session_first"] = (
        out.groupby(["session", "session_date"]).cumcount() == 0
    )

    # 첫 봉에서만 High - Low, 나머지는 NaN
    out["ktr"] = np.where(
        out["is_session_first"],
        out["High"] - out["Low"],
        np.nan,
    )

    # 같은 (날짜, 세션) 내 나머지 봉에 첫 봉 ktr 값을 전달
    out["ktr"] = out.groupby(["session", "session_date"])["ktr"].ffill()

    return out


# =========================================================
# 5) 캔들 피처 (몸통, 꼬리, 비율, 위치)
# =========================================================
def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLC에서 캔들 특징(몸통, 꼬리, 비율, 위치)을 계산해서 열을 추가한다.
    """
    out = df.copy()

    o = out["Open"]
    h = out["High"]
    l = out["Low"]
    c = out["Close"]

    # 기본 길이
    out["body"] = (c - o).abs()
    out["body_signed"] = c - o
    out["range"] = h - l

    out["upper_wick"] = h - np.maximum(o, c)
    out["lower_wick"] = np.minimum(o, c) - l

    # 비율 (0으로 나누는 것 방지)
    out["body_pct"] = out["body"] / (out["range"] + eps)  # 몸통 / 전체범위
    out["upper_pct"] = out["upper_wick"] / (out["range"] + eps)  # 위꼬리 / 전체범위
    out["lower_pct"] = out["lower_wick"] / (out["range"] + eps)  # 아래꼬리 / 전체범위

    out["upper_body_ratio"] = out["upper_wick"] / (out["body"] + eps)
    out["lower_body_ratio"] = out["lower_wick"] / (out["body"] + eps)
    out["wick_body_ratio"] = (out["upper_wick"] + out["lower_wick"]) / (out["body"] + eps)

    # 몸통 위치 (0=캔들 맨 아래, 1=캔들 맨 위)
    out["body_top_pos"] = (np.maximum(o, c) - l) / (out["range"] + eps)
    out["body_bot_pos"] = (np.minimum(o, c) - l) / (out["range"] + eps)

    # 방향 플래그
    out["is_bull"] = (out["body_signed"] > 0).astype(int)
    out["is_bear"] = (out["body_signed"] < 0).astype(int)

    return out


# =========================================================
# 6) 캔들 패턴 분류 (Doji, 장대, Hammer, Inv Hammer 등)
# =========================================================
def classify_candle(
    row,
    # --- 파라미터: 필요하면 나중에 백테스트 보면서 튜닝 ---
    doji_body_thresh=0.10,  # 도지: 몸통이 전체의 10% 이하
    doji_min_wick_body=0.5,  # 도지: 꼬리 합 / 몸통 >= 0.5

    long_body_thresh=0.70,  # 장대: 몸통이 전체의 70% 이상
    long_max_wick_body=0.7,  # 장대: 꼬리 합 / 몸통 <= 0.7

    hammer_body_min=0.10,  # 망치/역망치: 몸통 최소 비율
    hammer_body_max=0.60,  # 망치/역망치: 몸통 최대 비율
    hammer_tail_body_min=1.5,  # 망치/역망치: 긴 꼬리 / 몸통 >= 1.5
    hammer_opposite_tail_max=0.7,  # 반대쪽 꼬리 / 몸통 <= 0.7
    hammer_body_top_min=0.6,  # 망치: 몸통 상단이 전체 범위의 60% 이상
    invhammer_body_bot_max=0.4,  # 역망치: 몸통 하단이 전체 범위의 40% 이하
):
    body = row["body"]
    body_signed = row["body_signed"]
    range_ = row["range"]

    # degenerate: high == low 같은 경우
    if range_ < 1e-5 or body < eps:
        return "normal"

    body_pct = row["body_pct"]
    upper_body_ratio = row["upper_body_ratio"]
    lower_body_ratio = row["lower_body_ratio"]
    wick_body_ratio = row["wick_body_ratio"]
    body_top_pos = row["body_top_pos"]
    body_bot_pos = row["body_bot_pos"]

    # 1) 도지 (Doji)
    if (body_pct <= doji_body_thresh) and (wick_body_ratio >= doji_min_wick_body):
        return "doji"

    # 2) 장대양봉 / 장대음봉 (Long Bull/Bear)
    if (body_pct >= long_body_thresh) and (wick_body_ratio <= long_max_wick_body):
        if body_signed > 0:
            return "long_bull"
        elif body_signed < 0:
            return "long_bear"

    # 3) 망치형 (Hammer)
    if (
        hammer_body_min < body_pct <= hammer_body_max
        and lower_body_ratio >= hammer_tail_body_min
        and upper_body_ratio <= hammer_opposite_tail_max
        and body_top_pos >= hammer_body_top_min
    ):
        return "hammer"

    # 4) 역망치형 (Inverted Hammer)
    if (
        hammer_body_min < body_pct <= hammer_body_max
        and upper_body_ratio >= hammer_tail_body_min
        and lower_body_ratio <= hammer_opposite_tail_max
        and body_bot_pos <= invhammer_body_bot_max
    ):
        return "inv_hammer"

    # 5) 나머지 전부 normal
    return "normal"


def add_candle_pattern_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    add_candle_features로 만든 피처를 이용해서
    candle_type 컬럼(문자열)을 추가한다.
    """
    needed = {
        "body",
        "body_signed",
        "range",
        "body_pct",
        "upper_body_ratio",
        "lower_body_ratio",
        "wick_body_ratio",
        "body_top_pos",
        "body_bot_pos",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"캔들 패턴 분류에 필요한 피처가 없습니다. "
            f"먼저 add_candle_features를 호출하세요. 누락: {missing}"
        )

    out = df.copy()
    out["candle_type"] = out.apply(classify_candle, axis=1)
    return out


# =========================================================
# 7) MA 피처 (20SMA, 120SMA, 정배열/역배열, 골든/데드 크로스)
# =========================================================
def add_ma_features(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 120,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    20SMA, 120SMA와 정배열/역배열, 골든/데드 크로스 피처 추가.

    추가 컬럼:
    - sma20        : short_window 이동평균 (기본 20)
    - sma120       : long_window  이동평균 (기본 120)
    - s_l_diff     : sma20 - sma120 (양수=정배열, 음수=역배열)
    - ma_bull      : 정배열 상태 더미 (s_l_diff > 0 → 1, else 0)
    - ma_bear      : 역배열 상태 더미 (s_l_diff < 0 → 1, else 0)
    - golden_cross : s_l_diff가 0 이하 → 0 초과로 바뀐 시점 (골든 크로스)
    - dead_cross   : s_l_diff가 0 이상 → 0 미만으로 바뀐 시점 (데드 크로스)
    - ma_cross     : 골든=1, 데드=-1, 나머지=0
    """
    out = df.copy()

    if price_col not in out.columns:
        raise ValueError(f"MA 계산에 필요한 '{price_col}' 컬럼이 없습니다.")

    # 단순 이동평균
    out["sma20"] = (
        out[price_col]
        .rolling(window=short_window, min_periods=short_window)
        .mean()
    )
    out["sma120"] = (
        out[price_col]
        .rolling(window=long_window, min_periods=long_window)
        .mean()
    )

    # 단기 - 장기 차이
    out["s_l_diff"] = out["sma20"] - out["sma120"]

    # 정배열 / 역배열 상태 더미
    out["ma_bull"] = (out["s_l_diff"] > 0).astype(int)  # 정배열
    out["ma_bear"] = (out["s_l_diff"] < 0).astype(int)  # 역배열

    # 골든 / 데드 크로스 이벤트
    diff_prev = out["s_l_diff"].shift(1)

    out["golden_cross"] = ((diff_prev <= 0) & (out["s_l_diff"] > 0)).astype(int)
    out["dead_cross"] = ((diff_prev >= 0) & (out["s_l_diff"] < 0)).astype(int)

    # 하나로 합친 cross 컬럼 (골든=1, 데드=-1, 나머지=0)
    out["ma_cross"] = 0
    out.loc[out["golden_cross"] == 1, "ma_cross"] = 1
    out.loc[out["dead_cross"] == 1, "ma_cross"] = -1
    out["ma_cross"] = out["ma_cross"].astype(int)

    return out


# =========================================================
# 8) 전체 파이프라인
# =========================================================
def prepare_mt5_with_sessions_ktr_candles(
    path: PathLike,
    sep: str = ";",
    drop_helper_cols: bool = True,
) -> pd.DataFrame:
    """
    1) CSV 로드
    2) UTC+2 → UTC+9 변환
    3) 세션 컬럼 추가 (session, session_cat=0~3)
    4) ktr(세션 첫 봉 고가-저가) 컬럼 추가
    5) 캔들 피처 / 캔들 타입 추가
    6) 20SMA / 120SMA 및 정배열/역배열, 골든/데드 크로스 피처 추가
    """
    df = load_mt5_csv(path, sep=sep)
    df = convert_to_utc9(df)
    df = add_session_columns(df)
    df = add_ktr_first_bar_range(df)
    df = add_candle_features(df)
    df = add_candle_pattern_column(df)
    df = add_ma_features(df)

    if drop_helper_cols:
        for col in ["session_date", "is_session_first"]:
            if col in df.columns:
                df = df.drop(columns=[col])

    return df


# 예전 이름 유지용 (호환)
def prepare_mt5_with_sessions_and_ktr(
    path: PathLike,
    sep: str = ";",
    drop_helper_cols: bool = True,
) -> pd.DataFrame:
    """
    예전 함수 이름 호환용.
    현재는 캔들 피처/패턴까지 모두 포함해서 리턴.
    """
    return prepare_mt5_with_sessions_ktr_candles(
        path=path,
        sep=sep,
        drop_helper_cols=drop_helper_cols,
    )


# =========================================================
# 9) CLI로도 돌릴 수 있게 (선택)
#    예:
#       cd ~/projects/trading-project
#       python -m src.preprocess.mt5_prep -i usdjpym5_indicators_0529-1204.csv -o out.csv
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MT5 USDJPY M5 전처리 (UTC+9, 세션4분할, ktr, 캔들 피처/패턴, MA 피처)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="입력 CSV 경로 (예: usdjpym5_indicators_0529-1204.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="저장할 출력 CSV 경로 (미지정 시 저장하지 않음)",
    )
    parser.add_argument(
        "--no-drop-helper",
        action="store_true",
        help="session_date, is_session_first 컬럼을 드롭하지 않음.",
    )

    args = parser.parse_args()

    df_out = prepare_mt5_with_sessions_ktr_candles(
        args.input,
        sep=";",
        drop_helper_cols=not args.no_drop_helper,
    )

    if args.output:
        df_out.to_csv(args.output, sep=";", index=False)
        print(f"저장 완료: {args.output}")
    else:
        print(df_out.head())
