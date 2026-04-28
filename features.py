"""
Shared feature calculation module for BIST Algorithmic Trading Bot.
Contains all technical indicator calculations used by both training and live scanning scripts.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np

# ==========================================
# CONFIGURATION - Technical Analysis Parameters
# ==========================================
RSI_LENGTH = 14               # RSI period
ATR_LENGTH = 14                # ATR period
BB_LENGTH = 20                # Bollinger Bands period
VOLUME_ROLLING = 20           # Volume rolling window
OBV_ROLLING = 10              # OBV rolling window
SMOOTHING_FACTOR = 0.0001     # Smoothing factor to avoid division by zero

# Feature list for model prediction
FEATURE_COLUMNS = [
    'Bagil_Guc_Alpha', 'OBV_Egimi', 'Endeks_RSI', 'RSI_14', 'ATRr_14',
    'Hacim_Ort_Kati', 'Bugun_Marj_%', 'Bugun_Gap_%', 'Bollinger_Genislik',
    'Kapanis_Gucu', 'Bant_Tasma_Orani', 'RSI_Sisme_Skoru'
]


def preprocess_stock_data(df, xu100_df):
    """
    Preprocess stock data and join with index data.

    Args:
        df: Stock DataFrame with OHLCV data
        xu100_df: BIST100 index DataFrame with Endeks_Getiri and Endeks_RSI

    Returns:
        Preprocessed DataFrame with index data joined
    """
    # Normalize datetime index
    df.index = pd.to_datetime(df.index).normalize().tz_localize(None)

    # Remove duplicate indices
    df = df[~df.index.duplicated(keep='last')]

    # Join with index data
    df = df.join(xu100_df[['Endeks_Getiri', 'Endeks_RSI']], how='left').ffill()

    # Handle zero volumes
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)

    # Handle zero prices
    for col in ['Low', 'High', 'Open', 'Close']:
        df[col] = df[col].replace(0, np.nan).ffill()

    return df


def calculate_volume_features(df):
    """
    Calculate volume-based features.

    Args:
        df: DataFrame with Volume column

    Returns:
        DataFrame with additional volume features
    """
    df['Ort_Lot_Hacmi'] = df['Volume'].rolling(VOLUME_ROLLING).mean()
    df['Hacim_Ort_Kati'] = df['Volume'] / (df['Ort_Lot_Hacmi'] + SMOOTHING_FACTOR)
    return df


def calculate_relative_strength(df):
    """
    Calculate relative strength vs index.

    Args:
        df: DataFrame with Hisse_Getiri and Endeks_Getiri

    Returns:
        DataFrame with Bagil_Guc_Alpha feature
    """
    df['Hisse_Getiri'] = df['Close'].pct_change()
    df['Bagil_Guc_Alpha'] = df['Hisse_Getiri'] - df['Endeks_Getiri']
    return df


def calculate_technical_indicators(df):
    """
    Calculate technical indicators (RSI, ATR, OBV).

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with technical indicators
    """
    # On-Balance Volume
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_Egimi'] = df['OBV'] / (df['OBV'].rolling(OBV_ROLLING).mean() + SMOOTHING_FACTOR)

    # RSI and ATR
    df['RSI_14'] = ta.rsi(df['Close'], length=RSI_LENGTH)
    df['ATRr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)

    return df


def calculate_price_features(df):
    """
    Calculate price-based features.

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with price features
    """
    # Daily margin percentage
    df['Bugun_Marj_%'] = ((df['High'] - df['Low']) / (df['Low'] + SMOOTHING_FACTOR)) * 100

    # Gap percentage
    df['Bugun_Gap_%'] = ((df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + SMOOTHING_FACTOR)) * 100

    # Close strength (where in day's range did it close)
    df['Kapanis_Gucu'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + SMOOTHING_FACTOR)

    return df


def calculate_bbands_features(df):
    """
    Calculate Bollinger Bands features.

    Args:
        df: DataFrame with Close price

    Returns:
        DataFrame with Bollinger Bands features
    """
    bbands = df.ta.bbands(length=BB_LENGTH, std=2)

    # Bollinger Band width
    df['Bollinger_Genislik'] = bbands.iloc[:, 3] if bbands is not None and not bbands.empty else 1.0

    # Upper band for band breakout ratio
    bbu_20 = bbands.iloc[:, 2] if bbands is not None and not bbands.empty else df['Close']
    df['Bant_Tasma_Orani'] = (df['Close'] - bbu_20) / (bbu_20 + SMOOTHING_FACTOR)

    return df


def calculate_composite_features(df):
    """
    Calculate composite features combining multiple indicators.

    Args:
        df: DataFrame with RSI_14 and Hacim_Ort_Kati

    Returns:
        DataFrame with composite features
    """
    df['RSI_Sisme_Skoru'] = df['RSI_14'] * df['Hacim_Ort_Kati']
    return df


def calculate_all_features(df, xu100_df):
    """
    Calculate all features for the trading model.

    This is the main function that applies all feature engineering steps.

    Args:
        df: Stock DataFrame with OHLCV data
        xu100_df: BIST100 index DataFrame with Endeks_Getiri and Endeks_RSI

    Returns:
        DataFrame with all features calculated, ready for model prediction
    """
    # Preprocess
    df = preprocess_stock_data(df, xu100_df)

    # Calculate features in order (dependencies noted)
    df = calculate_volume_features(df)          # Adds Ort_Lot_Hacmi, Hacim_Ort_Kati
    df = calculate_relative_strength(df)       # Adds Hisse_Getiri, Bagil_Guc_Alpha
    df = calculate_technical_indicators(df)     # Adds OBV, OBV_Egimi, RSI_14, ATRr_14
    df = calculate_price_features(df)           # Adds Bugun_Marj_%, Bugun_Gap_%, Kapanis_Gucu
    df = calculate_bbands_features(df)          # Adds Bollinger_Genislik, Bant_Tasma_Orani
    df = calculate_composite_features(df)        # Adds RSI_Sisme_Skoru

    return df


def calculate_target(df, hedef_kar=1.03, stop_loss=0.98):
    """
    Calculate target labels for training.

    Target is 1 if the stock hits 3% profit target within 2 trading days.

    Args:
        df: DataFrame with Open, High, Low of future periods
        hedef_kar: Profit target multiplier (default 1.03 = 3%)
        stop_loss: Stop loss multiplier (default 0.98 = 2%)

    Returns:
        DataFrame with Target column added
    """
    gun1_open = df['Open'].shift(-1)
    gun1_high = df['High'].shift(-1)
    gun1_low = df['Low'].shift(-1)
    gun2_high = df['High'].shift(-2)

    hedef_fiyat = gun1_open * hedef_kar
    stop_fiyat = gun1_open * stop_loss

    # Scenario 1: Hits 3% target on day 1
    gun1_hedefe_gitti = gun1_high >= hedef_fiyat
    gun1_stop_oldu = gun1_low <= stop_fiyat

    # Scenario 2: No stop on day 1, hits 3% target on day 2
    gun2_hedefe_gitti = (~gun1_stop_oldu) & (gun2_high >= hedef_fiyat)

    # Target is 1 if either scenario occurs
    df['Target'] = (gun1_hedefe_gitti | gun2_hedefe_gitti).astype(int)

    return df


def get_feature_columns():
    """
    Returns the list of feature column names used by the model.

    Returns:
        List of feature column names
    """
    return FEATURE_COLUMNS.copy()
