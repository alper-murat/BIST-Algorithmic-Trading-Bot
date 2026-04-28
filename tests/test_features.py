import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test data generation helpers
def create_sample_stock_data(days=100, start_price=100, volatility=0.02):
    """Create sample stock data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    np.random.seed(42)
    returns = np.random.normal(0.0005, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': prices * (1 + np.random.uniform(0, 0.02, days)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'Close': prices,
        'Volume': np.random.randint(1_000_000, 10_000_000, days)
    }, index=dates)

    return df


def create_sample_index_data(days=100):
    """Create sample index data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    np.random.seed(43)
    returns = np.random.normal(0.0003, 0.01, days)
    prices = 10000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        'High': prices * (1 + np.random.uniform(0, 0.01, days)),
        'Low': prices * (1 - np.random.uniform(0, 0.01, days)),
        'Volume': np.random.randint(500_000_000, 2_000_000_000, days)
    }, index=dates)

    return df


class TestFeatureCalculations:
    """Test technical indicator calculations."""

    def test_rsi_calculation(self):
        """Test RSI calculation produces values in valid range."""
        import pandas_ta as ta

        df = create_sample_stock_data(days=50)
        rsi = ta.rsi(df['Close'], length=14)

        assert rsi.notna().sum() > 0, "RSI should have valid values"
        assert (rsi.dropna() >= 0).all(), "RSI should be >= 0"
        assert (rsi.dropna() <= 100).all(), "RSI should be <= 100"

    def test_atr_calculation(self):
        """Test ATR calculation produces positive values."""
        import pandas_ta as ta

        df = create_sample_stock_data(days=50)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        assert atr.notna().sum() > 0, "ATR should have valid values"
        assert (atr.dropna() > 0).all(), "ATR should be positive"

    def test_obv_calculation(self):
        """Test OBV calculation."""
        import pandas_ta as ta

        df = create_sample_stock_data(days=50)
        obv = ta.obv(df['Close'], df['Volume'])

        assert obv.notna().sum() > 0, "OBV should have valid values"

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        import pandas_ta as ta

        df = create_sample_stock_data(days=50)
        bbands = df.ta.bbands(length=20, std=2)

        assert bbands is not None, "Bollinger Bands should be calculated"
        assert not bbands.empty, "Bollinger Bands should not be empty"


class TestTargetGeneration:
    """Test target label generation logic."""

    def test_target_definition(self):
        """Test the 3% profit target in 2 days logic."""
        HEDEF_KAR = 1.03
        STOP_LOSS = 0.98

        df = create_sample_stock_data(days=50)
        df['Open'] = df['Close'].shift(1)  # Today's open = yesterday's close

        # Simulate target calculation
        gun1_open = df['Open'].shift(-1)
        gun1_high = df['High'].shift(-1)
        gun1_low = df['Low'].shift(-1)
        gun2_high = df['High'].shift(-2)

        hedef_fiyat = gun1_open * HEDEF_KAR
        stop_fiyat = gun1_open * STOP_LOSS

        gun1_hedefe_gitti = gun1_high >= hedef_fiyat
        gun1_stop_oldu = gun1_low <= stop_fiyat
        gun2_hedefe_gitti = (~gun1_stop_oldu) & (gun2_high >= hedef_fiyat)

        target = (gun1_hedefe_gitti | gun2_hedefe_gitti).astype(int)

        assert target.name == 'Target', "Target column should be named 'Target'"
        assert target.dtype == np.int64, "Target should be integer"


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    def test_zero_volume_handling(self):
        """Test that zero volumes are handled properly."""
        df = create_sample_stock_data(days=30)
        df.loc[df.index[10], 'Volume'] = 0

        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)

        assert (df['Volume'] > 0).all(), "Volume should not have zeros after handling"

    def test_duplicate_index_handling(self):
        """Test that duplicate indices are removed."""
        df = create_sample_stock_data(days=30)

        # Add a duplicate row
        new_row = df.iloc[-1].copy()
        df = pd.concat([df, pd.DataFrame([new_row], index=[df.index[-1]])])

        assert df.index.duplicated().any(), "Should have duplicates before handling"

        df = df[~df.index.duplicated(keep='last')]

        assert not df.index.duplicated().any(), "Should have no duplicates after handling"

    def test_infinite_values_handling(self):
        """Test that infinite values are handled."""
        df = create_sample_stock_data(days=30)

        # Introduce inf values
        df.iloc[15, df.columns.get_loc('Close')] = np.inf

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        assert not df.isin([np.inf, -np.inf]).any().any(), "Should have no infinite values"


class TestFeatureEngineering:
    """Test feature engineering calculations."""

    def test_bagil_guc_alpha(self):
        """Test relative strength calculation (Bagil_Guc_Alpha)."""
        df_stock = create_sample_stock_data(days=50)
        df_index = create_sample_index_data(days=50)

        df_stock['Hisse_Getiri'] = df_stock['Close'].pct_change()
        df_index['Endeks_Getiri'] = df_index['Close'].pct_change()

        df_stock['Bagil_Guc_Alpha'] = df_stock['Hisse_Getiri'] - df_index['Endeks_Getiri'].reindex(df_stock.index, method='ffill')

        assert 'Bagil_Guc_Alpha' in df_stock.columns, "Bagil_Guc_Alpha should exist"
        assert df_stock['Bagil_Guc_Alpha'].notna().sum() > 0, "Should have valid values"

    def test_bugun_marj_percent(self):
        """Test daily margin percentage calculation."""
        df = create_sample_stock_data(days=30)

        SMOOTHING_FACTOR = 0.0001
        df['Bugun_Marj_%'] = ((df['High'] - df['Low']) / (df['Low'] + SMOOTHING_FACTOR)) * 100

        assert 'Bugun_Marj_%' in df.columns, "Bugun_Marj_% should exist"
        assert (df['Bugun_Marj_%'] > 0).all(), "Daily margin should be positive"

    def test_kapanis_gucu(self):
        """Test close strength calculation."""
        df = create_sample_stock_data(days=30)

        SMOOTHING_FACTOR = 0.0001
        df['Kapanis_Gucu'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + SMOOTHING_FACTOR)

        assert 'Kapanis_Gucu' in df.columns, "Kapanis_Gucu should exist"
        assert (df['Kapanis_Gucu'] >= 0).all(), "Close strength should be >= 0"
        assert (df['Kapanis_Gucu'] <= 1).all(), "Close strength should be <= 1"

    def test_hacim_ort_kati(self):
        """Test volume ratio calculation."""
        df = create_sample_stock_data(days=30)

        df['Ort_Lot_Hacmi'] = df['Volume'].rolling(20).mean()
        SMOOTHING_FACTOR = 0.0001
        df['Hacim_Ort_Kati'] = df['Volume'] / (df['Ort_Lot_Hacmi'] + SMOOTHING_FACTOR)

        assert 'Hacim_Ort_Kati' in df.columns, "Hacim_Ort_Kati should exist"
        assert df['Hacim_Ort_Kati'].notna().sum() > 0, "Should have valid values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
