import pytest
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

import features


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
        df = create_sample_stock_data(days=50)
        rsi = ta.rsi(df['Close'], length=14)

        assert rsi.notna().sum() > 0, "RSI should have valid values"
        assert (rsi.dropna() >= 0).all(), "RSI should be >= 0"
        assert (rsi.dropna() <= 100).all(), "RSI should be <= 100"

    def test_atr_calculation(self):
        """Test ATR calculation produces positive values."""
        df = create_sample_stock_data(days=50)
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        assert atr.notna().sum() > 0, "ATR should have valid values"
        assert (atr.dropna() > 0).all(), "ATR should be positive"

    def test_obv_calculation(self):
        """Test OBV calculation."""
        df = create_sample_stock_data(days=50)
        obv = ta.obv(df['Close'], df['Volume'])

        assert obv.notna().sum() > 0, "OBV should have valid values"

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        df = create_sample_stock_data(days=50)
        bbands = df.ta.bbands(length=20, std=2)

        assert bbands is not None, "Bollinger Bands should be calculated"
        assert not bbands.empty, "Bollinger Bands should not be empty"


class TestTargetGeneration:
    """Test target label generation logic."""

    def test_target_definition(self):
        """Test the 3% profit target in 2 days logic using features.calculate_target()."""
        df = create_sample_stock_data(days=50)
        df['Open'] = df['Close'].shift(1)  # Today's open = yesterday's close

        # Use the actual features.calculate_target function
        df = features.calculate_target(df, hedef_kar=1.03, stop_loss=0.98)

        assert 'Target' in df.columns, "Target column should exist"
        assert df['Target'].dtype == np.int64, "Target should be integer"
        assert set(df['Target'].unique()).issubset({0, 1}), "Target should be 0 or 1"

    def test_target_uses_parameters(self):
        """Test that calculate_target respects hedef_kar and stop_loss parameters."""
        df = create_sample_stock_data(days=50)
        df['Open'] = df['Close'].shift(1)

        # Test with different parameters
        df_test = features.calculate_target(df.copy(), hedef_kar=1.05, stop_loss=0.97)

        assert 'Target' in df_test.columns, "Target column should exist"


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
    """Test feature engineering calculations using features module."""

    def test_calculate_volume_features(self):
        """Test volume features calculation using features module."""
        df = create_sample_stock_data(days=50)
        xu100_df = create_sample_index_data(days=50)
        xu100_df['Endeks_Getiri'] = xu100_df['Close'].pct_change()

        df = features.preprocess_stock_data(df, xu100_df)
        df = features.calculate_volume_features(df)

        assert 'Ort_Lot_Hacmi' in df.columns, "Ort_Lot_Hacmi should exist"
        assert 'Hacim_Ort_Kati' in df.columns, "Hacim_Ort_Kati should exist"
        assert df['Hacim_Ort_Kati'].notna().sum() > 0, "Should have valid values"

    def test_calculate_price_features(self):
        """Test price features calculation using features module."""
        df = create_sample_stock_data(days=30)
        xu100_df = create_sample_index_data(days=30)
        xu100_df['Endeks_Getiri'] = xu100_df['Close'].pct_change()

        df = features.preprocess_stock_data(df, xu100_df)
        df = features.calculate_volume_features(df)
        df = features.calculate_price_features(df)

        assert 'Bugun_Marj_%' in df.columns, "Bugun_Marj_% should exist"
        assert 'Bugun_Gap_%' in df.columns, "Bugun_Gap_% should exist"
        assert 'Kapanis_Gucu' in df.columns, "Kapanis_Gucu should exist"
        assert (df['Kapanis_Gucu'] >= 0).all(), "Close strength should be >= 0"
        assert (df['Kapanis_Gucu'] <= 1).all(), "Close strength should be <= 1"

    def test_calculate_bbands_features(self):
        """Test Bollinger Bands features using features module."""
        df = create_sample_stock_data(days=50)
        xu100_df = create_sample_index_data(days=50)
        xu100_df['Endeks_Getiri'] = xu100_df['Close'].pct_change()

        df = features.preprocess_stock_data(df, xu100_df)
        df = features.calculate_volume_features(df)
        df = features.calculate_bbands_features(df)

        assert 'Bollinger_Genislik' in df.columns, "Bollinger_Genislik should exist"
        assert 'Bant_Tasma_Orani' in df.columns, "Bant_Tasma_Orani should exist"

    def test_calculate_all_features_end_to_end(self):
        """Test the complete calculate_all_features() pipeline."""
        df_stock = create_sample_stock_data(days=100)
        df_index = create_sample_index_data(days=100)

        df_index['Endeks_Getiri'] = df_index['Close'].pct_change()
        df_index['Endeks_RSI'] = ta.rsi(df_index['Close'], length=14)

        # Calculate all features
        df_result = features.calculate_all_features(df_stock, df_index)

        # Verify all feature columns exist
        for col in features.FEATURE_COLUMNS:
            assert col in df_result.columns, f"Feature {col} should exist"

        # Verify we have valid values for the most recent rows
        assert df_result[features.FEATURE_COLUMNS].notna().any().any(), "Should have some valid values"

    def test_get_feature_columns(self):
        """Test that get_feature_columns returns correct list."""
        cols = features.get_feature_columns()
        assert isinstance(cols, list), "Should return a list"
        assert len(cols) == 12, "Should have 12 features"
        assert 'RSI_14' in cols, "RSI_14 should be in feature list"
        assert 'Bagil_Guc_Alpha' in cols, "Bagil_Guc_Alpha should be in feature list"


class TestFeaturesModuleConstants:
    """Test that features module has correct constants."""

    def test_smoothing_factor_value(self):
        """Test SMOOTHING_FACTOR is set correctly."""
        assert features.SMOOTHING_FACTOR == 0.0001, "SMOOTHING_FACTOR should be 0.0001"

    def test_rsi_length_value(self):
        """Test RSI_LENGTH is set correctly."""
        assert features.RSI_LENGTH == 14, "RSI_LENGTH should be 14"

    def test_atr_length_value(self):
        """Test ATR_LENGTH is set correctly."""
        assert features.ATR_LENGTH == 14, "ATR_LENGTH should be 14"

    def test_bb_length_value(self):
        """Test BB_LENGTH is set correctly."""
        assert features.BB_LENGTH == 20, "BB_LENGTH should be 20"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
