
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import numpy as np
import warnings
import os
import sys
import logging
import time
from datetime import datetime


warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').disabled = True

# ==========================================
# CONFIGURATION - Trading Parameters
# ==========================================
GUVEN_ESIGI = 0.60            # Confidence threshold for live scanning
MIN_VOLUME_LIVE = 30_000_000  # Minimum daily volume in TL for live scanning
SMOOTHING_FACTOR = 0.0001     # Smoothing factor to avoid division by zero
SLEEP_TIME = 0.01             # Sleep time between API calls (seconds)
MIN_ROWS_LIVE = 50            # Minimum data rows required for live scanning

# Technical analysis parameters
RSI_LENGTH = 14               # RSI period
ATR_LENGTH = 14               # ATR period
BB_LENGTH = 20                # Bollinger Bands period
VOLUME_ROLLING = 20           # Volume rolling window
OBV_ROLLING = 10              # OBV rolling window
DATA_PERIOD = "3y"            # Historical data period

# ==========================================
# ==========================================

class SessizIslem:
    def __enter__(self):
        self._orijinal_stdout = sys.stdout
        self._orijinal_stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._devnull.close()
        finally:
            sys.stdout = self._orijinal_stdout
            sys.stderr = self._orijinal_stderr

print("\ntarama")

# ==========================================
# ==========================================
calisma_klasoru = os.path.dirname(os.path.abspath(__file__))
ana_klasor = os.path.dirname(calisma_klasoru)

model_yolu = os.path.join(calisma_klasoru, "bist_model_v5.json")
if not os.path.exists(model_yolu):
    model_yolu = os.path.join(ana_klasor, "bist_model_v5.json")

model = XGBClassifier()
try:
    model.load_model(model_yolu)
except Exception:
    print(f"🛑 HATA: Önce eğitimi tamamlayın.");
    sys.exit()

try:
    with open('hisseler.txt', 'r', encoding='utf-8') as dosya:
        bist_hisseler = [satir.strip() for satir in dosya if satir.strip()]
except Exception:
    print("🛑 HATA: 'hisseler.txt' bulunamadı!");
    sys.exit()

#
# 2. ENDEKS (XU100) VERİSİ ÇEKİMİ
#
print(f"\n🔍 Güncel BIST100 (XU100) verisi çekiliyor...")
try:
    with SessizIslem():
        xu100_df = yf.Ticker("XU100.IS").history(period="3y")

    xu100_df.index = pd.to_datetime(xu100_df.index).normalize().tz_localize(None)
    xu100_df['Endeks_Getiri'] = xu100_df['Close'].pct_change()
    xu100_df['Endeks_RSI'] = ta.rsi(xu100_df['Close'], length=RSI_LENGTH)
    print("✅ Endeks verisi onaylandı. Hisse taramasına geçiliyor...\n")
except Exception:
    print(" HATA: İnternet veya Yahoo bağlantı sorunu!");
    sys.exit()

tum_potansiyeller = []
yahoo_bos_gelen = 0
sığ_tahta_elenen = 0
hesaplama_hatasi = 0
basariyla_puanlanan = 0

#
# 3. CANLI TARAMA DÖNGÜSÜ
#
for i, hisse in enumerate(bist_hisseler):
    if (i + 1) % 50 == 0: print(f"... {i + 1} hisse tarandı ...")

    try:
        with SessizIslem():
            df = yf.Ticker(hisse).history(period="3y")

        time.sleep(SLEEP_TIME)

        if df.empty or len(df) < MIN_ROWS_LIVE:
            yahoo_bos_gelen += 1;
            continue

        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        df = df[~df.index.duplicated(keep='last')]
        df = df.join(xu100_df[['Endeks_Getiri', 'Endeks_RSI']], how='left').ffill()

        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
        for col in ['Low', 'High', 'Open', 'Close']:
            df[col] = df[col].replace(0, np.nan).ffill()

        df['Ort_Lot_Hacmi'] = df['Volume'].rolling(VOLUME_ROLLING).mean()

        gunluk_tl_hacim = df['Ort_Lot_Hacmi'].iloc[-1] * df['Close'].iloc[-1]
        if gunluk_tl_hacim < MIN_VOLUME_LIVE:
            sığ_tahta_elenen += 1;
            continue

        df['Hisse_Getiri'] = df['Close'].pct_change()
        df['Bagil_Guc_Alpha'] = df['Hisse_Getiri'] - df['Endeks_Getiri']

        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV_Egimi'] = df['OBV'] / (df['OBV'].rolling(OBV_ROLLING).mean() + SMOOTHING_FACTOR)

        df['RSI_14'] = ta.rsi(df['Close'], length=RSI_LENGTH)
        df['ATRr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=ATR_LENGTH)

        df['Hacim_Ort_Kati'] = df['Volume'] / (df['Ort_Lot_Hacmi'] + SMOOTHING_FACTOR)

        df['Bugun_Marj_%'] = ((df['High'] - df['Low']) / (df['Low'] + SMOOTHING_FACTOR)) * 100
        df['Bugun_Gap_%'] = ((df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + SMOOTHING_FACTOR)) * 100

        bbands = df.ta.bbands(length=BB_LENGTH, std=2)
        df['Bollinger_Genislik'] = bbands.iloc[:, 3] if bbands is not None and not bbands.empty else 1.0

        df['Kapanis_Gucu'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + SMOOTHING_FACTOR)

        bbu_20 = bbands.iloc[:, 2] if bbands is not None and not bbands.empty else df['Close']
        df['Bant_Tasma_Orani'] = (df['Close'] - bbu_20) / (bbu_20 + SMOOTHING_FACTOR)

        df['RSI_Sisme_Skoru'] = df['RSI_14'] * df['Hacim_Ort_Kati']

        #
        #
        #
        ozellikler = [
            'Bagil_Guc_Alpha', 'OBV_Egimi', 'Endeks_RSI', 'RSI_14', 'ATRr_14',
            'Hacim_Ort_Kati', 'Bugun_Marj_%', 'Bugun_Gap_%', 'Bollinger_Genislik',
            'Kapanis_Gucu', 'Bant_Tasma_Orani', 'RSI_Sisme_Skoru'
        ]

        son_satir = df.iloc[[-1]].copy()

        if son_satir[ozellikler].isna().any().any():
            hesaplama_hatasi += 1;
            continue

        X_canli = son_satir[ozellikler].astype(float)
        olasilik = model.predict_proba(X_canli)[0][1]

        if olasilik >= GUVEN_ESIGI:
            atr_yuzde = (son_satir['ATRr_14'].values[0] / son_satir['Close'].values[0]) * 100
            kapanis_fiyati = son_satir['Close'].values[0]

            tum_potansiyeller.append({
                'Hisse': hisse.replace('.IS', ''),
                'Güven': olasilik,
                'Kapanış': round(kapanis_fiyati, 2),
                'Marj (ATR)': f"%{atr_yuzde:.1f}",
                'Hacim (Mn TL)': round(gunluk_tl_hacim / 1_000_000, 1)
            })

        basariyla_puanlanan += 1

    except Exception as e:
        logging.warning(f"Hisse '%s' işlenirken hata: %s", hisse, str(e))

    # ==========================================
# 4. GÜNÜN YILDIZLARI RAPORU VE KAYIT
# ==========================================
print("\n" + "=" * 80)
print(f" CANLI PİYASA ")
print("-" * 80)
print(f"Verisi Boş Gelen / Banlanan          : {yahoo_bos_gelen} hisse")
print(f"Hacmi Sığ Olduğu İçin Elenen         : {sığ_tahta_elenen} hisse")
print(f"Hesaplama Hatası (NaN)               : {hesaplama_hatasi} hisse")
print(f"Başarıyla Puanlanan                  : {basariyla_puanlanan} hisse")

print("\n" + "=" * 80)
if len(tum_potansiyeller) == 0:
    print(f" Hiç hisse düşmedi. Modelin %{GUVEN_ESIGI *100:.0f} güvenlik eşiğini geçen hisse yok.")
else:
    sonuc_df = pd.DataFrame(tum_potansiyeller).sort_values(by='Güven', ascending=False).head(10).reset_index(drop=True)

    ekran_df = sonuc_df.copy()
    ekran_df['Güven'] = ekran_df['Güven'].apply(lambda x: f"%{x * 100:.1f}")

    tarih_bugun = datetime.now().strftime("%Y-%m-%d")

    print(f"🎯 {tarih_bugun} KAPANIŞINA GÖRE YARININ EN İYİ FIRSATLARI")
    print(ekran_df[['Hisse', 'Güven', 'Kapanış', 'Marj (ATR)', 'Hacim (Mn TL)']].to_string(index=False))

    print("-" * 80)
    print \
        (" \n  ")

    dosya_adi = os.path.join(calisma_klasoru, "v5_canli_tarama_sonuclari.csv")
    dosya_var_mi = os.path.isfile(dosya_adi)

    kayit_df = sonuc_df.copy()
    kayit_df['Tarih'] = tarih_bugun
    cols = ['Tarih'] + [col for col in kayit_df.columns if col != 'Tarih']
    kayit_df = kayit_df[cols]

    kayit_df.to_csv(dosya_adi, mode='a', index=False, header=not dosya_var_mi, encoding='utf-8-sig')
    print(f"İşlemler başarıyla günlüğe kaydedildi: {dosya_adi}")
print("=" * 80)