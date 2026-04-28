
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

import features

warnings.filterwarnings('ignore')

print("\n🧠 YAPAY ZEKA V5.0 (ESNEK ZAMANLAMA & 2 GÜNLÜK HEDEF) 🧠")

calisma_klasoru = os.path.dirname(os.path.abspath(__file__))
txt_yolu = os.path.join(calisma_klasoru, "hisseler.txt")

try:
    with open(txt_yolu, 'r', encoding='utf-8') as dosya:
        bist_hisseler = [satir.strip() for satir in dosya if satir.strip()]
except FileNotFoundError:
    print(f"🛑 HATA: 'hisseler.txt' dosyası bulunamadı!"); exit()

print("📈 BIST100 Endeks verisi indiriliyor...")
try:
    xu100_df = yf.Ticker("XU100.IS").history(period="3y")
    xu100_df.index = pd.to_datetime(xu100_df.index).normalize().tz_localize(None)
    xu100_df['Endeks_Getiri'] = xu100_df['Close'].pct_change()
    xu100_df['Endeks_RSI'] = ta.rsi(xu100_df['Close'], length=features.RSI_LENGTH)
except Exception as e:
    print(f"🛑 HATA: Endeks verisi çekilemedi!"); exit()

print(f"📥 {len(bist_hisseler)} hissenin verisi indiriliyor. (2 Günlük hedefler aranıyor)...")

tum_veriler = []
HEDEF_KAR = 1.03
STOP_LOSS = 0.98
MIN_VOLUME_TRAINING = 20_000_000
MIN_ROWS_TRAINING = 150

for i, hisse in enumerate(bist_hisseler):
    if (i + 1) % 20 == 0: print(f"... {i + 1} hisse işlendi ...")

    try:
        df = yf.Ticker(hisse).history(period="3y")
        if len(df) < MIN_ROWS_TRAINING: continue

        # Calculate all features using shared module
        df = features.calculate_all_features(df, xu100_df)

        # Calculate daily TL volume for filtering
        df['Gunluk_TL_Hacim'] = df['Ort_Lot_Hacmi'] * df['Close']
        if df['Gunluk_TL_Hacim'].iloc[-1] < MIN_VOLUME_TRAINING: continue

        # Calculate target labels for training
        df = features.calculate_target(df, hedef_kar=HEDEF_KAR, stop_loss=STOP_LOSS)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty: continue

        tum_veriler.append(df)

    except Exception as e:
        pass

ana_veri = pd.concat(tum_veriler, ignore_index=True)
print(f"\n✅ Veri toplama tamamlandı. Toplam eğitim satırı: {len(ana_veri)}")

ozellikler = features.get_feature_columns()

X = ana_veri[ozellikler]
y = ana_veri['Target']

split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

ratio = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1)

print("\n⚙️ XGBoost Modeli Yeni 'Esnek' Hedeflere Göre Eğitiliyor...")
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    scale_pos_weight=ratio * 1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.2,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='auc'
)

model.fit(X_train, y_train)

tahmin_olasilik = model.predict_proba(X_test)[:, 1]
tahminler = (tahmin_olasilik > 0.65).astype(int)

print("\n" + "=" * 55)
print(" 📊 V5.0 (2 GÜNLÜK PENCERE) DOĞRULUK ANALİZİ 📊")
print("=" * 55)
print(classification_report(y_test, tahminler, target_names=['Başarısız/Tuzak (0)', 'Net %3 Yaptı (1)']))

kayit_yolu = os.path.join(calisma_klasoru, "bist_model_v5.json")
model.save_model(kayit_yolu)

print("=" * 55)
print(f"🎉 Yeni model V5 olarak kaydedildi:\n👉 {kayit_yolu}")
