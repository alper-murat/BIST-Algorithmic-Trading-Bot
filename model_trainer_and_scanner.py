
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
import numpy as np
import warnings
import os
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

print("\n🧠 YAPAY ZEKA V5.0 (ESNEK ZAMANLAMA & 2 GÜNLÜK HEDEF) 🧠")

calisma_klasoru = os.path.dirname(os.path.abspath(__file__))
txt_yolu = os.path.join(calisma_klasoru, "hisseler.txt")

try:
    with open(txt_yolu, 'r') as dosya:
        bist_hisseler = [satir.strip() for satir in dosya if satir.strip()]
except FileNotFoundError:
    print(f"🛑 HATA: 'hisseler.txt' dosyası bulunamadı!"); exit()

print("📈 BIST100 Endeks verisi indiriliyor...")
try:
    xu100_df = yf.Ticker("XU100.IS").history(period="3y")
    xu100_df.index = pd.to_datetime(xu100_df.index).normalize().tz_localize(None)
    xu100_df['Endeks_Getiri'] = xu100_df['Close'].pct_change()
    xu100_df['Endeks_RSI'] = ta.rsi(xu100_df['Close'], length=14)
except:
    print("🛑 HATA: Endeks verisi çekilemedi!"); exit()

print(f"📥 {len(bist_hisseler)} hissenin verisi indiriliyor. (2 Günlük hedefler aranıyor)...")

tum_veriler = []
HEDEF_KAR = 1.03
STOP_LOSS = 0.98

for i, hisse in enumerate(bist_hisseler):
    if (i + 1) % 20 == 0: print(f"... {i + 1} hisse işlendi ...")

    try:
        df = yf.Ticker(hisse).history(period="3y")
        if len(df) < 150: continue

        df.index = pd.to_datetime(df.index).normalize().tz_localize(None)
        df = df[~df.index.duplicated(keep='last')]

        df = df.join(xu100_df[['Endeks_Getiri', 'Endeks_RSI']], how='left').ffill()

        df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
        for col in ['Low', 'High', 'Open', 'Close']:
            df[col] = df[col].replace(0, np.nan).ffill()

        df['Ort_Lot_Hacmi'] = df['Volume'].rolling(20).mean()
        df['Gunluk_TL_Hacim'] = df['Ort_Lot_Hacmi'] * df['Close']
        if df['Gunluk_TL_Hacim'].iloc[-1] < 20_000_000: continue

        # --- ÖZELLİKLER ---
        df['Hisse_Getiri'] = df['Close'].pct_change()
        df['Bagil_Guc_Alpha'] = df['Hisse_Getiri'] - df['Endeks_Getiri']

        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['OBV_Egimi'] = df['OBV'] / (df['OBV'].rolling(10).mean() + 0.0001)

        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['ATRr_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['Hacim_Ort_Kati'] = df['Volume'] / (df['Ort_Lot_Hacmi'] + 0.0001)

        df['Bugun_Marj_%'] = ((df['High'] - df['Low']) / (df['Low'] + 0.0001)) * 100
        df['Bugun_Gap_%'] = ((df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 0.0001)) * 100

        bbands = df.ta.bbands(length=20, std=2)
        df['BBU_20'] = bbands.iloc[:, 2] if bbands is not None and not bbands.empty else df['Close']
        df['Bollinger_Genislik'] = bbands.iloc[:, 3] if bbands is not None and not bbands.empty else 1.0

        df['Kapanis_Gucu'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
        df['Bant_Tasma_Orani'] = (df['Close'] - df['BBU_20']) / (df['BBU_20'] + 0.0001)
        df['RSI_Sisme_Skoru'] = df['RSI_14'] * df['Hacim_Ort_Kati']

        # ==============================================================
        # 🔥 V5.0 BÜYÜK YENİLİK: 2 GÜNLÜK ESNEK HEDEF PENCERESİ 🔥
        # ==============================================================
        gun1_open = df['Open'].shift(-1)
        gun1_high = df['High'].shift(-1)
        gun1_low  = df['Low'].shift(-1)

        gun2_high = df['High'].shift(-2)

        hedef_fiyat = gun1_open * HEDEF_KAR
        stop_fiyat  = gun1_open * STOP_LOSS

        # Senaryo 1: Daha ilk günden fişeği takıp %3 hedefe ulaştı
        gun1_hedefe_gitti = gun1_high >= hedef_fiyat
        gun1_stop_oldu = gun1_low <= stop_fiyat

        # Senaryo 2: İlk gün dinlendi (stop olmadı), ikinci gün %3 hedefe ulaştı
        gun2_hedefe_gitti = (~gun1_stop_oldu) & (gun2_high >= hedef_fiyat)

        # İki senaryodan biri gerçekleşirse model "DOĞRU" bildi kabul ediyoruz
        df['Target'] = (gun1_hedefe_gitti | gun2_hedefe_gitti).astype(int)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if df.empty: continue

        tum_veriler.append(df)

    except Exception as e:
        pass

ana_veri = pd.concat(tum_veriler, ignore_index=True)
print(f"\n✅ Veri toplama tamamlandı. Toplam eğitim satırı: {len(ana_veri)}")

ozellikler = [
    'Bagil_Guc_Alpha', 'OBV_Egimi', 'Endeks_RSI', 'RSI_14', 'ATRr_14',
    'Hacim_Ort_Kati', 'Bugun_Marj_%', 'Bugun_Gap_%', 'Bollinger_Genislik',
    'Kapanis_Gucu', 'Bant_Tasma_Orani', 'RSI_Sisme_Skoru'
]

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
    scale_pos_weight=ratio * 1.0, # Etiketler düzeldiği için ekstra baskıya gerek kalmadı
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