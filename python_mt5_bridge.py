# ============================================================
#  ربط Python مع MT5 في الوقت الفعلي
#  Python ↔ MT5 Real-Time Bridge
# ============================================================
#
#  المتطلبات:
#  pip install MetaTrader5 pandas numpy scikit-learn pyzmq
#
#  الهيكل:
#  ┌──────────────┐   ZeroMQ / Socket   ┌─────────────────┐
#  │  Python      │ ◄────────────────► │  MT5 (MQL5 EA)  │
#  │  (Isolation  │                     │  (Bridge EA)    │
#  │   Forest)    │                     │                 │
#  └──────────────┘                     └─────────────────┘

# ══════════════════════════════════════════════════════════
#  الجزء الأول: Python Bridge Server
# ══════════════════════════════════════════════════════════

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import zmq
import json
import time
import threading
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class PythonMT5Bridge:
    """
    جسر في الوقت الفعلي بين Python و MT5
    يستقبل بيانات السوق من MT5، يحلّلها، ويُعيد إشارات التداول
    """

    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5,
                 lookback=200, zmq_port=5555):
        self.symbol     = symbol
        self.timeframe  = timeframe
        self.lookback   = lookback
        self.zmq_port   = zmq_port
        self.model      = None
        self.scaler     = StandardScaler()
        self.is_running = False
        self.context    = zmq.Context()

    # ────────────────────────────────────────────────────
    #  الاتصال بـ MT5
    # ────────────────────────────────────────────────────
    def connect_mt5(self, login=None, password=None, server=None):
        if not mt5.initialize():
            print(f"❌  فشل الاتصال بـ MT5: {mt5.last_error()}")
            return False

        if login and password and server:
            if not mt5.login(login, password=password, server=server):
                print(f"❌  فشل تسجيل الدخول: {mt5.last_error()}")
                return False

        info = mt5.terminal_info()
        print(f"✅  متصل بـ MT5 | الإصدار: {mt5.version()}")
        print(f"    الشركة: {info.company}")
        return True

    # ────────────────────────────────────────────────────
    #  جلب البيانات من MT5
    # ────────────────────────────────────────────────────
    def fetch_rates(self):
        rates = mt5.copy_rates_from_pos(
            self.symbol, self.timeframe, 0, self.lookback
        )
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df

    # ────────────────────────────────────────────────────
    #  استخراج الميزات
    # ────────────────────────────────────────────────────
    def extract_features(self, df):
        features = pd.DataFrame(index=df.index)

        # 1. نسبة التغير السعري
        features["price_change"]   = df["close"].pct_change() * 100

        # 2. مدى الشمعة نسبةً للمتوسط
        candle_range               = df["high"] - df["low"]
        features["rel_range"]      = candle_range / (candle_range.rolling(14).mean() + 1e-9)

        # 3. ضغط الحجم
        features["vol_surge"]      = df["tick_volume"] / (df["tick_volume"].rolling(14).mean() + 1e-9)

        # 4. الزخم
        features["momentum"]       = df["close"].pct_change(14) * 100

        # 5. الانحراف عن المتوسط المتحرك
        ema20 = df["close"].ewm(span=20).mean()
        features["ema_deviation"]  = (df["close"] - ema20) / (ema20 + 1e-9) * 100

        return features.dropna()

    # ────────────────────────────────────────────────────
    #  تدريب النموذج
    # ────────────────────────────────────────────────────
    def train_model(self):
        df = self.fetch_rates()
        if df is None:
            print("❌  لا توجد بيانات للتدريب")
            return False

        features = self.extract_features(df)
        X = self.scaler.fit_transform(features.values)

        self.model = IsolationForest(
            n_estimators=150,
            contamination=0.05,
            max_samples=min(256, len(X)),
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        print(f"✅  النموذج مُدرَّب | {len(X)} شمعة | {features.shape[1]} ميزة")
        return True

    # ────────────────────────────────────────────────────
    #  تحليل الشمعة الأخيرة وإرسال الإشارة
    # ────────────────────────────────────────────────────
    def analyze_latest(self):
        df = self.fetch_rates()
        if df is None or self.model is None:
            return None

        features = self.extract_features(df)
        if len(features) == 0:
            return None

        X = self.scaler.transform(features.values)

        # درجة الشذوذ للشمعة الأخيرة
        score   = self.model.decision_function(X[-1:])  # موجب = طبيعي، سالب = شاذ
        is_anom = self.model.predict(X[-1:])[0] == -1

        last_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        direction  = "BUY" if last_close > prev_close else "SELL"

        signal = {
            "timestamp"    : datetime.now().isoformat(),
            "symbol"       : self.symbol,
            "is_anomaly"   : bool(is_anom),
            "anomaly_score": float(score[0]),
            "direction"    : direction if is_anom else "NONE",
            "close"        : float(last_close),
        }
        return signal

    # ────────────────────────────────────────────────────
    #  خادم ZeroMQ — يُرسل الإشارات إلى MT5
    # ────────────────────────────────────────────────────
    def start_server(self, retrain_every=50):
        socket = self.context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.zmq_port}")
        print(f"🌐  خادم ZeroMQ يستمع على المنفذ {self.zmq_port} …")

        self.is_running  = True
        tick_count       = 0

        while self.is_running:
            try:
                # انتظار طلب من MT5
                msg = socket.recv_string(flags=zmq.NOBLOCK)

                if msg == "GET_SIGNAL":
                    signal  = self.analyze_latest()
                    reply   = json.dumps(signal) if signal else '{"is_anomaly": false}'
                    socket.send_string(reply)

                    if signal and signal["is_anomaly"]:
                        print(f"⚠️  [{signal['timestamp'][:19]}] "
                              f"شذوذ مكتشَف | {signal['direction']} | "
                              f"score={signal['anomaly_score']:.4f}")

                elif msg == "TRAIN":
                    self.train_model()
                    socket.send_string('{"status": "trained"}')

                tick_count += 1
                if tick_count % retrain_every == 0:
                    print(f"🔄  إعادة تدريب النموذج (tick #{tick_count}) …")
                    threading.Thread(target=self.train_model, daemon=True).start()

            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                print(f"❌  خطأ: {e}")
                time.sleep(1)

        socket.close()

    def stop(self):
        self.is_running = False
        mt5.shutdown()
        print("🔴  الجسر مُوقَف")


# ══════════════════════════════════════════════════════════
#  الجسر بدون ZeroMQ (استخدام ملفات مؤقتة كبديل بسيط)
# ══════════════════════════════════════════════════════════

class FileBridge:
    """
    بديل بسيط لـ ZeroMQ — يكتب الإشارات في ملف JSON
    MT5 يقرأ الملف باستمرار (أبسط للمبتدئين)
    """
    SIGNAL_FILE = "C:/MT5_Signals/signal.json"  # يجب أن يطابق مسار MQL5

    def __init__(self, symbol="EURUSD"):
        self.symbol  = symbol
        self.model   = None
        self.scaler  = StandardScaler()
        import os
        os.makedirs("C:/MT5_Signals", exist_ok=True)

    def write_signal(self, signal: dict):
        with open(self.SIGNAL_FILE, "w") as f:
            json.dump(signal, f)

    def run_loop(self, interval_sec=5):
        """تشغيل الحلقة الرئيسية"""
        print("▶️  بدء حلقة الإشارات — اضغط Ctrl+C للإيقاف")
        while True:
            try:
                # هنا تضع منطق التحليل الخاص بك
                dummy_signal = {
                    "timestamp"    : datetime.now().isoformat(),
                    "symbol"       : self.symbol,
                    "is_anomaly"   : False,
                    "direction"    : "NONE",
                    "anomaly_score": 0.0,
                }
                self.write_signal(dummy_signal)
                time.sleep(interval_sec)
            except KeyboardInterrupt:
                print("⏹️  إيقاف")
                break


# ══════════════════════════════════════════════════════════
#  نقطة الدخول الرئيسية
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  🔗  Python ↔ MT5 Real-Time Bridge")
    print("=" * 60)

    MODE = "demo"  # غيّر إلى "live" للاستخدام الفعلي

    if MODE == "live":
        bridge = PythonMT5Bridge(
            symbol    = "EURUSD",
            timeframe = mt5.TIMEFRAME_M5,
            lookback  = 200,
            zmq_port  = 5555,
        )
        if bridge.connect_mt5():          # أضف login/password/server عند الحاجة
            bridge.train_model()
            bridge.start_server(retrain_every=100)
    else:
        # وضع التجربة — بدون MT5
        print("\n📋  وضع التجربة (بدون MT5)")
        print("    لتشغيل الجسر الفعلي:")
        print("    1. ثبّت MT5 على جهازك")
        print("    2. ثبّت: pip install MetaTrader5 pyzmq")
        print("    3. غيّر MODE = 'live'")
        print("    4. أضف بيانات حسابك في connect_mt5()")
        print("\n" + "=" * 60)
        print("  خطوات تثبيت كود MQL5 في MT5:")
        print("=" * 60)
        steps = [
            "افتح MT5 → Tools → MetaEditor (F4)",
            "File → New → Expert Advisor (template)",
            "الصق كود AnomalyTrader_EA.mq5",
            "اضغط F7 للترجمة",
            "في MT5: Navigator → Expert Advisors → الخبير",
            "اسحبه على الشارت واضبط الإعدادات",
            "تأكد من تفعيل Allow Algo Trading",
        ]
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")

# ══════════════════════════════════════════════════════════
#  كود MQL5 المصاحب لقراءة إشارات Python (يُضاف في EA)
# ══════════════════════════════════════════════════════════
MQL5_READER_CODE = """
// ── أضف هذا الكود داخل AnomalyTrader_EA.mq5 ──────────────

// لقراءة إشارة من ملف JSON (File Bridge):
bool ReadPythonSignal(string &direction, double &score)
{
   string path = "C:\\\\MT5_Signals\\\\signal.json";
   int    fh   = FileOpen(path, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE) return false;

   string content = "";
   while(!FileIsEnding(fh)) content += FileReadString(fh);
   FileClose(fh);

   // استخراج بسيط من JSON
   int posDir   = StringFind(content, "\"direction\": \"") + 14;
   int posScore = StringFind(content, "\"anomaly_score\": ") + 17;
   int posAnom  = StringFind(content, "\"is_anomaly\": ") + 14;

   if(posDir > 14)
   {
      int endPos = StringFind(content, "\"", posDir);
      direction  = StringSubstr(content, posDir, endPos - posDir);
   }
   if(posScore > 17)
   {
      int endPos = StringFind(content, ",", posScore);
      if(endPos < 0) endPos = StringFind(content, "}", posScore);
      score = StringToDouble(StringSubstr(content, posScore, endPos - posScore));
   }
   string anomStr = StringSubstr(content, posAnom, 4);
   return (anomStr == "true");
}

// استخدام في OnTick():
// string dir; double sc;
// bool isAnomaly = ReadPythonSignal(dir, sc);
// if(isAnomaly && dir == "BUY") trade.Buy(...);
"""
print("\n" + "─" * 60)
print("📄  كود MQL5 لقراءة إشارات Python:")
print("─" * 60)
print(MQL5_READER_CODE)
