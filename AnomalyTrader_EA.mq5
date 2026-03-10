//+------------------------------------------------------------------+
//|  AnomalyTrader_EA.mq5                                            |
//|  روبوت تداول يكشف الحركات السعرية الشاذة (Isolation Forest Logic)|
//|  Anomaly-Based Trading Robot — MT5 Expert Advisor                |
//+------------------------------------------------------------------+
#property copyright "Anomaly Trader System"
#property version   "2.00"
#property description "كشف الحركات السعرية الشاذة والتداول بناءً عليها"

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Math\Stat\Math.mqh>

//────────────────────────────────────────────────────────────────────
//  مدخلات المستخدم
//────────────────────────────────────────────────────────────────────
input group "═══ إعدادات كشف الشذوذ ═══"
input int     InpLookback         = 100;    // نافذة البيانات التاريخية
input double  InpContamination    = 0.05;   // نسبة الشذوذ المتوقعة (0.01–0.20)
input int     InpFeatureWindow    = 14;     // نافذة حساب المؤشرات
input double  InpAnomalyThreshold = 2.5;   // عتبة درجة الشذوذ (z-score)

input group "═══ إعدادات التداول ═══"
input double  InpLotSize          = 0.10;   // حجم اللوت
input int     InpSL_Points        = 150;    // وقف الخسارة (نقاط)
input int     InpTP_Points        = 300;    // هدف الربح (نقاط)
input int     InpMaxPositions     = 3;      // أقصى عدد صفقات مفتوحة
input bool    InpUseTrailingStop  = true;   // استخدام trailing stop
input int     InpTrailingPoints   = 80;     // trailing stop (نقاط)

input group "═══ فلاتر إضافية ═══"
input int     InpRSI_Period       = 14;     // فترة RSI
input double  InpRSI_OB          = 70.0;   // مستوى التشبع الشرائي
input double  InpRSI_OS          = 30.0;   // مستوى التشبع البيعي
input bool    InpUseTrendFilter   = true;   // فلتر الاتجاه (EMA)
input int     InpTrendEMA        = 50;     // فترة EMA للاتجاه
input int     InpMagicNumber     = 20250101; // Magic Number

//────────────────────────────────────────────────────────────────────
//  متغيرات عامة
//────────────────────────────────────────────────────────────────────
CTrade         trade;
CPositionInfo  posInfo;

int    hRSI, hEMA, hATR;
double anomalyScores[];          // درجات الشذوذ
double featureMatrix[][5];       // مصفوفة المؤشرات (5 ميزات)

datetime lastBarTime = 0;

//────────────────────────────────────────────────────────────────────
//  دالة الحسابات الإحصائية المساعدة
//────────────────────────────────────────────────────────────────────
double ArrayMean(const double &arr[], int size)
{
   double s = 0;
   for(int i = 0; i < size; i++) s += arr[i];
   return size > 0 ? s / size : 0;
}

double ArrayStdDev(const double &arr[], int size, double mean)
{
   double s = 0;
   for(int i = 0; i < size; i++) s += MathPow(arr[i] - mean, 2);
   return size > 1 ? MathSqrt(s / (size - 1)) : 1e-9;
}

//────────────────────────────────────────────────────────────────────
//  استخراج ميزات السوق (Feature Engineering)
//  يحاكي منطق Isolation Forest بحساب z-score متعدد الأبعاد
//────────────────────────────────────────────────────────────────────
bool ExtractFeatures(int lookback)
{
   double close[], high[], low[], volume[];
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, lookback + InpFeatureWindow, close)   < 0) return false;
   if(CopyHigh (_Symbol, PERIOD_CURRENT, 0, lookback + InpFeatureWindow, high)    < 0) return false;
   if(CopyLow  (_Symbol, PERIOD_CURRENT, 0, lookback + InpFeatureWindow, low)     < 0) return false;
   if(CopyTickVolume(_Symbol, PERIOD_CURRENT, 0, lookback + InpFeatureWindow, volume) < 0) return false;

   ArrayResize(featureMatrix, lookback);
   int offset = InpFeatureWindow;

   for(int i = 0; i < lookback; i++)
   {
      int idx = offset + i;

      // ميزة 1: نسبة التغير السعري (%)
      double priceChange = (close[idx] - close[idx-1]) / (close[idx-1] + 1e-9) * 100.0;

      // ميزة 2: مدى الشمعة نسبةً للمتوسط (ATR نسبي)
      double candleRange = (high[idx] - low[idx]);
      double avgRange    = 0;
      for(int k = 1; k <= InpFeatureWindow; k++) avgRange += (high[idx-k] - low[idx-k]);
      avgRange /= InpFeatureWindow;
      double relRange = (avgRange > 0) ? candleRange / avgRange : 1.0;

      // ميزة 3: ضغط الحجم (Volume Surge)
      double avgVol = 0;
      for(int k = 1; k <= InpFeatureWindow; k++) avgVol += volume[idx-k];
      avgVol /= InpFeatureWindow;
      double volSurge = (avgVol > 0) ? volume[idx] / avgVol : 1.0;

      // ميزة 4: الفجوة السعرية (Gap)
      double gap = MathAbs(close[idx] - close[idx-1]) / (close[idx-1] + 1e-9) * 100.0;

      // ميزة 5: الزخم (Momentum)
      double momentum = (close[idx] - close[idx - InpFeatureWindow]) /
                        (close[idx - InpFeatureWindow] + 1e-9) * 100.0;

      featureMatrix[i][0] = priceChange;
      featureMatrix[i][1] = relRange;
      featureMatrix[i][2] = volSurge;
      featureMatrix[i][3] = gap;
      featureMatrix[i][4] = momentum;
   }
   return true;
}

//────────────────────────────────────────────────────────────────────
//  حساب درجة الشذوذ (يحاكي منطق Isolation Forest)
//  كلما ارتفعت الدرجة كلما كانت النقطة أكثر شذوذاً
//────────────────────────────────────────────────────────────────────
void ComputeAnomalyScores(int lookback)
{
   ArrayResize(anomalyScores, lookback);
   int nFeatures = 5;

   // حساب المتوسط والانحراف المعياري لكل ميزة
   double means[5], stds[5];
   for(int f = 0; f < nFeatures; f++)
   {
      double col[];
      ArrayResize(col, lookback);
      for(int i = 0; i < lookback; i++) col[i] = featureMatrix[i][f];
      means[f] = ArrayMean(col, lookback);
      stds[f]  = ArrayStdDev(col, lookback, means[f]);
   }

   // درجة الشذوذ = متوسط القيم المطلقة للـ z-score لكل الميزات
   for(int i = 0; i < lookback; i++)
   {
      double score = 0;
      for(int f = 0; f < nFeatures; f++)
      {
         double z = (stds[f] > 1e-9) ? MathAbs((featureMatrix[i][f] - means[f]) / stds[f]) : 0;
         score += z;
      }
      anomalyScores[i] = score / nFeatures;
   }
}

//────────────────────────────────────────────────────────────────────
//  عدد الصفقات المفتوحة الحالية
//────────────────────────────────────────────────────────────────────
int CountOpenPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(posInfo.SelectByIndex(i) &&
         posInfo.Symbol() == _Symbol &&
         posInfo.Magic()  == InpMagicNumber) count++;
   return count;
}

//────────────────────────────────────────────────────────────────────
//  تطبيق Trailing Stop
//────────────────────────────────────────────────────────────────────
void ApplyTrailingStop()
{
   double trailDist = InpTrailingPoints * _Point;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!posInfo.SelectByIndex(i)) continue;
      if(posInfo.Symbol() != _Symbol || posInfo.Magic() != InpMagicNumber) continue;

      double currentSL = posInfo.StopLoss();
      double openPrice = posInfo.PriceOpen();
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      if(posInfo.PositionType() == POSITION_TYPE_BUY)
      {
         double newSL = bid - trailDist;
         if(newSL > currentSL + _Point)
            trade.PositionModify(posInfo.Ticket(), newSL, posInfo.TakeProfit());
      }
      else if(posInfo.PositionType() == POSITION_TYPE_SELL)
      {
         double newSL = ask + trailDist;
         if(newSL < currentSL - _Point || currentSL == 0)
            trade.PositionModify(posInfo.Ticket(), newSL, posInfo.TakeProfit());
      }
   }
}

//────────────────────────────────────────────────────────────────────
//  التهيئة
//────────────────────────────────────────────────────────────────────
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(10);

   hRSI = iRSI(_Symbol, PERIOD_CURRENT, InpRSI_Period, PRICE_CLOSE);
   hEMA = iMA (_Symbol, PERIOD_CURRENT, InpTrendEMA, 0, MODE_EMA, PRICE_CLOSE);
   hATR = iATR(_Symbol, PERIOD_CURRENT, 14);

   if(hRSI == INVALID_HANDLE || hEMA == INVALID_HANDLE || hATR == INVALID_HANDLE)
   {
      Print("❌ خطأ في تهيئة المؤشرات");
      return INIT_FAILED;
   }

   Print("✅ AnomalyTrader EA جاهز — ", _Symbol, " | ", EnumToString(Period()));
   return INIT_SUCCEEDED;
}

//────────────────────────────────────────────────────────────────────
//  الحدث الرئيسي — OnTick
//────────────────────────────────────────────────────────────────────
void OnTick()
{
   // ── تشغيل على شمعة جديدة فقط ────────────────────────────
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBar == lastBarTime) return;
   lastBarTime = currentBar;

   // ── Trailing Stop ────────────────────────────────────────
   if(InpUseTrailingStop) ApplyTrailingStop();

   // ── فحص عدد الصفقات ─────────────────────────────────────
   if(CountOpenPositions() >= InpMaxPositions) return;

   // ── استخراج الميزات وحساب الشذوذ ─────────────────────────
   if(!ExtractFeatures(InpLookback)) return;
   ComputeAnomalyScores(InpLookback);

   // الشمعة الأخيرة المكتملة (index = lookback - 1 في المصفوفة)
   double latestScore = anomalyScores[InpLookback - 1];
   double latestPriceChange = featureMatrix[InpLookback - 1][0];

   // ── فلتر RSI ─────────────────────────────────────────────
   double rsiVal[];
   if(CopyBuffer(hRSI, 0, 1, 1, rsiVal) < 0) return;
   double rsi = rsiVal[0];

   // ── فلتر الاتجاه (EMA) ───────────────────────────────────
   double emaVal[];
   if(InpUseTrendFilter && CopyBuffer(hEMA, 0, 1, 1, emaVal) < 0) return;
   double ema   = InpUseTrendFilter ? emaVal[0] : 0;
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // ── شرط الشذوذ ───────────────────────────────────────────
   bool isAnomaly = (latestScore >= InpAnomalyThreshold);
   if(!isAnomaly) return;

   Print("⚠️  شذوذ مكتشَف | درجة: ", DoubleToString(latestScore, 3),
         " | تغيير السعر: ", DoubleToString(latestPriceChange, 4), "%",
         " | RSI: ", DoubleToString(rsi, 1));

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double slPts = InpSL_Points * _Point;
   double tpPts = InpTP_Points * _Point;

   // ── منطق الدخول ─────────────────────────────────────────
   // شراء: سعر يتحرك بقوة للأعلى + RSI غير مشبع + فوق EMA
   bool buySignal  = (latestPriceChange > 0) &&
                     (rsi < InpRSI_OB) &&
                     (!InpUseTrendFilter || price > ema);

   // بيع: سعر يتحرك بقوة للأسفل + RSI غير مشبع + تحت EMA
   bool sellSignal = (latestPriceChange < 0) &&
                     (rsi > InpRSI_OS) &&
                     (!InpUseTrendFilter || price < ema);

   if(buySignal)
   {
      double sl = ask - slPts;
      double tp = ask + tpPts;
      if(trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "AnomalyBuy"))
         Print("✅  صفقة شراء مفتوحة | ask=", ask, " | SL=", sl, " | TP=", tp);
   }
   else if(sellSignal)
   {
      double sl = bid + slPts;
      double tp = bid - tpPts;
      if(trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "AnomalySell"))
         Print("✅  صفقة بيع مفتوحة | bid=", bid, " | SL=", sl, " | TP=", tp);
   }
}

//────────────────────────────────────────────────────────────────────
//  تنظيف عند الإغلاق
//────────────────────────────────────────────────────────────────────
void OnDeinit(const int reason)
{
   IndicatorRelease(hRSI);
   IndicatorRelease(hEMA);
   IndicatorRelease(hATR);
   Print("🔴 AnomalyTrader EA أُوقف | السبب: ", reason);
}

//────────────────────────────────────────────────────────────────────
//  إحصائيات في لوحة المعلومات (OnChartEvent)
//────────────────────────────────────────────────────────────────────
void OnChartEvent(const int id, const long &lparam,
                  const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_CHART_CHANGE)
   {
      double rsiVal[]; CopyBuffer(hRSI, 0, 1, 1, rsiVal);
      string info = StringFormat(
         "AnomalyTrader EA\n"
         "─────────────────────\n"
         "الرمز    : %s\n"
         "الصفقات  : %d / %d\n"
         "RSI      : %.1f\n"
         "عتبة الشذوذ: %.1f\n"
         "─────────────────────",
         _Symbol,
         CountOpenPositions(), InpMaxPositions,
         ArraySize(rsiVal) > 0 ? rsiVal[0] : 0.0,
         InpAnomalyThreshold
      );
      Comment(info);
   }
}
//+------------------------------------------------------------------+
