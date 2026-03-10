# ============================================================
#  كشف الاحتيال المالي باستخدام Isolation Forest
#  Financial Fraud Detection using Isolation Forest
#  يعمل على Google Colab مباشرةً
# ============================================================

# ── تثبيت المكتبات (فك التعليق عند الحاجة في Colab) ──────
# !pip install scikit-learn matplotlib seaborn pandas numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.datasets import fetch_kddcup99          # بيانات حقيقية للشذوذ الشبكي
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── إعداد نمط الرسم البياني ───────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#e6edf3",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#e6edf3",
    "grid.color":       "#21262d",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
})

print("=" * 65)
print("  🔐  نظام كشف الاحتيال المالي — Isolation Forest")
print("=" * 65)

# ══════════════════════════════════════════════════════════
#  1. تحميل بيانات حقيقية  (KDD Cup 99 — Network Intrusion)
# ══════════════════════════════════════════════════════════
print("\n📥  جارٍ تحميل بيانات KDD Cup 99 (حقيقية من sklearn) …")

raw = fetch_kddcup99(subset="SA", percent10=True, as_frame=True)
df  = raw.frame.copy()
print(f"✅  تم تحميل {len(df):,} سجلاً | الأعمدة: {df.shape[1]}")

# ── ترميز الأعمدة النصية ─────────────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ── إنشاء تسمية ثنائية: 0 = طبيعي، 1 = شاذ (احتيال) ──────
target_col = "labels"
df["is_fraud"] = (df[target_col] != le.transform(["normal."])[0] 
                  if "normal." in le.classes_ else 
                  (df[target_col] != df[target_col].mode()[0])).astype(int)

X = df.drop(columns=[target_col, "is_fraud"])
y_true = df["is_fraud"].values

fraud_rate = y_true.mean() * 100
print(f"📊  نسبة الحالات الشاذة (احتيال): {fraud_rate:.2f}%")

# ══════════════════════════════════════════════════════════
#  2. تطبيع البيانات
# ══════════════════════════════════════════════════════════
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n⚙️   تم تطبيع البيانات باستخدام StandardScaler")

# ══════════════════════════════════════════════════════════
#  3. تدريب نموذج Isolation Forest
# ══════════════════════════════════════════════════════════
contamination = min(max(fraud_rate / 100, 0.01), 0.5)

print(f"\n🌲  جارٍ تدريب Isolation Forest")
print(f"    contamination = {contamination:.4f} | n_estimators = 200")

model = IsolationForest(
    n_estimators=200,
    contamination=contamination,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
    bootstrap=False,
)
model.fit(X_scaled)

# ── التنبؤ: −1 = شاذ → 1 | 1 = طبيعي → 0 ────────────────
raw_pred      = model.predict(X_scaled)
y_pred        = (raw_pred == -1).astype(int)
anomaly_score = model.decision_function(X_scaled)   # أعلى = أكثر طبيعية

print(f"\n📈  الحالات المكتشفة كاحتيال : {y_pred.sum():,}")
print(f"    الحالات الطبيعية          : {(y_pred == 0).sum():,}")

# ══════════════════════════════════════════════════════════
#  4. تقييم الأداء
# ══════════════════════════════════════════════════════════
print("\n" + "─" * 45)
print("  📋  تقرير الأداء")
print("─" * 45)
print(classification_report(y_true, y_pred,
                             target_names=["طبيعي", "احتيال"]))

try:
    auc = roc_auc_score(y_true, -anomaly_score)
    print(f"  AUC-ROC : {auc:.4f}")
except Exception:
    pass

# ══════════════════════════════════════════════════════════
#  5. تخفيض الأبعاد بـ PCA للرسم البياني
# ══════════════════════════════════════════════════════════
print("\n🔬  تخفيض الأبعاد بـ PCA (2 مكوّنان) …")
sample_n = min(15_000, len(X_scaled))
idx      = np.random.choice(len(X_scaled), sample_n, replace=False)

pca   = PCA(n_components=2, random_state=42)
X_2d  = pca.fit_transform(X_scaled[idx])
scores_s = anomaly_score[idx]
preds_s  = y_pred[idx]
truth_s  = y_true[idx]

explained = pca.explained_variance_ratio_.sum() * 100
print(f"✅  التباين الموضَّح: {explained:.1f}%  |  عيّنة: {sample_n:,} سجل")

# ══════════════════════════════════════════════════════════
#  6. الرسوم البيانية (4 لوحات)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("🔐  نظام كشف الاحتيال المالي — Isolation Forest\n"
             "Financial Fraud Detection System",
             fontsize=15, fontweight="bold", color="#58a6ff", y=1.01)

# ── اللوحة 1: PCA — طبيعي vs احتيال ─────────────────────
ax1 = axes[0, 0]
mask_n = preds_s == 0
mask_f = preds_s == 1

ax1.scatter(X_2d[mask_n, 0], X_2d[mask_n, 1],
            c="#238636", s=6, alpha=0.35, label=f"طبيعي ({mask_n.sum():,})")
ax1.scatter(X_2d[mask_f, 0], X_2d[mask_f, 1],
            c="#f85149", s=18, alpha=0.8,  marker="x",
            label=f"احتيال مكتشَف ({mask_f.sum():,})")

ax1.set_title("توزيع الحالات الشاذة — PCA 2D", color="#58a6ff", fontweight="bold")
ax1.set_xlabel(f"المكوّن 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"المكوّن 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── اللوحة 2: توزيع Anomaly Score ─────────────────────────
ax2 = axes[0, 1]
bins = 80
ax2.hist(anomaly_score[y_pred == 0], bins=bins, color="#238636",
         alpha=0.7, label="طبيعي",   density=True)
ax2.hist(anomaly_score[y_pred == 1], bins=bins, color="#f85149",
         alpha=0.8, label="احتيال",  density=True)

thresh = np.percentile(anomaly_score, contamination * 100)
ax2.axvline(thresh, color="#d29922", linewidth=2, linestyle="--",
            label=f"حد الكشف: {thresh:.3f}")

ax2.set_title("توزيع درجات الشذوذ", color="#58a6ff", fontweight="bold")
ax2.set_xlabel("Anomaly Score (أعلى = أكثر طبيعية)")
ax2.set_ylabel("الكثافة")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── اللوحة 3: Confusion Matrix ────────────────────────────
ax3 = axes[1, 0]
cm   = confusion_matrix(y_true, y_pred)
cm_p = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

im = ax3.imshow(cm_p, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
plt.colorbar(im, ax=ax3, label="%")

labels = [["True Negative\n(طبيعي صحيح)",  "False Positive\n(إنذار خاطئ)"],
          ["False Negative\n(احتيال فائت)", "True Positive\n(احتيال مكتشَف)"]]
for i in range(2):
    for j in range(2):
        ax3.text(j, i, f"{cm[i,j]:,}\n({cm_p[i,j]:.1f}%)",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color="black" if cm_p[i,j] > 40 else "white")

ax3.set_xticks([0, 1]); ax3.set_xticklabels(["تنبؤ: طبيعي", "تنبؤ: احتيال"])
ax3.set_yticks([0, 1]); ax3.set_yticklabels(["فعلي: طبيعي", "فعلي: احتيال"])
ax3.set_title("مصفوفة التباين (Confusion Matrix)", color="#58a6ff", fontweight="bold")

# ── اللوحة 4: أعلى 20 معاملة احتيال حسب الشذوذ ──────────
ax4 = axes[1, 1]
fraud_idx    = np.where(y_pred == 1)[0]
top20_idx    = fraud_idx[np.argsort(anomaly_score[fraud_idx])[:20]]
top20_scores = anomaly_score[top20_idx]
colors_bar   = ["#f85149" if s < thresh else "#d29922" for s in top20_scores]

bars = ax4.barh(range(20), -top20_scores,
                color=colors_bar, edgecolor="#30363d", linewidth=0.5)
ax4.set_yticks(range(20))
ax4.set_yticklabels([f"معاملة #{i+1}" for i in range(20)], fontsize=8)
ax4.set_xlabel("شدة الشذوذ (مقلوب Anomaly Score)")
ax4.set_title("أعلى 20 حالة احتيال — أكثرها خطورةً", color="#58a6ff", fontweight="bold")
ax4.grid(True, alpha=0.3, axis="x")

red_p   = mpatches.Patch(color="#f85149", label="احتيال عالي الخطورة")
yell_p  = mpatches.Patch(color="#d29922", label="احتيال متوسط")
ax4.legend(handles=[red_p, yell_p], fontsize=8)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/fraud_detection_results.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\n✅  تم حفظ الرسم البياني: fraud_detection_results.png")

# ══════════════════════════════════════════════════════════
#  7. ملخص نهائي
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  📊  ملخص النتائج")
print("=" * 65)
print(f"  إجمالي السجلات المحللة  : {len(X_scaled):>10,}")
print(f"  حالات طبيعية            : {(y_pred==0).sum():>10,}")
print(f"  حالات احتيال مكتشَفة    : {(y_pred==1).sum():>10,}")
print(f"  نسبة الاحتيال المكتشَفة : {y_pred.mean()*100:>9.2f}%")
print(f"  حد الكشف (threshold)   : {thresh:>10.4f}")
print("=" * 65)
print("\n💡  لتشغيل الكود على Google Colab:")
print("    1. افتح colab.research.google.com")
print("    2. أنشئ Notebook جديد")
print("    3. الصق الكود وشغّله مباشرةً ✅")
