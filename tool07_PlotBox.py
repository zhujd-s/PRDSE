import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 用户需要修改的部分
# =============================
CSV_PATH = "./data/initial_data_warehouse_VGG16.csv"  # 换成你的 CSV 文件路径
COLUMN_EDP = "edp"                 # 或 "latency"
USE_VALUE = True                   # True → 使用 1/EDP, False → 使用 EDP
# =============================

# ====== 读取数据 ======
df = pd.read_csv(CSV_PATH)

if COLUMN_EDP not in df.columns:
    raise ValueError(f"列名 {COLUMN_EDP} 不在 CSV 中，请检查！")

edp = df[COLUMN_EDP].values

# ====== 转换成 value = 1/EDP ======
if USE_VALUE:
    values = 1.0 / edp
else:
    values = edp

# ====== 绘制箱线图 ======
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
})

plt.figure(figsize=(3.5, 2.8))

plt.boxplot(
    values,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="lightgray", color="black"),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(marker='o', markerfacecolor='blue', markersize=4, linestyle='none')
)

ylabel = "Value (1/EDP)" if USE_VALUE else "EDP"
plt.ylabel(ylabel)
plt.title("Boxplot (with Outliers)", fontsize=12)
plt.tight_layout()

plt.savefig("boxplot_edp.pdf", bbox_inches='tight')
plt.show()


# ====== 离群点检测 ======
Q1 = np.percentile(values, 25)
Q3 = np.percentile(values, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = np.where((values < lower_bound) | (values > upper_bound))[0]

print("\n===== 离群点检测结果 =====")
print(f"下界: {lower_bound:.6f}, 上界: {upper_bound:.6f}")
print(f"离群点数量: {len(outliers)}")

if len(outliers) > 0:
    print("离群点位置与对应值：")
    for idx in outliers:
        print(f"Index {idx}: {values[idx]}")
else:
    print("无明显离群点。")
