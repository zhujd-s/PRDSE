import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# =============================
#   用户需要修改的部分
# =============================
CSV_PATH_1 = "./data/initial_data_warehouse_ResNet50.csv"   # 换成你的 CSV 路径
CSV_FILES = [
    "./data/initial_data_warehouse_VGG16.csv",
    "./data/initial_data_warehouse_MobileNetV2.csv",
    "./data/initial_data_warehouse_MnasNet.csv",
    "./data/initial_data_warehouse_ResNet50.csv",
    "./data/initial_data_warehouse_Transformer.csv",    
    "./data/initial_data_warehouse_GNMT.csv"
]
COLUMN_EDP = "edp"     # 每个文件的 EDP 列名
DRAW_SHADOW = False     # 阴影区域开关（True/False）
VALUE_THRESHOLD = 0.9   # 高价值区阈值（0~1）
MODEL_NAMES = [
    "VGG16",
    "MobileNet",
    "MnasNet",
    "ResNet50",
    "Trans",
    "GNMT"
]
# =============================
TOP_PERCENT = 0.10   # Top 10%
DRAW_SHADOW = False


def load_and_process(csv_path, edp_col):
    df = pd.read_csv(csv_path)
    edp = df[edp_col].values
    edp = np.sort(edp)[100:9900]
    # 价值 = 1/EDP（EDP 越低越好）
    v = 1.0 / edp

    # 归一化
    v = v / v.max()

    # 排序
    v_sorted = np.sort(v)
    cdf = np.arange(1, len(v_sorted) + 1) / len(v_sorted)
    return v_sorted, cdf


# ====== 读取六个文件 ======
data_list = []
all_values_concat = []
for csv_path in CSV_FILES:
    v, cdf = load_and_process(csv_path, COLUMN_EDP)
    data_list.append((v, cdf))
    all_values_concat.extend(v)

all_values_concat = np.array(all_values_concat)
n_all = len(all_values_concat)

# ====== 共用 Top 10% cutoff ======
global_sorted = np.sort(all_values_concat)
global_threshold_idx = int((1 - TOP_PERCENT) * n_all)
global_threshold_value = global_sorted[global_threshold_idx]

# ====== 计算每条 CDF 在该 cutoff 下的 CDF 值 ======

ratio_all = 0.0
for v_sorted, cdf in data_list:
    # 找到第一个 v >= cutoff 的CDF
    idx = np.searchsorted(v_sorted, global_threshold_value, side="left")
    idx = min(idx, len(cdf) - 1)
    ratio = np.mean(v_sorted > 0.9)
    ratio_all += ratio
    print("value > 0.9 的占比 =", ratio)

print(ratio_all / len(data_list))




# ====== 绘图：IEEE 单栏风格 ======
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 600
})

fig, ax = plt.subplots(figsize=(3.5, 2.5))
plt.subplots_adjust(top=0.80)  # 为图例腾出空间
colors = [
    "#1f77b4",  # 蓝
    "#ff7f0e",  # 橙
    "#2ca02c",  # 绿
    "#d62728",  # 红
    "#9467bd",  # 紫
    "#8c564b"   # 棕
]
# 六种 line style（单栏图中容易区分）
linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]


# ====== 绘制每条 CDF ======
for i, (v_sorted, cdf) in enumerate(data_list):
    ax.plot(
        v_sorted, cdf,
        linestyle="-",
        linewidth=1.2,
        color=colors[i],
        label=MODEL_NAMES[i]
    )



legend = ax.legend(
    loc='upper right',
    ncol=3,
    bbox_to_anchor=(1.02, 1.15),
    frameon=False,
    columnspacing=1.0,
    handlelength=1.5,
    labelspacing=0.3,
    borderpad=0.2,
    prop={'weight':'bold',
          'size':6}
)

legend._legend_box.align = "center"

# ====== 共用 Top10% 竖线 ======
ax.axvline(0.9, color='red', linestyle='--', linewidth=1)

ax.text(
    VALUE_THRESHOLD, 0.03,
    f"Reward Value = 0.9",
    rotation=90,
    ha='right', va='bottom',
    fontsize=6, color='red',
    fontweight="bold"
)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

for label in ax.get_xticklabels():
    label.set_fontweight("bold")

for label in ax.get_yticklabels():
    label.set_fontweight("bold")

ax.tick_params(axis='both', labelsize=6)   # 设置所有刻度数字大小

# ====== 阴影区域 ======
if DRAW_SHADOW:
    ax.axvspan(global_threshold_value, 1.0, color="red", alpha=0.12)


# ====== Label ======
ax.set_xlabel("Reward", fontsize=6, fontweight="bold")
ax.set_ylabel("CDF Value", fontsize=6, fontweight="bold")

ax.grid(True, linestyle="-", alpha=0.4, linewidth=0.5)
# ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("cdf_design_point.pdf", bbox_inches="tight")
plt.show()
