import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_CSV_PATH = "./data/initial_data_warehouse_ResNet50.csv"
DEFAULT_OUTPUT_PATH = "design_space_sparsity_resnet50.pdf"
DEFAULT_OBJECTIVE = "edp"
HIGHLIGHT_POINT_SIZE = 9

METRIC_COLUMNS = {
	"latency",
	"energy",
	"area",
	"power",
	"cnt_pes",
	"l1_mem",
	"l2_mem",
	"edp",
	"pe_util",
	"noc_bw_req",
	"l1_mem_req",
	"l2_mem_req",
}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Plot PCA projection of DNN accelerator design space and highlight sparse high-value designs."
	)
	parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input design warehouse CSV file.")
	parser.add_argument("--objective", default=DEFAULT_OBJECTIVE, help="Objective column. Lower is treated as better.")
	parser.add_argument("--top", type=float, default=0.0087, help="Top design ratio to highlight in orange.")
	parser.add_argument("--elite", type=float, default=0.0030, help="Top design ratio to highlight in red.")
	parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output figure path.")
	parser.add_argument("--dpi", type=int, default=600, help="Saved figure DPI.")
	return parser.parse_args()


def get_design_parameter_columns(dataframe, objective):
	excluded_columns = set(METRIC_COLUMNS)
	excluded_columns.add(objective)
	feature_columns = [
		column for column in dataframe.columns
		if column not in excluded_columns and pd.api.types.is_numeric_dtype(dataframe[column])
	]
	if not feature_columns:
		raise ValueError("No numeric design parameter columns found for PCA.")
	return feature_columns


def select_top_indices(objective_values, ratio):
	if not (0.0 < ratio < 1.0):
		raise ValueError("Top ratio must be in (0, 1).")
	count = max(1, int(np.ceil(len(objective_values) * ratio)))
	return np.argsort(objective_values)[:count]


def main():
	args = parse_args()
	dataframe = pd.read_csv(args.csv)
	if args.objective not in dataframe.columns:
		raise ValueError(f"Objective column {args.objective!r} does not exist in {args.csv}.")

	feature_columns = get_design_parameter_columns(dataframe, args.objective)
	features = dataframe[feature_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
	valid_index = features.index
	objective_values = dataframe.loc[valid_index, args.objective].to_numpy(dtype=np.float64)

	scaled_features = StandardScaler().fit_transform(features.to_numpy(dtype=np.float64))
	projected_features = PCA(n_components=2, random_state=0).fit_transform(scaled_features)
	projected_features = projected_features - projected_features.min(axis=0)

	top_indices = select_top_indices(objective_values, args.top)
	elite_indices = select_top_indices(objective_values, args.elite)
	top_mask = np.zeros(len(objective_values), dtype=bool)
	elite_mask = np.zeros(len(objective_values), dtype=bool)
	top_mask[top_indices] = True
	elite_mask[elite_indices] = True
	top_only_mask = top_mask & ~elite_mask

	plt.rcParams.update({
		"font.family": "Times New Roman",
		"font.size": 9,
		"axes.labelsize": 10,
		"xtick.labelsize": 8,
		"ytick.labelsize": 8,
		"legend.fontsize": 8,
		"axes.linewidth": 1.0,
		"figure.dpi": 300,
		"savefig.dpi": args.dpi,
		"svg.fonttype": "none",
	})

	fig, ax = plt.subplots(figsize=(3.5, 2.75))
	ax.scatter(
		projected_features[:, 0],
		projected_features[:, 1],
		s=8,
		c="lightgray",
		alpha=0.42,
		linewidths=0,
		label=f"All sampled designs ({len(objective_values)})",
	)
	ax.scatter(
		projected_features[top_only_mask, 0],
		projected_features[top_only_mask, 1],
		s=HIGHLIGHT_POINT_SIZE,
		c="#f2a51a",
		alpha=0.75,
		linewidths=0,
		label=f"Top {args.top * 100:.2f}% designs ({int(top_mask.sum())})",
	)
	ax.scatter(
		projected_features[elite_mask, 0],
		projected_features[elite_mask, 1],
		s=HIGHLIGHT_POINT_SIZE,
		c="#d62728",
		alpha=0.9,
		edgecolors="black",
		linewidths=0.25,
		label=f"Top {args.elite * 100:.2f}% designs ({int(elite_mask.sum())})",
	)

	ax.set_xlabel("PCA Dimension 1", fontweight="bold")
	ax.set_ylabel("PCA Dimension 2", fontweight="bold")
	ax.set_xlim(left=0)
	ax.set_ylim(bottom=0)
	ax.grid(True, linestyle="-", linewidth=0.35, alpha=0.28)
	legend_handles = [
		Line2D([0], [0], marker="o", color="none", markerfacecolor="lightgray", markeredgecolor="none", markersize=4, label="Reward < 0.85"),
		Line2D([0], [0], marker="o", color="none", markerfacecolor="#f2a51a", markeredgecolor="none", markersize=5, label="Reward \u2265 0.85"),
		Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markeredgecolor="black", markeredgewidth=0.4, markersize=5, label="Reward \u2265 0.95"),
	]
	ax.legend(handles=legend_handles, loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.2)

	for spine in ax.spines.values():
		spine.set_linewidth(1.0)
	for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
		tick_label.set_fontweight("bold")

	plt.tight_layout()
	plt.savefig(args.output, bbox_inches="tight")
	print(f"Saved {args.output}")
	print(f"Design parameter dimensions: {len(feature_columns)}")
	print(f"Top {args.top * 100:.2f}% threshold ({args.objective}, lower is better): {objective_values[top_indices[-1]]:.6e}")
	print(f"Top {args.elite * 100:.2f}% threshold ({args.objective}, lower is better): {objective_values[elite_indices[-1]]:.6e}")


if __name__ == "__main__":
	main()
