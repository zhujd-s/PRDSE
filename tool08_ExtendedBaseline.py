import os
import pandas
import numpy

from config import config_global

import sys
sys.path.append("./util/")
from space import create_space_maestro
from evaluation_maestro import evaluation_maestro


EXTRA_METRICS = ["throughput", "throughput_per_energy", "offchip_bw_req"]
MODEL_NAMES = ["VGG16", "MobileNetV2", "MnasNet", "ResNet50", "Transformer", "GNMT"]
TARGET = "cloud"
SAMPLE_COUNT = 200
SAMPLE_SEED = 20250608
INPUT_TEMPLATE = "data/initial_data_warehouse_{}.csv"
OUTPUT_TEMPLATE = "data/baseline_extended_{}.csv"


def _build_status(row, status_keys):
	status = {}
	for key in status_keys:
		value = row[key]
		if pandas.isna(value):
			raise ValueError(f"invalid status value for key={key}")
		status[key] = int(round(float(value)))
	return status


def generate_extended_baseline(nnmodel, sample_count=SAMPLE_COUNT, sample_seed=SAMPLE_SEED, target=TARGET):
	input_path = INPUT_TEMPLATE.format(nnmodel)
	output_path = OUTPUT_TEMPLATE.format(nnmodel)
	assert(os.path.exists(input_path)), f"missing input sample pool: {input_path}"

	design_space = create_space_maestro(nnmodel, target=target)
	status_keys = list(design_space.get_status().keys())
	dataframe = pandas.read_csv(input_path)
	assert(all(key in dataframe.columns for key in status_keys)), f"input file {input_path} missing status columns"

	sampled_dataframe = dataframe.sample(
		n=min(sample_count, len(dataframe)),
		random_state=sample_seed,
		replace=False,
	)

	evaluator = evaluation_maestro(iindex=0, nnmodel=nnmodel, pid=os.getpid(), space=design_space)
	records = []
	for index, (_, row) in enumerate(sampled_dataframe.iterrows(), start=1):
		print(f"[{nnmodel}] evaluating sample {index}/{len(sampled_dataframe)}", end="\r")
		status = _build_status(row, status_keys)
		metrics = evaluator.evaluate(status)
		if(metrics is None):
			continue
		records.append([metrics[name] for name in EXTRA_METRICS])
	print("")

	assert(records), f"no valid samples collected for {nnmodel}"
	metrics_array = numpy.array(records, dtype=numpy.float64)

	data = {}
	for metric_index, metric_name in enumerate(EXTRA_METRICS):
		values = metrics_array[:, metric_index]
		data[metric_name] = [float(numpy.max(values))]
		data[f"{metric_name}_avg"] = [float(numpy.mean(values))]
		data[f"{metric_name}_median"] = [float(numpy.median(values))]
	data["sample_count"] = [int(len(records))]
	data["sample_seed"] = [int(sample_seed)]

	output_dataframe = pandas.DataFrame(data)
	output_dataframe.to_csv(output_path, index=None)
	print(f"saved {output_path}")


def main():
	global_config = config_global(is_setup=True)
	_ = global_config  # keep one place to follow repo config style
	for nnmodel in MODEL_NAMES:
		generate_extended_baseline(nnmodel)


if __name__ == "__main__":
	main()
