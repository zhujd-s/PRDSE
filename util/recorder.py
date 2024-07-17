import numpy as np
import pandas

def writelog(log_table, iindex):
	path = "./record/log/log_table_{}.csv".format(iindex)
	log_table = pandas.DataFrame(log_table)
	log_table.transpose()
	log_table.to_csv(path, index = None, header = None)

def	recorder(algoname, global_config, objective_record, timecost_record):
	SCEN_NUM = global_config.SCEN_NUM
	SCEN_TYPE = global_config.SCEN_TYPE
	PASS = global_config.PASS
	objective_path = "./record/objectvalue/{}_{}.csv".format(algoname, global_config.goal)
	py_objective_record = list()
	py_objective_record = objective_record[0:len(objective_record)]
	py_objective_record.sort(key = lambda olist:olist[-1])
	py_objective_record = np.array(py_objective_record).T
	objective_df = pandas.DataFrame(py_objective_record)
	actual_SCEN_TYPE = SCEN_TYPE - int(len(PASS)/SCEN_NUM)
	for scen in range(actual_SCEN_TYPE):
		objective_df["avg_{}".format(scen)] = objective_df.iloc[:, scen*SCEN_NUM:(scen+1)*SCEN_NUM].mean(axis=1)
		#objective_df["min_{}".format(scen)] = objective_df.iloc[:, scen*SCEN_NUM:(scen+1)*SCEN_NUM].min(axis=1)
	objective_df.to_csv(objective_path, index = None)

	timecost_path = "./record/timecost/{}_{}.csv".format(algoname, global_config.goal)
	py_timecost_record = list()
	py_timecost_record = timecost_record[0:len(timecost_record)]
	py_timecost_record.sort(key = lambda olist:olist[-1])
	py_timecost_record = np.array(py_timecost_record).T
	timecost_df = pandas.DataFrame(py_timecost_record)
	for scen in range(actual_SCEN_TYPE):
		timecost_df["avg_{}".format(scen)] = timecost_df.iloc[:, scen*SCEN_NUM:(scen+1)*SCEN_NUM].mean(axis=1)
	timecost_df.to_csv(timecost_path, index = None)

