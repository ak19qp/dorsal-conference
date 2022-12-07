#!/usr/bin/python
# -*- coding: utf-8 -*-
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import math
from random import randrange
import statistics
import statsmodels.api as sm
import time
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randrange
import statistics
import statsmodels.api as sm
import warnings


warnings.filterwarnings("ignore")

last_corr_reading = []
last_corr_std = []

corr_hit_counter = 0

anomalyScore = []

metrics_list = ["cpu_percent","cpu_user_time","cpu_system_time","cpu_idle_time","cpu_iowait","cpu_irq","cpu_softirq","cpu_numbers_of_ctx_switches","cpu_numbers_of_interrupts","cpu_numbers_of_soft_interrupts","cpu_load_runable_state","memory_percent","memory_active","memory_buffers","memory_cached","memory_shared","memory_swap_percent","memory_swap_sin","memory_swap_sout","disk_usage_percent","disk_read_count","disk_write_count","disk_read_time","disk_write_time","disk_busy_time","network_bytes_sent","network_bytes_recv","network_packets_sent","network_packets_recv","network_errin","network_errout","network_dropin","network_dropout"]





def get_events_list(action=None, get_all=False):

	events = ''

	cpu_events = \
		'module_load,module_free,module_get,module_put,module_request,power_cpu_idle,power_cpu_frequency,power_machine_suspend,rcu_utilization,sched_kthread_stop,sched_waking,sched_wakeup,sched_wakeup_new,sched_switch,sched_migrate_task,sched_process_free,sched_process_exit,sched_wait_task,sched_process_wait,sched_process_fork,sched_process_exec,sched_stat_wait,sched_stat_sleep,sched_stat_iowait,sched_stat_blocked,sched_stat_runtime,'
	memory_events = \
		'kmem_kmalloc,kmem_cache_alloc,kmem_kmalloc_node,kmem_cache_alloc_node,kmem_kfree,kmem_cache_free,kmem_mm_page_free,kmem_mm_page_free_batched,kmem_mm_page_alloc,kmem_mm_page_alloc_zone_locked,kmem_mm_page_pcpu_drain,kmem_mm_page_alloc_extfrag,mm_vmscan_kswapd_sleep,mm_vmscan_kswapd_wake,mm_vmscan_wakeup_kswapd,mm_vmscan_direct_reclaim_begin,mm_vmscan_memcg_reclaim_begin,mm_vmscan_memcg_softlimit_reclaim_begin,mm_vmscan_direct_reclaim_end,mm_vmscan_memcg_reclaim_end,mm_vmscan_memcg_softlimit_reclaim_end,mm_vmscan_shrink_slab_start,mm_vmscan_shrink_slab_end,mm_vmscan_lru_isolate,mm_vmscan_writepage,mm_vmscan_lru_shrink_inactive,kvm_userspace_exit,kvm_set_irq,kvm_ioapic_set_irq,kvm_msi_set_irq,kvm_ack_irq,kvm_mmio,kvm_fpu,kvm_age_page,kvm_try_async_get_page,kvm_async_pf_doublefault,kvm_async_pf_not_present,kvm_async_pf_ready,kvm_async_pf_completed,'
	disk_events = \
		'writeback_dirty_page,writeback_write_inode_start,writeback_exec,writeback_wait,writeback_congestion_wait,writeback_thread_start,writeback_thread_stop,'
	network_events = \
		'net_dev_xmit,net_dev_queue,net_if_receive_skb,net_if_rx,net_napi_gro_frags_entry,net_napi_gro_receive_entry,net_if_receive_skb_entry,net_if_rx_entry,net_if_rx_ni_entry,net_if_receive_skb_list_entry,net_napi_gro_frags_exit,net_napi_gro_receive_exit,net_if_receive_skb_exit,net_if_rx_exit,net_if_rx_ni_exit,net_if_receive_skb_list_exit'

	if get_all:
		events = cpu_events + memory_events + disk_events + network_events
	else:
		for x in action:
			if "cpu" in x:
				events = events + cpu_events
			if "memory" in x:
				events = events + memory_events
			if "disk" in x:
				events = events + disk_events
			if "network" in x:
				events = events + network_events

	if events != '':
		events = events[:-1]

	return events

def arima(metric_dataset, metric_id, use_sd_or_mean = "sd"):
	global anomalyScore
	errorUpperBound = 0.03
	errorLowerBound = 0.03
	betaVal = 0.5
	samplingFrequency = 1.0
	anomalyScoreThreshold = 0.3
	model = sm.tsa.ARIMA(metric_dataset[0:-1], order=(1,1,1))
	model_fit = model.fit()
	output = model_fit.forecast()
	output.reset_index(drop=True, inplace=True)
	arima_prediction = output[0]
	obs = metric_dataset[len(metric_dataset)-1]

	meanOfSample = statistics.mean(metric_dataset[0:-2])
	sdOfSample = statistics.stdev(metric_dataset[0:-2])
	meanAndPredictionDifferencePercentage = abs(arima_prediction - meanOfSample) / meanOfSample

	betaVal = abs(abs(arima_prediction)-abs(meanOfSample)) / abs(meanOfSample)

	isAnomaly = 0
	if use_sd_or_mean == "mean":
		if arima_prediction > (meanOfSample + (errorUpperBound * meanOfSample)) or arima_prediction < (meanOfSample - (errorLowerBound * meanOfSample)):
			isAnomaly = 1

	if use_sd_or_mean == "sd":
		if arima_prediction > (meanOfSample + sdOfSample) or arima_prediction < (meanOfSample - sdOfSample):
			isAnomaly = 1
	anomalyScore[metric_id] = (betaVal * isAnomaly) + (abs(1 - betaVal) * anomalyScore[metric_id])

	ARIMAtakeAction = 0
	if anomalyScore[metric_id] >= anomalyScoreThreshold:
		ARIMAtakeAction = 1


	#samplingFrequency =  1 - anomalyScore

	saveStr = str(arima_prediction) + "," + str(obs) + "," + str(meanOfSample) + "," + str(sdOfSample) + "," + str(meanAndPredictionDifferencePercentage) + "," + str(betaVal) + "," + str(isAnomaly) + "," + str(anomalyScore[metric_id]) + "," + str(samplingFrequency) + "," + str(ARIMAtakeAction) + ","
	#print(saveStr)
	f = open("results_of_metric-"+str(metric_id)+".csv", "a")
	f.write(saveStr)
	f.close()

	return ARIMAtakeAction


def save_correlation(corr_mat):
	write_str = ""

	for item in corr_mat.unstack():
	    write_str = write_str + "," + str(item)

	write_str = write_str[1:] + "\n"

	f = open("corr_data.csv", "a")
	f.write(write_str)
	f.close()


def is_corr_broken(corr_mat): # pass last values of the dataframe here

	global last_corr_std
	global last_corr_reading
	global corr_hit_counter

	#print("last_std:"+str(last_corr_std)+" last reading:"+str(last_corr_reading)+" corr_hit_counter:"+str(corr_hit_counter))

	flagged_index = []
	corr_broken = False
	count = -1
	for item in corr_mat.unstack():
		count = count + 1
		if abs(float(item)) > abs(last_corr_reading[count]) + abs(last_corr_std[count]) or abs(float(item)) < abs(last_corr_reading[count]) - abs(last_corr_std[count]):
			flagged_index.append(count)
			corr_broken = True

	if corr_broken:
		corr_hit_counter = corr_hit_counter + 1

	return flagged_index


def reset_correlation(corr_mat_dataset):
	global last_corr_std
	global last_corr_reading
	global corr_hit_counter
	corr_mat_dataset.reset_index(drop=True, inplace=True)

	corr_hit_counter = 0

	last_corr_std = corr_mat_dataset.std(skipna = True).to_numpy()
	last_corr_reading = corr_mat_dataset.iloc[-1:,:].to_numpy()[0]


def start(trace_file_output, csv_file, sample_size, start_at, loop, reset_calibration_step):
	global anomalyScore
	global corr_hit_counter

	os.system('lttng create test --output='+trace_file_output)
	#os.system('lttng enable-event -k --syscall --all')
	os.system('lttng enable-event -k irq_softirq_entry,irq_softirq_raise,irq_softirq_exit,irq_handler_entry,irq_handler_exit,lttng_statedump_process_state,lttng_statedump_start,lttng_statedump_end,lttng_statedump_network_interface,lttng_statedump_block_device,block_rq_complete,block_rq_insert,block_rq_issue,block_bio_frontmerge,sched_migrate,sched_migrate_task,power_cpu_frequency,net_dev_queue,netif_receive_skb,net_if_receive_skb,timer_hrtimer_start,timer_hrtimer_cancel,timer_hrtimer_expire_entry,timer_hrtimer_expire_exit')
	os.system('lttng add-context --kernel --type=tid')
	os.system('lttng start')

	i = 0
	corr_saved_counter = 0

	while i < loop:
		i += 1

		df = read_csv(csv_file)

		metrics_list = df.columns.values.tolist()

		pct_progress = (i/loop) * 100
		pct_progress = round(pct_progress, 2)
		print("Progress: "+str(pct_progress)+"%\n")

		arima_act_taken_for_save = 0
		corr_act_taken_for_save = 0

		arima_action_taken = []

		sample_dataset = df.iloc[start_at+i : start_at+i+sample_size,] # 1 is used because 0 has the anomaly labeling

		sample_dataset.reset_index(drop=True, inplace=True)

		no_of_metrics = len(sample_dataset.columns)
		arima_action_taken = [0] * no_of_metrics

		if i == 1:

			anomalyScore = [0] * no_of_metrics

			z = 0

			while z < no_of_metrics:
				f = open("results_of_metric-"+str(z)+".csv", "w")
				write_str = "predicted,actual,mean_of_sample,sd_of_sample,meanAndPredictionDifferencePercentage,betaVal,ARIMAflagged,anomalyScore,sampling_Freq,ARIMAtakeAction,CorrelationBroken,ActionTaken(Both)\n"
				f.write(write_str)
				f.close()

				f = open("corr_data.csv", "w")
				f.write("")
				f.close()

				f = open("result_summary.csv", "w")
				f.write("arima_flag,correlation_flag,both_flag,flagged_metrics_id,flagged_metrics\n")
				f.close()

				z = z + 1

		z = 0
		while z < no_of_metrics:
			arima_action_taken[z] = arima(sample_dataset.iloc[:,z], z)
			if arima_action_taken[z] == 1:
				arima_act_taken_for_save = 1
			z = z + 1

		#saving correlations
		correlation_mat = sample_dataset.corr()
		save_correlation(correlation_mat)
		corr_saved_counter = corr_saved_counter + 1

		#setting up the global variables for storing last corr and the last std.
		if corr_saved_counter == sample_size or corr_hit_counter >= reset_calibration_step:
			corr_mat_dataset = read_csv('corr_data.csv').iloc[-sample_size:,:]
			reset_correlation(corr_mat_dataset)

		broken_corr_mat=[]
		if i>= sample_size*2:
			last_corr_mat = read_csv('corr_data.csv').iloc[-1:,:]
			broken_corr_mat = is_corr_broken(last_corr_mat)

		corr_action_arr = [0] * no_of_metrics

		if len(broken_corr_mat)>0:
			corr_act_taken_for_save = 1
			for item in broken_corr_mat:
				flagidx = int(int(item) / int(no_of_metrics))
				corr_action_arr[flagidx] = 1


		f = open("result_summary.csv", "a")

		if arima_act_taken_for_save == 1:
			f.write("1,")
		else:
			f.write("0,")

		if corr_act_taken_for_save == 1:
			f.write("1,")
		else:
			f.write("0,")

		if arima_act_taken_for_save == 1 and corr_act_taken_for_save == 1:
			f.write("1,")
		else:
			f.write("0,")

		flagged_metrics = ""
		flagged_metrics_id = ""
		z = 0
		while z < no_of_metrics:
			if corr_action_arr[z] == 1:
				if arima_action_taken[z] == 1:
					fs = open("results_of_metric-"+str(z)+".csv", "a")
					fs.write("1,1\n")
					fs.close()
					flagged_metrics = flagged_metrics + metrics_list[z] + " | "
					flagged_metrics_id = flagged_metrics_id + str(z) + " | "
				else:
					fs = open("results_of_metric-"+str(z)+".csv", "a")
					fs.write("1,0\n")
					fs.close()
			else:
				fs = open("results_of_metric-"+str(z)+".csv", "a")
				fs.write("0,0\n")
				fs.close()
			z = z + 1
		f.write(flagged_metrics_id+",")
		f.write(flagged_metrics+"\n")
		f.close()


		print("Disabling all kernel syscall events.")
		#all_events_list = get_events_list(None, True)
		#os.system('lttng disable-event -k ' + all_events_list)
		os.system('lttng disable-event -k --syscall --all-events')
		events_list = get_events_list(flagged_metrics.split(" | "), False)

		if arima_act_taken_for_save == 1 and corr_act_taken_for_save == 1:
			print("Enabling flagged events.")
			os.system('lttng enable-event -k ' + events_list)
		else:
			print("No flagged metrics, no events to enable.")


		print("Rotate")
		os.system('lttng rotate')

		time.sleep(4)


start("/home/a/Desktop/experiments/tracing","dump.csv", 24, 50, 1000, 25)
