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


ceiling = []
ceiling_hits = []
floor = []
floor_hits = []

def get_events_list(action, get_all=False):
	events = ''

	cpu_percent = \
		'module_load,module_free,module_get,module_put,module_request,power_cpu_idle,power_cpu_frequency,power_machine_suspend,rcu_utilization,sched_kthread_stop,sched_waking,sched_wakeup,sched_wakeup_new,sched_switch,sched_migrate_task,sched_process_free,sched_process_exit,sched_wait_task,sched_process_wait,sched_process_fork,sched_process_exec,sched_stat_wait,sched_stat_sleep,sched_stat_iowait,sched_stat_blocked,sched_stat_runtime,'
	memory_shared = \
		'kmem_kmalloc,kmem_cache_alloc,kmem_kmalloc_node,kmem_cache_alloc_node,kmem_kfree,kmem_cache_free,kmem_mm_page_free,kmem_mm_page_free_batched,kmem_mm_page_alloc,kmem_mm_page_alloc_zone_locked,kmem_mm_page_pcpu_drain,kmem_mm_page_alloc_extfrag,mm_vmscan_kswapd_sleep,mm_vmscan_kswapd_wake,mm_vmscan_wakeup_kswapd,mm_vmscan_direct_reclaim_begin,mm_vmscan_memcg_reclaim_begin,mm_vmscan_memcg_softlimit_reclaim_begin,mm_vmscan_direct_reclaim_end,mm_vmscan_memcg_reclaim_end,mm_vmscan_memcg_softlimit_reclaim_end,mm_vmscan_shrink_slab_start,mm_vmscan_shrink_slab_end,mm_vmscan_lru_isolate,mm_vmscan_writepage,mm_vmscan_lru_shrink_inactive,kvm_userspace_exit,kvm_set_irq,kvm_ioapic_set_irq,kvm_msi_set_irq,kvm_ack_irq,kvm_mmio,kvm_fpu,kvm_age_page,kvm_try_async_get_page,kvm_async_pf_doublefault,kvm_async_pf_not_present,kvm_async_pf_ready,kvm_async_pf_completed,'
	disk_usage_percent = \
		'writeback_dirty_page,writeback_write_inode_start,writeback_exec,writeback_wait,writeback_congestion_wait,writeback_thread_start,writeback_thread_stop,'

	if get_all:
		events = cpu_percent + memory_shared + disk_usage_percent
	else:
		for x in action:
			if x == 1:  # cpu_percent
				events = events + cpu_percent
			if x == 2:  # memory_shared
				events = events + memory_shared
			if x == 3:  # disk_usage_percent
				events = events + disk_usage_percent

	if events != '':
		events = events[:-1]

	return events

def get_syscalls_list(action, get_all=False):
	events = ''

	cpu_percent = \
		'module_load,module_free,module_get,module_put,module_request,power_cpu_idle,power_cpu_frequency,power_machine_suspend,rcu_utilization,sched_kthread_stop,sched_waking,sched_wakeup,sched_wakeup_new,sched_switch,sched_migrate_task,sched_process_free,sched_process_exit,sched_wait_task,sched_process_wait,sched_process_fork,sched_process_exec,sched_stat_wait,sched_stat_sleep,sched_stat_iowait,sched_stat_blocked,sched_stat_runtime,'
	memory_shared = \
		'kmem_kmalloc,kmem_cache_alloc,kmem_kmalloc_node,kmem_cache_alloc_node,kmem_kfree,kmem_cache_free,kmem_mm_page_free,kmem_mm_page_free_batched,kmem_mm_page_alloc,kmem_mm_page_alloc_zone_locked,kmem_mm_page_pcpu_drain,kmem_mm_page_alloc_extfrag,mm_vmscan_kswapd_sleep,mm_vmscan_kswapd_wake,mm_vmscan_wakeup_kswapd,mm_vmscan_direct_reclaim_begin,mm_vmscan_memcg_reclaim_begin,mm_vmscan_memcg_softlimit_reclaim_begin,mm_vmscan_direct_reclaim_end,mm_vmscan_memcg_reclaim_end,mm_vmscan_memcg_softlimit_reclaim_end,mm_vmscan_shrink_slab_start,mm_vmscan_shrink_slab_end,mm_vmscan_lru_isolate,mm_vmscan_writepage,mm_vmscan_lru_shrink_inactive,kvm_userspace_exit,kvm_set_irq,kvm_ioapic_set_irq,kvm_msi_set_irq,kvm_ack_irq,kvm_mmio,kvm_fpu,kvm_age_page,kvm_try_async_get_page,kvm_async_pf_doublefault,kvm_async_pf_not_present,kvm_async_pf_ready,kvm_async_pf_completed,'
	disk_usage_percent = \
		'writeback_dirty_page,writeback_write_inode_start,writeback_exec,writeback_wait,writeback_congestion_wait,writeback_thread_start,writeback_thread_stop,'

	if get_all:
		events = cpu_percent + memory_shared + disk_usage_percent
	else:
		for x in action:
			if x == 1:  # cpu_percent
				events = events + cpu_percent
			if x == 2:  # memory_shared
				events = events + memory_shared
			if x == 3:  # disk_usage_percent
				events = events + disk_usage_percent

	if events != '':
		events = events[:-1]

	return events

def bin_data(
	metric_data,
	window,
	bin_max=5,
	spv=0.4,
	):

	to_be_normalized_data = []
	size = len(metric_data) - 1

	for i in range(window):
		to_be_normalized_data.append(metric_data[i])

	mean = sum(to_be_normalized_data) / len(to_be_normalized_data)

	normalized_data = []

	for i in range(len(to_be_normalized_data)):
		normalized_data.append(to_be_normalized_data[i] / mean)

	bin = []

	for i in range(len(normalized_data)):
		if normalized_data[i] > 2:
			bin.append(bin_max)
		else:
			bin.append(math.trunc(normalized_data[i] / spv))

	print('Data:' + str(to_be_normalized_data))

  # print("Normalized:"+str(normalized_data))

	print('Bin:' + str(bin))
	print('----')
	return bin

def get_bin_max_and_culprits(dataset_tuple):
	bin = []
	for i in range(len(dataset_tuple[0])):
		maximum = -1
		culprit = []
		for j in range(len(dataset_tuple)):
			if dataset_tuple[j][i] >= maximum:
				maximum = dataset_tuple[j][i]
				culprit.append(j)
		bin.append([maximum, culprit])
	print(bin)
	return bin

def get_anomaly_counter(binMaxCulprits, triggerBinValue, no_of_metrics):
	anomalyCounter = [0] * no_of_metrics
	for i in range(len(binMaxCulprits)):
		if binMaxCulprits[i][0] >= triggerBinValue:
			culprit = binMaxCulprits[i][1]
			for j in range(len(binMaxCulprits[i][1])):
				anomalyCounter[binMaxCulprits[i][1][j]] = \
					anomalyCounter[binMaxCulprits[i][1][j]] + 1

	print(anomalyCounter)
	return anomalyCounter  # the array indecies are the metrics

def ceiling_floor_controller(currentPct, i, reclibrate_hits, reclibrate_pct):
	global ceiling
	global floor
	global ceiling_hits
	global floor_hits

	return_tuple=[0,0] #ceiling, floor
	if ceiling[i] <= currentPct:
		ceiling_hits[i] += 1
		return_tuple[0] = 1

	f = open("flagged.txt", "a")
	f.write("Metrics# "+str(i)+" ceiling:"+str(ceiling[i])+"\n")
	f.close()
		
	if floor[i] >= currentPct:
		floor_hits[i] += 1
		return_tuple[1] = 1

	f = open("flagged.txt", "a")
	f.write("Metrics# "+str(i)+" floor:"+str(floor[i])+"\n")
	f.close()
		
	if ceiling_hits[i] >= reclibrate_hits or floor_hits[i] >= reclibrate_hits :
		reset_ceiling_floor(currentPct, i, reclibrate_pct)

	return return_tuple

def reset_ceiling_floor(currentPct, i, reclibrate_pct):
	global ceiling
	global floor
	global ceiling_hits
	global floor_hits

	ceiling_hits[i] = 0
	floor_hits[i] = 0

	ceiling[i] = currentPct + (currentPct * reclibrate_pct)
	floor[i] = currentPct - (currentPct * reclibrate_pct)

	if ceiling[i] == 0.0:
		ceiling[i] = reclibrate_pct
		f = open("flagged.txt", "a")
		f.write("Metrics# "+str(i)+" reclibrate: DONE newceiling: "+str(ceiling[i])+"\n")
		f.close()
	
	if floor[i] == 0.0:
		floor[i] = -reclibrate_pct
		f = open("flagged.txt", "a")
		f.write("Metrics# "+str(i)+" reclibrate: DONE newfloor: "+str(floor[i])+"\n")
		f.close()

def decision_start(anomalyCounter,no_of_metrics,reclibrate_hits,reclibrate_pct,reset_calibration=False, anomaly_percent_threshold=0.6,anomaly_score_threshold=0.3,decreasing_factor=0.15):

	#anomaly_score = [0] * no_of_metrics
	#anomaly_percent_array_current = [0] * no_of_metrics


	anomaly_list_array = []

	for i in range(no_of_metrics):
		currentPct = anomalyCounter[i] / sample_size

		if reset_calibration:
			reset_ceiling_floor(currentPct, i, reclibrate_pct)

		dec_tup = ceiling_floor_controller(currentPct, i, reclibrate_hits, reclibrate_pct)

		if dec_tup[0] == 1:
			print('Flagged for ceiling break: ' + str(i))
			anomaly_list_array.append(i)
			f = open("flagged.txt", "a")
			f.write("***Metrics# "+str(i)+" flagged for ceiling break\n")
			f.close()

		if dec_tup[1] == 1:
			print('Flagged for floor break: ' + str(i))
			anomaly_list_array.append(i)
			f = open("flagged.txt", "a")
			f.write("***Metrics# "+str(i)+" flagged for floor break\n")
			f.close()

		f = open("flagged.txt", "a")
		f.write("...\n")
		f.close()


	'''
		if currentPct >= anomaly_percent_threshold:
			anomaly_score[i] = 1
			print('Flagged: ' + str(i))
			anomaly_list_array.append(i)
		else:
			anomaly_score[i] = (1 - decreasing_factor) \
				* anomaly_score[i]
			if anomaly_score[i] >= anomaly_score_threshold:
				print('Not flagged but enabled: ' + str(i))
				anomaly_list_array.append(i)
	'''
	print('---------------')

	events_list = get_events_list(anomaly_list_array)
	syscalls_list = get_syscalls_list(anomaly_list_array)

	both_list = [events_list, syscalls_list]

	return both_list

def prepare(dataset_tuple, sample_size=100):
	binned_dataset = []
	num_of_metrics = len(dataset_tuple)
	for i in range(num_of_metrics):
		binned_dataset.append(bin_data(dataset_tuple[i], sample_size))
	return binned_dataset


# end of func

os.system('lttng create test --output=/home/uvm1/Desktop/research/NewTestTraces/tracing')
os.system('lttng enable-event -k --syscall --all')
os.system('lttng add-context --kernel --type=tid')
os.system('lttng enable-event -k irq_softirq_entry,irq_softirq_raise,irq_softirq_exit,irq_handler_entry,irq_handler_exit,lttng_statedump_process_state,lttng_statedump_start,lttng_statedump_end,lttng_statedump_network_interface,lttng_statedump_block_device,block_rq_complete,block_rq_insert,block_rq_issue,block_bio_frontmerge,sched_migrate,sched_migrate_task,power_cpu_frequency,net_dev_queue,netif_receive_skb,net_if_receive_skb,timer_hrtimer_start,timer_hrtimer_cancel,timer_hrtimer_expire_entry,timer_hrtimer_expire_exit')
os.system('lttng start')



reset_calibration_step = 3

i = 0

last_ceiling_hits = []
last_floor_hits = []

while i < 100:
	i += 1


	f = open("flagged.txt", "a")
	f.write("-------------New rotate("+str(i)+")------------\n")
	f.close()

	reset_calibration = False
	if i % reset_calibration_step == 0:
		reset_calibration = True


	df = read_csv('dump.csv').iloc[:500 * i]

	cpu_percent = df['cpu_percent']

	# cpu_user_time = df['cpu_user_time']
	# cpu_system_time = df['cpu_system_time']
	# cpu_idle_time = df['cpu_idle_time']
	# cpu_iowait = df['cpu_iowait']
	# cpu_irq = df['cpu_irq']
	# cpu_softirq = df['cpu_softirq']
	# cpu_numbers_of_ctx_switches = df['cpu_numbers_of_ctx_switches']
	# cpu_numbers_of_interrupts = df['cpu_numbers_of_interrupts']
	# cpu_numbers_of_soft_interrupts = df['cpu_numbers_of_soft_interrupts']
	# cpu_load_runable_state = df['cpu_load_runable_state']

	memory_percent = df['memory_percent']

	# memory_active = df['memory_active']
	# memory_cached = df['memory_cached']
	# memory_shared = df['memory_shared']
	# memory_swap_percent = df['memory_swap_percent']
	# memory_swap_sin = df['memory_swap_sin']
	# memory_swap_sout = df['memory_swap_sout']

	disk_usage_percent = df['disk_usage_percent']

	# disk_read_count = df['disk_read_count']
	# disk_write_count = df['disk_write_count']
	# disk_read_time = df['disk_read_time']
	# isk_write_time = df['disk_write_time']

	dataset_cpu1 = np.array(cpu_percent[::-1])
	dataset_ram1 = np.array(memory_percent[::-1])
	dataset_hdd1 = np.array(disk_usage_percent[::-1])

	# dataset_ram1 = np.array(memory_active[::-1])

	sample_size = 20

	dataset_tuple = (dataset_cpu1, dataset_ram1, dataset_hdd1)
	no_of_metrics = len(dataset_tuple)

	if i == 1:
		ceiling = [0] * no_of_metrics
		floor = [0] * no_of_metrics
		ceiling_hits = [0] * no_of_metrics
		floor_hits = [0] * no_of_metrics
		last_ceiling_hits = [0] * no_of_metrics
		last_floor_hits = [0] * no_of_metrics

	f = open("cfdata.txt", "a")
	f.write(str(ceiling)[1:-1] + ",|," + str(floor)[1:-1] + ",|,")
	f.close()

	binned_dataset = prepare(dataset_tuple, sample_size)

	binMaxCulprits = get_bin_max_and_culprits(binned_dataset)

	print('--------------')

	anomalyCounter = get_anomaly_counter(binMaxCulprits, 3, no_of_metrics)

	anomaly_percent_threshold = 0.7
	anomaly_score_threshold = 0.3
	decreasing_factor = 0.7

	all_events_list = get_events_list(None, True)

	f = open("cfdata.txt", "a")

	ceiling_hits_string = ""
	floor_hits_string = ""

	counter = 0
	while counter < no_of_metrics:
		if ceiling_hits[counter] == last_ceiling_hits[counter]:
			ceiling_hits_string = ceiling_hits_string + "0,"
		else:
			ceiling_hits_string = ceiling_hits_string + "1,"
			last_ceiling_hits[counter] = ceiling_hits[counter]

		if floor_hits[counter] == last_floor_hits[counter]:
			floor_hits_string = floor_hits_string + "0,"
		else:
			floor_hits_string = floor_hits_string + "1,"
			last_floor_hits[counter] = floor_hits[counter]
		counter = counter + 1

	ceiling_hits_string = ceiling_hits_string[1:]
	floor_hits_string = floor_hits_string[1:]

	f.write(ceiling_hits_string + ",|," + floor_hits_string)
	f.write("\n")
	f.close()

	#all_syscalls_list = get_syscalls_list(None, True)
	#os.system('lttng disable-event -k --syscall ' + all_syscalls_list)

	os.system('lttng disable-event -k ' + all_events_list)

	
	reclibrate_hits = 5
	reclibrate_pct = 0.03
	both_list = decision_start(anomalyCounter,
								 no_of_metrics,
								 reclibrate_hits,
								 reclibrate_pct,
								 reset_calibration,
								 anomaly_percent_threshold,
								 anomaly_score_threshold,
								 decreasing_factor
								 )
	events_list = both_list[0]
	syscalls_list = both_list[1]
	#os.system('lttng enable-event -k --syscall ' + syscalls_list)
	os.system('lttng enable-event -k ' + events_list)
	os.system('lttng rotate')
	time.sleep(4)
