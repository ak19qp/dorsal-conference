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


import warnings
warnings.filterwarnings("ignore")


last_corr_reading = []
last_corr_std = []

corr_hit_counter = 0

anomalyScore = []



from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randrange
import statistics
import statsmodels.api as sm

def arima(metric_dataset, metric_id, use_sd_or_mean = "sd"):

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




reset_calibration_step = 24

i = 0

corr_saved_counter = 0

df = read_csv('20c_managebooks_5pct_rand_wait_20c_data.csv')

#Normalizing Start
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
df.to_csv('normalized.csv')
#Normalizing End

sample_size = 24
start_at = 0
loop = 300

while i < loop:
	
	i += 1
	pct_progress = (i/loop) * 100
	pct_progress = round(pct_progress, 2)
	print("Progress: "+str(pct_progress)+"%\n")

	arima_act_taken_for_save = 0
	corr_act_taken_for_save = 0

	isActualAnomaly = df.iloc[start_at+i+sample_size-1, 0]
	
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
			f.write("actual_is_anomally,arima_flag,correlation_flag,both_flag,flagged_metrics\n")
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
	if isActualAnomaly:
		f.write("1,"+str(arima_act_taken_for_save)+","+str(corr_act_taken_for_save)+",")
	else:
		f.write("0,"+str(arima_act_taken_for_save)+","+str(corr_act_taken_for_save)+",")

	if arima_act_taken_for_save == 1 and corr_act_taken_for_save == 1:
		f.write("1,")
	else:
		f.write("0,")
	

	flagged_metrics = ""
	z = 0
	while z < no_of_metrics:
		if corr_action_arr[z] == 1:
			if arima_action_taken[z] == 1:
				fs = open("results_of_metric-"+str(z)+".csv", "a")
				fs.write("1,1\n")
				fs.close()
				flagged_metrics = flagged_metrics + str(z) + "|"
			else:
				fs = open("results_of_metric-"+str(z)+".csv", "a")
				fs.write("1,0\n")
				fs.close()
		else:
			fs = open("results_of_metric-"+str(z)+".csv", "a")
			fs.write("0,0\n")
			fs.close()
		z = z + 1
	f.write(flagged_metrics+"\n")
	f.close()

print("Progress: Complete!\n")
