

import argparse
import numpy as np


def argument_parser():
	parser = argparse.ArgumentParser(description='generates a mel spectrogram from an input feature')
	parser.add_argument('classifier', action='store', help='pre-trained classifier (.npz format)')
	parser.add_argument('inverter', action='store', help='pre-trained feature inverter (.npz format)')
	parser.add_argument('dataset_path', action='store', help='dataset path')
	parser.add_argument('results_dir', action='store', help='path to save inversion plots')
	parser.add_argument('--dataset', default='jamendo', help='dataset name')
	parser.add_argument('--cache-spectra', metavar='DIR', default=None, help='store spectra in the given directory (disabled by default).')
	parser.add_argument('--augment', action='store_true', default=True, help='If given, perform train-time data augmentation.')
	parser.add_argument('--no-augment', action='store_false', dest='augment', help='If given, disable train-time data augmentation.')
	parser.add_argument('--featloss', default=False, action='store_true', help='If given, calculate feature space loss.')
	parser.add_argument('--quant_analysis', default=False, action='store_true', help='If given, performs quantitative analysis by sampling random instances. If not given, the code executes single instance inversion')
	return parser

def normalise(x):
	'''
	Normalise a vector/ matrix, in range 0 - 1
    '''
	return((x-x.min())/(x.max()-x.min()))


def area_calculation(norm_inv, debug_flag = False):
	"""
	For the quantitative anaysis case, it computes
	the area enabled by bins that are set to 1
	"""
	
	n_bin_enabled = (norm_inv == 1).sum()
	n_bins = norm_inv.shape[0] * norm_inv.shape[1]
	area_enabled = n_bin_enabled/float(n_bins)
	if debug_flag:
		print("Percentage of enabled bins: %.2f" %(100 * area_enabled))
	return area_enabled


def quant_result_analysis(pred_before, pred_after, area_per_instance, mt, class_threshold, debug_flag=False):
	"""
	analyses the saved results per threshold index and returns are results tuple
	"""
	
	# quantify the classification performance after masking
	ground = (np.asarray(pred_before))>class_threshold
	pred = (np.asarray(pred_after))>class_threshold
	class_change = np.zeros(len(ground), dtype = 'bool')
	
	count_pass = 0
	count_fail = 0
	
	# explanation loss
	for i in range(len(ground)):
		if ground[i]==pred[i]:
			count_pass +=1
		else:
			count_fail +=1
			class_change [i]= True
	
	if debug_flag:
		print("Total instances:%d" %(count_pass+ count_fail))
		print("Number of fails:%d" %(count_fail))
		print("Percentage explanation loss: %.2f" %(100*count_fail/float(count_fail+count_pass)))
		print("Average area: %f" %(sum(area_per_instance)/len(area_per_instance)))
	
	# save the final results in each iteration (govern by threshold) as a tuple
	return (mt, count_pass+count_fail, count_fail, round(100*count_fail/float(count_fail+count_pass), 2), round(sum(area_per_instance)/len(area_per_instance), 2))
