import copy
import numpy
import csv
import random
import time
import operator
from operator import itemgetter
import generate_training_data


# predicts whether a feature set classifier pair will be optimal or sub optimal using logistic and linear regression models
def predict(data):	
	
	def normalize(data):
		# converts structural statistics to z-scores
		types = ['f1', 'f2', 'f3', 'f4', 'f5','f6']
		for idx in types:
			vals = []
			for record in data:
				vals.append(float(record[idx]))
				
			mean = numpy.mean(vals)
			sd  = numpy.std(vals, ddof = 1)
			if sd == 0:
				sd = 1
			
			for record in data:
				new_val = (float(record[idx]) - mean)  / (sd*sd)
				record[idx] = new_val
		
		
	def lin_predict(record):
		j = numpy.matrix([-1.039011e-12, -9.114375e-01, -1.223389e-01, -2.006449e-01])
		feats = ['f3', 'f4', 'f5']
		t = [1]
		for idx in feats:
			val = float(record[idx])
			t.append(val)
		t = numpy.matrix(t)
		t = numpy.transpose(t)
	
		lin_pred = j * t
		if lin_pred > 1:
			lin_pred = 1
		record['lin_pred'] = lin_pred
		return record
	
	def log_predict(record):
		b = numpy.matrix([-0.64063267,   0.15706603,   0.13272974,  -0.03350878,  -0.15182902,  -0.19548473,  -0.68787718 ])
		feats = ['f1','f2','f3','f4','f5','f6']
		Q = [1]
		for idx in feats:
			val = float(record[idx])
			Q.append(val)

		Q = numpy.matrix(Q)
		q = numpy.transpose(Q)

		a = b * q
		value = (1/(numpy.exp(-a)+1))
		if value >= .5:
			init_pred = 1
		else:
			init_pred = 0
		
		# to get a prediction of 'optimal', logistic and linear models must agree
		if record['lin_pred'] > 0 and init_pred == 1:
			pred = 1
		else:
			pred = 0
		record['pred'] = pred
		
		return record
	
	normalize(data)
	for record in data:
		lin_predict(record)
	for record in data:
		log_predict(record)
	
	return data
	
# calculates performance statistics for prediction algorithm
def results(data):
	
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	
	for record in data:
		label = int(record['set_label'])
		pred = record['pred']
		
		if label == 1 and pred == 1:
			tp +=1
		elif label == 1 and pred == 0:
			fn +=1
		elif label == 0 and pred == 0:
			tn +=1
		else:
			fp +=1
	
	accuracy = float(tp + tn) / len(data)
	if tp + fp == 0:
		precision = 0
	else:
		precision  = float(tp) / (tp + fp)
		recall = float(tp) / (tp + fn)
		
	return accuracy, precision, recall

# calculates dimension for a particular test
def get_dimension(test):
	total = 0
	for interval in test:
		bounds = interval.split(':')
		a = int(bounds[0])
		b = int(bounds[1])
		length = b - a
		total = total + length
	return total

def main(filename):
	# read in training vectors
	# this data set has max dimension
	data = generate_training_data.read_data(filename)
	random.shuffle(data)
	
	# read in test dimensions from file containing lists of strings indicating feature type boundaries
	tests = []
	for line in open("amazon2_tests.txt"):
		tests.append(eval(line.strip()))
	tests = tests[1:]
	
	# set up output file
	file1 = open('dimension_testing2_results.csv','w')
	w = csv.writer(file1, dialect = 'excel')
	w.writerow(["Column Dimension", "Accuracy", "Precision", "Recall", "CPU Time"])
	
	# run an individual dimension test
	for test in tests:
		dimension = get_dimension(test)
		print "testing with ", dimension, " dimensions..."
		data = copy.deepcopy(data)
		
		# generate neccessary parameters for test
		subsets = generate_training_data.make_subsets(test)
		label_lists = generate_training_data.make_label_lists(["Grove", "Riley", "Comdet", "Vernon","Janson"])
		label_lists = label_lists[1:]
		
		# tailor original data set to only include dimensions of test
		for record in data:
			generate_training_data.make_feature_set(record,set(test))
		
		# generate structures, ranks, labels, and predictions for each feature set classifier pair in test	
		times = [] # track CPU time to generate structures
		all_fscps = []
		
		# loop over unique binary classifiers 
		for label_list in label_lists:
			print label_list
			structs = []
			struct_start = time.clock()
			
			# loop over each possible combination of feature types
			for j, x in enumerate(subsets):
				if j == 0:# cannot generate structures for the empty set, so manually set to 0
					struct = {'subset': x, 'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0}
				else: # generate geometric structual information for each subset
					struct = generate_training_data.analyze_structure(data, x, label_list)
				structs.append(struct)
				
			struct_time = time.clock() - struct_start	
			times.append(struct_time)
			
			# this information is used to evaluate the algorithm
			for row in structs:
				generate_training_data.feature_set_ranker(data, row, label_list)
			labeled_structs = generate_training_data.label_maker(structs)
			
			for record in labeled_structs:
				all_fscps.append(record)
				
		end = sum(times)
 		print "finished test at: ", end
		#make predictions
		predict(all_fscps)
		# check predictions
		accuracy, precision, recall = results(all_fscps)
		# output results
		output = [dimension, accuracy, precision, recall, end]
		w.writerow(output)
		file1.flush()
	
main("/Users/carlystambaugh/Desktop/vector_text_files/dimension_testing2_vectors.txt")
			