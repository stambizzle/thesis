# Given a training set, this program predicts which subsets of features will give optimal performance
# It does so by generating structural information about each subset of features
# and applying logistic and linear regression models

import copy
import csv
import numpy
import operator
import random
import sys
import math
import os
import numpy.linalg
from optparse import OptionParser

# reads in data from the training set provided by user
def read_data(filename):
	dicts = []
	for line in open(filename):
		things = line.strip().split() 
		label = things[0]
		vector = things[1:]
		
		info = {}
		info['class_label'] = label
		info['feature_vector'] = parse_features(vector)
		
		dicts.append(info)

	return dicts

# converts sparse vector format to list of present features
def parse_features(curr_vector):
	return [int(atts.split(":")[0]) for atts in curr_vector]

# returns a vector only containing features within the current subset being analyzed
def make_feature_set(curr_vector, features_to_use):
	new_vector = []
	for chunk in features_to_use:
		bounds = chunk.split(':')
		a = int(bounds[0])
		b = int(bounds[1])
		for att in curr_vector:
			if a <= att <= b:
				new_vector.append(att)
	return new_vector
	
# makes all possible combinations of feature types
def make_subsets(features):
	if len(features) == 0:
		r = [set()]
		return r

	r = []
	x = features.pop()
	for t in make_subsets(features):
		r.append(t)
		r.append(t.union(set([x])))
	return r

# generates a list of all unique binary classifiers
def make_label_lists(labels):
	unique_set = []
	rv =[]
	labs = set(labels)
	full_set = make_subsets(labels)
	for subset in full_set:
		comp = labs - subset
		if comp in unique_set or subset in unique_set:
			continue
		else:
			if len(subset) < len(comp):
				unique_set.append(subset)
			else:
				unique_set.append(comp)
	for x in unique_set:
		rv.append(list(x))
	return rv

# generates structural information for each feature set
def analyze_structure(data, curr_subset, label_list):
	data = copy.deepcopy(data)
	# create feature vector for curr_subset
	#create dictionary for current test
	info = {}
	info['subset'] = curr_subset
	for record in data:
		record['feature_vector'] = make_feature_set(record['feature_vector'], curr_subset)

	reindexed, values = reindex(data)
	pos, neg = splitter(reindexed, label_list)
	posmatrix = build_matrix(pos, values)
	negmatrix = build_matrix(neg, values)
	matrix = build_matrix(reindexed, values)

	# identify number of unique values in each class
	uniqpos = column_totals(posmatrix)
	uniqneg = column_totals(negmatrix)

	# calculate polytope dimension
	dimpos = get_poly_dim(posmatrix)
	dimneg = get_poly_dim(negmatrix)
	dimfull = get_poly_dim(matrix)

	# create polytope features	
	info['f1'] = float(dimpos)/uniqpos
	info ['f2'] = float(dimneg)/uniqneg
	info['f3'] = float(dimpos)/len(values)
	info['f4'] = float(dimneg)/len(values)
	info['f5'] = float(dimfull)/len(values)

	# calculate affine overlap ratio
	PsetminusQ = affine_hull_intersection(posmatrix, neg)
	QsetminusP = affine_hull_intersection(negmatrix, pos)
	overlap = len(reindexed) - PsetminusQ - QsetminusP
	info['f6'] = float(overlap)/len(reindexed)

	return info

# determines ambient dimension
def column_totals(matrix):
	count = 0
	a,b = numpy.shape(matrix)
	for i in xrange(0,b):
		z = matrix[:,i].sum()
		if z > 0:
			count +=1
	return count

# calculate the affine dimension of each polytope
def get_poly_dim(mat):
	d = mat[-1]
	k = mat[:-1]
	g = k-d
	dim = my_matrix_rank(g)
	return dim


# reindex features to avoid large spaces of zeros
def reindex(data):
	# order present values from least to greatest
	vals = []
	for record in data:
		curr_vec = record['feature_vector']
		for item in curr_vec:
			vals.append(item)
	unique = set(vals)
	values = list(unique)
	values.sort()
	
	# create a dictionary for efficient mapping
	new_index = {}
	for i, v in enumerate(values):
		new_index[v] = i

	# rename each unique feature value as its index
	new_data = []
	for record in data:
		new_record = copy.deepcopy(record)
		new_vector = []
		for item in record['feature_vector']:
			new_vector.append(new_index[item])
		new_record['feature_vector'] = new_vector
		new_data.append(new_record)
	return new_data, values

# split into positive and negative examples
def splitter(data, split_vals):
	list1 = []
	list2 = []
	for record in data:
		if record['class_label'] in split_vals:
			list1.append(record)
		else:
			list2.append(record)
	return list1, list2

# takes list of feature indices and returns a numpy matrix
def build_matrix(examples, values):
	vectors = []
	num_features = len(values)
	for record in examples:
		vector = [0 for i in xrange(num_features)]
		curr_vec = record['feature_vector']
		for idx in curr_vec:
			vector[int(idx)] = 1
		vectors.append(vector)
	z = numpy.matrix(vectors)
	return z


# determines the overlap of the class represented polytopes
def affine_hull_intersection(poly1, examples2):
	setminus = 0			
	for record in examples2:
		if dimension_increase_check(record, poly1):
			setminus +=1

	return setminus

# adds a point from poly2 to poly1
# an increase in dimension indicates point does not lie in affine hull of poly1
def dimension_increase_check(record, poly1):	
	a, b = poly1.shape
	vector = numpy.zeros(b)
	vector[record['feature_vector']] = 1
	new_matrix = numpy.vstack((poly1, vector))
	if get_poly_dim(new_matrix) > get_poly_dim(poly1):
		return True
	else:
		return False
	
# this function makes up for buggy SVD function from the linalg library. It offers a conservative estimate 
# for rank (eps=1e-6), and returns the largest possible value for rank if SVD does not converge.
def my_matrix_rank(A,  eps=1e-6):
	try:
		u, s, vh = numpy.linalg.svd(A)
		return len([x for x in s if abs(x) > eps])
	
	except numpy.linalg.LinAlgError:
		print "SVD did not converge...making a conservative estimate for rank"
		a,b = A.shape
		return min(a, column_totals(A))

# generates the output file names inside the user provided directory
def generate_filenames(dir_name, label_lists):
	names = []
	if not os.path.exists(dir_name):
	    os.makedirs(dir_name)
	for i in xrange(1, len(label_lists)+1):
		curr_name = '%s/classifier_%d.csv' % (dir_name, i)
		names.append(curr_name)
	return names
	
# uses the logistic and linear regression models to predict whether a subset is optimal
def predict(data):
	
	# convert f1-f6 into z-scores	
	def standardize(data, idx):
			vals = []
			for record in data:
				vals.append(float(record[idx]))
			mean = numpy.mean(vals)
			sd = numpy.std(vals, ddof = 1)
			if sd == 0: # avoid division by zero if all values are the same, effectively removing the feature from consideration by the model
				sd = 1
			for record in data:
				new_val = (float(record[idx]) - mean)  / (sd * sd)
				record[idx] = new_val
	
	# linear regression model
	def lin_predict(record):
		j = numpy.matrix([-1.039011e-12, -9.114375e-01, -1.223389e-01, -2.006449e-01]) # coefficients of optimal linear model
		feats = ['f3', 'f4', 'f5']
		t = [1]
		for idx in feats:
			val = float(record[idx])
			t.append(val)
		t = numpy.matrix(t)
		t = numpy.transpose(t)

		lin_pred = j * t
		record['lin_pred'] = lin_pred
		return record
		
	# logistic regression model
	def log_predict(record):
		b = numpy.matrix([-0.64063267,   0.15706603,   0.13272974,  -0.03350878,  -0.15182902,  -0.19548473,  -0.68787718 ]) # coefficients of optimal logistic regression model
		feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
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
		if record['lin_pred'] > 0 and init_pred == 1: # requires both models to agree before assigning a prediction of optimal
			pred = 1
		else:
			pred = -1
		record['pred'] = pred
		return record

	for idx in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']:
		standardize(data, idx)
	for record in data:
		lin_predict(record)
	for record in data:
		log_predict(record)
	good_sets = [] # a list of the predicted optimal sets
	for record in data:
		if record['pred'] == 1:
			good_sets.append(record)
	return good_sets

# argument is a text file with labeled training examples in sparse vector format
# file needs to be formatted as follows:  label' 'feature#:1' 'feature#:1' '...
def main(filename, feature_groups, class_labels, dir_name):
	data = read_data(filename)
	random.shuffle(data)
	features = feature_groups.split(',')
	labels = class_labels.split(',')
	subsets = make_subsets(features)
	subsets = subsets[1:] # do not generate structures for the empty set
	label_lists = make_label_lists(labels)
	label_lists = label_lists[1:] # first label list is empty
	text_names = generate_filenames(dir_name, label_lists)
	for j, label_list in enumerate(label_lists):
		print "working on ", j+1, " of ", len(label_lists), " binary classifiers"
		structures = []
		for i, x in enumerate(subsets):
			print 'analyzing structure for ', i+1, " of ", len(subsets), ' feature sets.'
			structure = analyze_structure(data, x, label_list) # dictionary with six keys
			structures.append(structure) # list of dictionaries
		
		# determine good feature sets
		good_sets = predict(structures)
			
		# write results to file
		txt = text_names[j]
		print 'writing results from label ', label_list
		w = open(txt, 'w')
		w.write(str(label_list) + "\n")
		w.write("Features_Included\n")
		if not good_sets:
			w.write("this classifier does not produce good results")
		else:
			for record in good_sets:
				w.write(str(record["subset"]) + "\n")

if __name__=='__main__':
	parser = OptionParser()
	parser.add_option('-f', '--filename')
	parser.add_option('-t', '--feature_groups', help = 'for example 1:28,29:40 will make two feature groups 1-28 and 29-40')
	parser.add_option('-l', '--class_labels', help = 'enter class types separated by commas')
	parser.add_option('-d', '--directory_name', help = 'create directory for output files')
	options, args = parser.parse_args()
	main(options.filename, options.feature_groups, options.class_labels, options.directory_name)
	
	
	
	
	
	
	
	
	
	
	
	
	
	