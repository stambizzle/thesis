import copy
import csv
import numpy
import operator
import random
import sys
import math
import numpy.linalg
from optparse import OptionParser

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

def parse_features(curr_vector):
	return [int(atts.split(":")[0]) for atts in curr_vector]

def feature_selection(curr_vector, features_to_use):
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
	for record in data:
		record['feature_vector'] = feature_selection(record['feature_vector'], curr_subset)

	reindexed, values = reindex(data)
	splitlist = splitter(reindexed, label_list)
	pos = splitlist[0]
	neg = splitlist[1]
	posmatrix = build_matrix(pos, values)
	negmatrix = build_matrix(neg, values)
	matrix = build_matrix(reindexed, values)

	# identify number of unique values in each class
	uniqpos = column_totals(posmatrix, values)
	uniqneg = column_totals(negmatrix, values)

	# calculate the dimension of each polytope
	# if zero vector is present, then dimension is equal to rank, otherwise it is equal to rank - 1
	if zero_check(pos) == True:
		dimpos = my_matrix_rank(posmatrix, values)
	else:
		dimpos = my_matrix_rank(posmatrix, values) - 1
	if zero_check(neg) == True:
		dimneg = my_matrix_rank(negmatrix, values)
	else:
		dimneg = my_matrix_rank(negmatrix, values) - 1
	if zero_check(reindexed) == True:
		dimfull = my_matrix_rank(matrix, values)
	else:
		dimfull = my_matrix_rank(matrix, values) - 1

	# create polytope features
	f1 = float(dimpos)/uniqpos
	f2 = float(dimneg)/uniqneg
	f3 = float(dimpos)/len(values)
	f4 = float(dimneg)/len(values)
	f5 = float(dimfull)/len(values)

	# calculate affine overlap ratio
	PsetminusQ = affine_hull_intersection(pos, neg, values)
	QsetminusP = affine_hull_intersection(neg, pos, values)
	overlap = len(reindexed) - PsetminusQ - QsetminusP
	f6 = float(overlap)/len(reindexed)


	output = [f1,f2,f3,f4,f5,f6,curr_subset]
	return output

# determines ambient dimension
def column_totals(matrix, values):
	count = 0
	for i in xrange(len(values)):
		z = matrix[:,i].sum()
		if z > 0:
			count +=1
	return count

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

	# rename each unique feature value as its index
	new_data = []
	for record in data:
		new_record = copy.deepcopy(record)
		new_vector = []
		curr_vector = record['feature_vector']
		for item in curr_vector:
			for i, idx in enumerate(values):
				if item == idx:
					new_vector.append(i)
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
	lists = {}
	lists[0] = list1
	lists[1] = list2
	return lists

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

# check for presence of zero vector
def zero_check(records):
	for record in records:
		if not record['feature_vector']:
			return True
	return False

# determines the overlap of the class represented polytopes
def affine_hull_intersection(examples1,examples2, values):
	setminus = 0
	num_features = len(values)		
	poly1 = build_matrix(examples1,values)
	poly2 = build_matrix(examples2,values)

	if zero_check(examples1) == True: # dimension = rank if zero vector present
		dimpoly1 = my_matrix_rank(poly1, values) 
		for record in examples2:
			if not record['feature_vector']: # already know poly1 contains the origin
				continue
			else: # determines whether a new point from poly2 is within the the affine hull of poly1
				features = record['feature_vector']
				vector = [0 for i in xrange(num_features)]
				for val in features:
					vector[int(val)] = 1
				new_matrix = numpy.vstack((poly1,vector))
				dimnew = my_matrix_rank(new_matrix, values)
				if dimnew > dimpoly1: # count the points that belong to poly2, but not poly1
					setminus +=1

	elif zero_check(examples1) == False:
		dimpoly1 = my_matrix_rank(poly1, values) - 1
		for record in examples2:
			if not record['feature_vector']: #already know poly1 does not contain origin
				setminus +=1
			else:
				features = record['feature_vector']
				vector = [0 for i in xrange(num_features)]
				for val in features:
					vector[int(val)] = 1
				new_matrix = numpy.vstack((poly1,vector))
				dimnew = my_matrix_rank(new_matrix, values) - 1
				if dimnew > dimpoly1:
					setminus +=1

	return setminus

# this function makes up for buggy SVD function from the linalg library. It offers a conservative estimate 
# for rank (eps=1e-6), and returns the largest possible value for rank if SVD does not converge.
def my_matrix_rank(A, values,  eps=1e-6):
	try:
		u, s, vh = numpy.linalg.svd(A)
		return len([x for x in s if abs(x) > eps])
	except numpy.linalg.LinAlgError:
		print "SVD did not converge...making a conservative estimate for rank"
		return column_totals(A, values) 


def generate_filenames(label_lists):
	names = []
	for i in xrange(1, len(label_lists)+1):
		curr_name = 'classifier%d.csv'%i
		names.append(curr_name)
	return names
	
def predict(data):	
	def normalize(data):
		# normalize values
		types = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
		for idx in types:
			vals = []
			for record in data:
				vals.append(float(record[idx]))
			mean = numpy.mean(vals)
			sd  = numpy.std(vals, ddof = 1)
			# avoid division by zero if all values are the same
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
		if record['lin_pred'] > 0 and init_pred == 1:
			pred = 1
		else:
			pred = -1
		record['pred'] = pred
		return record

	normalize(data)
	for record in data:
		lin_predict(record)
	for record in data:
		log_predict(record)
	good_sets = []
	for record in data:
		if record['pred'] == 1:
			good_sets.append(record)
	return good_sets

# argument is a text file with labeled training examples in sparse vector format
# file needs to be formatted as follows:  label' 'feature#:1' 'feature#:1' '...
def main(filename, feature_groups, class_labels):
	data = read_data(filename)
	random.shuffle(data)
	features = feature_groups.split(',')
	labels = class_labels.split(',')
	subsets = make_subsets(features)
	subsets = subsets[1:] # do not generate structures for the empty set
	label_lists = make_label_lists(labels)
	label_lists = label_lists[1:]
	text_names = generate_filenames(label_lists)
	for j,label_list in enumerate(label_lists):
		print "working on ", j+1, " of ", len(label_lists), " binary classifiers"
		structures = []
		for i, x in enumerate(subsets):
			print 'analyzing structure for ', i+1, " of ", len(subsets), ' feature sets.'
			structure = analyze_structure(data, x, label_list) # list of 6 values and subset
			structures.append(structure) # list of lists
		
		# make subset stats into dictionaries
		structs = [] 	
		for struct in structures:
			info = {}
			info['f1'] = struct[0]
			info ['f2'] = struct[1]
			info['f3'] = struct[2]
			info['f4'] = struct[3]
			info['f5'] = struct[4]
			info['f6'] = struct[5]
			info['subset'] = struct[6]
			structs.append(info)	
		
		# determine good feature sets
		good_sets = predict(structs)
			
		# write results to file
		txt = text_names[j]
		print 'writing results from label ', label_list
		w = csv.writer(open(txt, 'w'), dialect = 'excel')
		w.writerow(label_list)
		w.writerow(['Features_Included'])
		for record in good_sets:
			output = [record['subset']]
			w.writerow(output)

if __name__=='__main__':
	parser = OptionParser()
	parser.add_option('-f', '--filename')
	parser.add_option('-t', '--feature_groups', help = 'for example 1:28,29:40 will make two feature groups 1-28 and 29-40')
	parser.add_option('-l', '--class_labels', help = 'enter class types separated by commas')
	options, args = parser.parse_args()
	main(options.filename, options.feature_groups, options.class_labels)
	
	
	
	
	
	
	
	
	
	
	
	
	
	