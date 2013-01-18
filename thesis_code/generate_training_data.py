# this program creates every possible subset of the feature types.
# for each subset it generates 3 matrices based on the full data for the feature set,
# the positive examples of the data set and the negative examples for the data set.
# information about the SVs of each matrix is recorded.
# a point cloud is built from the affine hull of the points contained in each matrix,
# and information about each of the point clouds is collected.
# the feature set is sent to the SVM, where a classifier is trained and then tested
# with a linear kernel. 
# Once this information has been collected for each feature subset, the feature sets are labeled
# as optimal or suboptimal, using a mann-whitney test for significance.


import copy
import csv
import operator
import random
import sys
import numpy
sys.path.append('/Users/carlystambaugh/util')
import svmlight
from optparse import OptionParser
from operator import itemgetter
from itertools import groupby


class SVMLight:
	def __init__(self):
		self.feature_map = {}

	def vectorize(self, record):
		return [(idx, 1) for idx in record['feature_vector']]	
			
	def train(self, data, label_list):
		currlists = splitter(data,label_list)
		list1 = currlists[0]
		list2 = currlists[1]

		#make into parallel lists of labels and vectors
		pos_labels, pos_vectors = [], []
		neg_labels, neg_vectors = [], []
		for record in list1:
			pos_labels.append(1)
			pos_vectors.append(dict(self.vectorize(record)))
		for record in list2:
			neg_labels.append(0)
			neg_vectors.append(dict(self.vectorize(record)))

		all_labels = pos_labels + neg_labels
		all_vectors = pos_vectors + neg_vectors
		self.model = svmlight.SVMLight(all_labels, all_vectors, kernel = 'linear')
	
	def predict(self, record):
		return self.model.classify([dict(self.vectorize(record))])[0]
		
# reads in training data
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

# changes sparse vector notation to list of present features
def parse_features(curr_vector):
	return [int(atts.split(":")[0]) for atts in curr_vector]
	
# performs feature selection for current feature set-classifier pair
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
		 
# performs 10-fold cross validation for an individual feature set-classifier pair	
def feature_set_ranker(data, row, label_list):
	
	data = copy.deepcopy(data)
	curr_subset = row['subset']

	# create feature vector for curr_subset
	for record in data:
		record['feature_vector'] = make_feature_set(record['feature_vector'], curr_subset)
	
	# build total confusion matrix
	total_confusion = {}
	total_confusion[1,1] = 0
	total_confusion[1,-1] = 0
	total_confusion[-1,1] = 0
	total_confusion[-1,-1] = 0
	
	# keep track of cv_performance
	cvaccs = []
	
	#10 fold cross validation
	for i in range(0,10):
		
		# build cv confusion matrix
		confusion = {}
		confusion [1,1] = 0		#true positives
		confusion[1,-1] = 0		#false negatives
		confusion[-1,1] = 0		#false positives
		confusion[-1,-1] = 0	#true negatives
	
		#split data into training set and test set
		train = []
		test = []
		for x, record in enumerate(data):
			if x%10 == i:
				test.append(record)
			else:
				train.append(record)
		
		# train classifier
		c = SVMLight()
		c.train(train, label_list)
				
		# fill in confusion matrices
		for record in test:
			
			# get actual label
			if record['class_label'] in label_list:
				actual = 1
			else:
				actual = -1
			
			# get predicted label
			plabel = c.predict(record)
			
			#update	confusion matrix
			if plabel > 0:
				plabel = 1
			else:
				plabel = -1
			confusion[actual, plabel] +=1
			total_confusion[actual,plabel] += 1
		
		# generate stats for cv trial
		cv_accuracy = float(confusion[1,1] + confusion[-1,-1]) / len(test)
		cvaccs.append(cv_accuracy)
		
	row['cvaccs'] = cvaccs
		
	# generate stats for classifier	
	if total_confusion[1,1] == 0 or total_confusion[-1,-1] == 0:# this identifies instances where classifier assigns every test point the same class.
		precision = 0
		recall = 0
		fscore = 0
	else:
		precision = float(total_confusion[1,1]) / (total_confusion[1,1] + total_confusion[-1,1])
		recall = float(total_confusion[1,1]) / (total_confusion[1,1] + total_confusion[1,-1])
		fscore = 2 * (precision * recall) / (precision + recall)
	accuracy = float((total_confusion[1,1] + total_confusion[-1,-1]))/float(len(data))
	row['fscore'] = fscore
	row['accuracy'] = accuracy
	

	return row
	
# performs the Mann-Whitney Test for two list of cross validation accuracies.
def mann_whitney(best, test):
	# integrate lists and sort
	all_vals = []
	for acc in best:
		all_vals.append(acc)
	for acc in test:
		all_vals.append(acc)
	all_vals.sort(reverse = True)

	# assign ranks
	uniqs = groupby(all_vals, key = lambda x : x)
	place = 1
	ranks = {}
	for key, pairs in uniqs:
		ties = list(pairs)
		n = len(ties)
		tot = 0
		for x in xrange(place, place + n):
			tot = tot + x 
		rank = float(tot) / n
		ranks[key] = rank
		place = place + n

	# split back into seperate lists of ranks
	best_ranks = []
	test_ranks = []
	for acc in best:
		best_ranks.append(ranks[acc])
	for acc in test:
		test_ranks.append(ranks[acc])
	
	# sum ranks for each list
	Tbest = 0
	for val in best_ranks:
		Tbest = Tbest + val
	Ttest = 0
	for val in test_ranks:
		Ttest = Ttest + val
	
	# set Tx = higher sum
	Tx = max(Tbest, Ttest)

	# n1 = n2 = nx = 10
	# U = n1 * n2 + (nx * ((nx+1) / 2) - Tx
	U = 10 * 10 + (10 *(11 / 2)) - Tx

	if U <= 23:
		return True
	else:
		return False

# labels feature set-classifier pairs as optimal(1) or suboptimal(-1)
def label_maker(structures):

	# rank feature sets on mean accuracy
	structures.sort(key=operator.itemgetter('accuracy'), reverse = True)
	best_cvs = structures[0]['cvaccs']

	for record in structures:
		if record['subset'] == set([]):
			empty_cvs = record['cvaccs']
		
	# test best and empty
	significant = mann_whitney(best_cvs, empty_cvs)
	if significant == False: # top feature set and empty feature set are not significantly different 
		for record in structures:
			record['set_label'] = -1 # no subsets are optimal
	else:
		for record in structures:
			curr_cvs = record['cvaccs']
			significant = mann_whitney(best_cvs, curr_cvs)
			if significant == True: # there is a significant difference between this feature set and the top feature set
			 	record['set_label'] = -1 # therefor it is suboptimal
			else:
				record['set_label'] = 1 # otherwise it is optimal
	return structures

# generate list of all unique binary classifiers
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
	
# generates structural information for each feature set
def analyze_structure(data, curr_subset, label_list):
	data = copy.deepcopy(data)

	#create dictionary for current test
	info = {}
	info['subset'] = curr_subset

	# create feature vector for curr_subset
	for record in data:
		record['feature_vector'] = make_feature_set(record['feature_vector'], curr_subset)

	reindexed, values = reindex(data)
	pos, neg = splitter(reindexed, label_list)
	posmatrix = build_matrix(pos, values)
	negmatrix = build_matrix(neg, values)
	matrix = build_matrix(reindexed, values)

	# identify number of unique values in each class
	uniqpos = column_totals(posmatrix, values)
	uniqneg = column_totals(negmatrix, values)

	# calculate polytope dimension
	dimpos = get_poly_dim(posmatrix, values)
	dimneg = get_poly_dim(negmatrix, values)
	dimfull = get_poly_dim(matrix, values)

	# create polytope features	
	info['f1'] = float(dimpos)/uniqpos
	info ['f2'] = float(dimneg)/uniqneg
	info['f3'] = float(dimpos)/len(values)
	info['f4'] = float(dimneg)/len(values)
	info['f5'] = float(dimfull)/len(values)

	# calculate affine overlap ratio
	PsetminusQ = affine_hull_intersection(pos, neg, values)
	QsetminusP = affine_hull_intersection(neg, pos, values)
	overlap = len(reindexed) - PsetminusQ - QsetminusP
	info['f6'] = float(overlap)/len(reindexed)
	
	return info
	
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
	
# determines ambient dimension
def column_totals(matrix, values):
	count = 0
	for i in xrange(len(values)):
		z = matrix[:,i].sum()
		if z > 0:
			count +=1
	return count

# calculate the affine dimension of each polytope
# if zero vector is present, then dimension is equal to rank, otherwise it is equal to rank - 1
def get_poly_dim(mat, values):
	d = mat[-1]
	k = mat[:-1]
	g = k-d
	dim = my_matrix_rank(g, values)
	return dim

def zero_vector_present(records):
	for record in records:
		if not record['feature_vector']:
			return True
	return False

# determines the overlap of the class represented polytopes
def affine_hull_intersection(examples1, examples2, values):
	setminus = 0
	num_features = len(values)		
	poly1 = build_matrix(examples1, values)
	poly2 = build_matrix(examples2, values)
	dimpoly1 = get_poly_dim(poly1, values)

	if zero_vector_present(poly1): 
		for record in examples2:
			if not record['feature_vector']: # already know poly1 contains the origin
				continue
			else: 
				if dimension_increase_check(record, num_features, values, poly1, dimpoly1):
					setminus +=1

	else:
		for record in examples2:
			if not record['feature_vector']: #already know poly1 does not contain origin
				setminus +=1
			else:
				if dimension_increase_check(record, num_features, values, poly1, dimpoly1):
					setminus +=1

	return setminus
	
# adds a point from poly2 to poly1
# an increase in dimension indicates point does not lie in affine hull of poly1
def dimension_increase_check(record, num_features, values, poly1, dimpoly1):	
	vector = numpy.zeros(num_features)
	vector[record['feature_vector']] = 1
	new_matrix = numpy.vstack((poly1, vector))
	dimnew = get_poly_dim(new_matrix, values)
	if dimnew > dimpoly1:
		return True
	else:
		return False
		
# this function makes up for buggy SVD function from the linalg library. It offers a conservative estimate 
# for rank (eps=1e-6), and returns the largest possible value for rank if SVD does not converge.		
def my_matrix_rank(A, values,  eps=1e-6):
	try:
		u, s, vh = numpy.linalg.svd(A)
		return len([x for x in s if abs(x) > eps])
	except numpy.linalg.LinAlgError:
		print "SVD did not converge...making a conservative estimate for rank"
		return column_totals(A, values)
	
def main(filename, feature_groups, class_labels, output):
	# read in data
	data = read_data(filename)
	random.shuffle(data)
	
	# create lists from args
	features = feature_groups.split(',')
	labels = class_labels.split(',')
	subsets = make_subsets(features)
	label_lists = make_label_lists(labels)
	label_lists = label_lists[1:] # do not run for empty label_list
	
	# set up output file
	w = csv.writer(open(output, 'w'), dialect = 'excel')
	header1 = ["features included"]
	header2 = ["accuracy", "fscore", "label",]
	header3 = ['f1','f2','f3','f4','f5','f6']
	header = header1 + header2 + header3	
	w.writerow(header)	
	
	#loop over classifiers
	for i, label_list in enumerate(label_lists):
		print "Building classifier ", i + 1, " of ", len(label_lists)
		w.writerow(label_list)
		
		# generate structural features for subsets
		structures = []
		for j, x in enumerate(subsets):
			print "analyzing structures..." 
			if j == 0:# cannot generate structures for the empty set, so manually set to 0
				structure = {'subset': x, 'f1': 0, 'f2': 0, 'f3': 0, 'f4': 0, 'f5': 0, 'f6': 0}
			else:
				structure = analyze_structure(data, x, label_list)
			structures.append(structure)
				
		# generate feature set ranking information for subsets
		print "testing ", len(subsets), "feature sets for the ", label_list, "binary classifer"
		for row in structures:
			feature_set_ranker(data, row, label_list)
	
		# label subsets 
		label_maker(structures)

		# write output to file
		for record in structures:
			subset_output = [record['subset']]
			rank_output = [record['accuracy'], record['fscore'], record['set_label']]
			struct_output = [record['f1'], record['f2'], record['f3'], record['f4'], record['f5'], record['f6']]
			output = subset_output + rank_output + struct_output
			w.writerow(output)
			
if __name__=='__main__':
	parser = OptionParser()
	parser.add_option('-f', '--filename')
	parser.add_option('-o', '--output')
	parser.add_option('-t', '--feature_groups', help = 'for example 1:28,29:40 will make two feature groups 1-28 and 29-40')
	parser.add_option('-l', '--class_labels', help = 'enter class types separated by commas')
	options, args = parser.parse_args()
	main(options.filename, options.feature_groups, options.class_labels, options.output)
# vim: noexpandtab ts=4 sts=4 sw=4
