import nltk
from nltk.tag.simplify import simplify_wsj_tag

#read in reviews from corpus, tokenize each review
print "reading in reviews..."
all_reviews = [] # list of dictionaries with words and label as keys
every_word = [] # all tokens
for item in nltk.corpus.movie_reviews.fileids():
	info = {}
	tokes = nltk.corpus.movie_reviews.words(item)
	tokens = [w for w in tokes if not w in nltk.corpus.stopwords.words('english')]
	for token in tokens:
		every_word.append(token)
	pieces = item.split('/')
	info['label'] = pieces[0]# label is determined by directory file came from (positive or negative)
	info['tokens'] = tokens
	all_reviews.append(info)

# find 1000 most frequent words 
freq = nltk.FreqDist(every_word)
uni_feats = freq.keys()
uni_feats.sort(key=lambda x: freq[x], reverse=True)
top_feats = uni_feats[:1000]

#gets rid of strings less than 2 characters 
for record in all_reviews:
	words = []
	for strang in record['tokens']:
		if len(strang) < 2:
			continue
		else:
			if strang in top_feats:
				words.append(strang)
	record['words'] = words
	
# pos tags using simple tags
print "doing part of speech tagging..."	
for record in all_reviews:
	tags = nltk.pos_tag(list(record['words']))
	simple = [(word, simplify_wsj_tag(tag)) for word, tag in tags]
	record['simple_tags'] = simple
	
# interested in these tags
parts = ['ADV', 'VD', 'VG', 'VN', 'N', 'V', 'NP', 'ADJ', 'MOD']
print "filtering tags..."
for record in all_reviews:
	curr_tags = record['simple_tags']
	short = []
	for pair in curr_tags:
		if pair[1] in parts:
			short.append((pair[0],pair[1]))
	record['short_list_tags'] = short

# assembles new list of dictionaries with only relevant keys
data_set = []
for record in all_reviews:
	info = {}
	info['label'] = record['label']
	info['short_list_tags'] = record['short_list_tags']
	data_set.append(info)


# establish feature types	 
type1 = ['ADV','ADJ'] #descriptors
type2 =['V', 'VD', 'VG', 'VN', 'MOD'] #verbs
type3 = ['N', 'NP'] #nouns

# replace POS tags with feature types
for record in data_set:
	curr_tags = record['short_list_tags']
	new_tags = []
	for pair in curr_tags:
		if pair[1] in type1:
			new_tags.append((pair[0],'DESC'))
		elif pair[1] in type2:
			new_tags.append((pair[0],'VERB'))
		elif pair[1] in type3:
			new_tags.append((pair[0],'NOUN'))
	record['new_tags'] = new_tags


#create list of unique words for feature map
all_words = []
descs = []
verbs = []
nouns = []
for record in data_set:
	for word, ftype in record['new_tags']:
		if ftype == 'DESC':
			descs.append((word, ftype))
		elif ftype == 'VERB':
			verbs.append((word, ftype))
		elif ftype == 'NOUN':
			nouns.append((word, ftype))	

wordsies = set()
for word, ftype in descs:
	if word in wordsies:
		continue
	else:
		wordsies.add(word)
		all_words.append((word, ftype))
for word, ftype in verbs:
	if word in wordsies:
		continue
	else:
		wordsies.add(word)
		all_words.append((word, ftype))
for word, ftype in nouns:
	if word in wordsies:
		continue
	else:
		wordsies.add(word)
		all_words.append((word, ftype))

# creates feature map
w = open('movie_review_featuremap.txt','w')
feats = {}
for i,item in enumerate(all_words):
	word = item[0]
	ftype = item[1]
	feats[word] = i+1
	w.write(str(ftype))
	w.write(':')
	w.write(str(word))
	w.write(" ")
	w.write(str(i+1)) 
	w.write('\n')

# create feature vectors
for record in data_set:
	curr_words = record['new_tags']
	vector = []
	for pair in curr_words:
		vector.append(feats[pair[0]])
	record['feature_vector'] = vector

# make sure all records contain at least one attribute
no_blanks = []
for record in data_set:
	if not record['feature_vector']:
		continue
	else:
		no_blanks.append(record)

# print out feature vectors
r = open('movie_review_vectors.txt', 'w')
for record in no_blanks:
	curr_vec = record['feature_vector']
	label = record['label']
	r.write(str(label))
	r.write('\t')
	for val in curr_vec:
		r.write(str(val))
		r.write(':')
		r.write('1')
		r.write('\t')
	r.write('\n')



	
		



