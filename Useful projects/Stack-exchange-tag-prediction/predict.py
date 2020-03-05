import csv, cPickle, nltk, re, sys, random, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from nltk.stem import *
import scripts.preprocessing as p
from gensim.models import Doc2Vec
from gensim  import utils
from scipy import sparse

# d2vdata = Doc2Vec.load('./stackovf300.d2v')

## Prediction ##
# fname = "data/"
# with open(fname, 'r') as f:
#     csv_reader = csv.reader(f)
#     i=-1
#     for row in csv_reader:
#     	i=i+1
#     	if exist.has_key(i):
#     		randomdata.append(row)
#     	else:
#     		continue

with open(os.path.join('data', 'testdata'),'r') as ff:
	testdata = cPickle.load(ff)

with open(os.path.join('models', 'vocab'),'r') as ff:
	tfdifVect = cPickle.load(ff)

with open(os.path.join('data', 'numtags'),'r') as ff:
	numtags = cPickle.load(ff)

print "Data Read"
testScore = [[] for i in xrange(len(testdata))]
questext = ["" for i in xrange(len(testdata))]
tag_names = {}
for i in xrange(numtags):
	print i
	with open(os.path.join('models', str(i)+"clf"),'r') as ff:
		clf = cPickle.load(ff)
		tagname = cPickle.load(ff)
	tag_names[i] = tagname
	testd2vfeature = []
	for j in xrange(len(testdata)):
		questext[j] = p.preprocessTitle(testdata[j][0])+" "+ p.preprocessBody(testdata[j][1])

	testDataVect = tfdifVect.transform(questext)
	# testd2vfeature = [d2vdata.infer_vector(utils.to_unicode(xx).split()) for xx in questext]
	# testfeature = sparse.hstack([testDataVect, sparse.csr_matrix(testd2vfeature)])

	tagTestScore= clf.predict_log_proba(testDataVect)
	for j in xrange(len(tagTestScore)):
		testScore[j] = testScore[j] + [ (tagTestScore[j][1],i) ]

print "Log probabilities predicted for each tag"

with open(os.path.join('data', 'prediction'), 'w') as fid:
	cPickle.dump(testScore, fid)

# with open(os.path.join('data', 'prediction'),'r') as ff:
# 	testScore = cPickle.load(ff)

Recallscore = 0.0
predicted_tags = []
for i in xrange(len(testScore)):
	sorted_tags = sorted(testScore[i], reverse=True)
	testScore[i] = sorted_tags[:5]
	predicted_tags.append([tag_names[b] for (a,b) in testScore[i]])
	actualtags = set(testdata[i][2].split(' '))
	print predicted_tags[i], actualtags
	Recallscore = Recallscore + ( 1.0 * len(set(predicted_tags[i]) & actualtags) / len(actualtags) )

	# tags = set(testdata[i][2].split(' '))
	# match = len( predicted_tags & tags )
	# precision = (1.0*match)/len(predicted_tags)
	# recall = (1.0*match)/len(tags)
	# F1 = 2*precision*recall/(precision+recall)
	# print F1

print " Recall Score ", Recallscore/len(testdata)

# # F-score Macro ##
# F1macro = []
# print predicted_tags
# for i in xrange(numtags):
# 	# print i
# 	fp=0
# 	fn=0
# 	tp=0
# 	tn=0
# 	# print tag_names[i]
# 	# print '---'
# 	for j in xrange(len(predicted_tags)):
# 		# print testdata[j][2]
# 		if tag_names[i] in predicted_tags[j]:
# 			if(tag_names[i] in testdata[j][2].split(' ')):
# 				tp = tp+1
# 			else:
# 				fp = fp+1
# 		else:
# 			if(tag_names[i] in testdata[j][2].split(' ')):
# 				fn = fn+1
# 			else:
# 				tn = tn+1
# 	precision = 1.0*tp/(tp+fp)
# 	recall = 1.0*tp/(tp+fn)
# 	F1macro.append(2*precision*recall/(precision+recall))

# print sum(F1macro)/len(F1macro)

