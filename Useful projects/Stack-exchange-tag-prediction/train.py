import sys, cPickle, random, os
import numpy as np
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import scripts.preprocessing as p
from gensim.models import Doc2Vec
from gensim  import utils
from scipy import sparse
from collections import Counter
stops = set(stopwords.words('english'))
# d2vdata = Doc2Vec.load('./stackovf300.d2v')

totalNumTags = -1
dataloc = "data/"
dataFile = dataloc+"trainsmall"
numTagFile = dataloc+"numtags"

# trainingPosEg = 100
# trainingNegEg = 1500
# testingPosEg = 10
# testingNegEg = 10


with open(dataFile,'r') as ff:
	fulldata = cPickle.load(ff)

with open(numTagFile,'r') as ff:
	totalNumTags = cPickle.load(ff)


# with open(os.path.join('data', 'words'), 'r') as fid:
#     words2 = cPickle.load(fid)

with open(os.path.join('data', 'alltags'), 'r') as fid:
    alltags = cPickle.load(fid)

print "data read"

# worddict = {}
# affinitydict = {}
# words = [a for (a,b) in words2]
# for w in words:
# 	worddict[w] = 0
# for t in alltags:
# 	affinitydict[t] = worddict

# print affinitydict['c']
for i in xrange(len(fulldata)):
	# print i
	fulldata[i][1] = p.preprocessTitle(fulldata[i][1])+" "+p.preprocessBody(fulldata[i][2])
	# tags = fulldata[i][3].split(' ')
	# ws = fulldata[i][1].split()
	# wsdict = Counter(ws)
	# for w in wsdict.keys():
	# 	if w in affinitydict[t].keys():
	# 		for t in tags:
	# 			affinitydict[t][w] = affinitydict[t][w] + wsdict[w]
	fulldata[i] = fulldata[i][1]
# print affinitydict['c']

print "data preprocessed"

tfidfvect = TfidfVectorizer(
						#encoding = "Latin-1"
					#,	ngram_range=(1,2)
					#,	binary = True
					#,	stop_words='english'
					#,	lowercase=True
						max_df = 0.95,
						min_df = 4,
						decode_error = 'ignore'
					)
tfidfvect.fit(fulldata)
# countvect = CountVectorizer(
# 						#encoding = "Latin-1"
# 					#,	ngram_range=(1,2)
# 					#,	binary = True
# 					#,	stop_words='english'
# 					#,	lowercase=True
# 						max_df = 0.95,
# 					#,	min_df = 0.1
# 						decode_error = 'ignore'
# 					)
# countvect.fit(fulldata)
# with open(os.path.join('models', 'vocabcount'), 'w') as fid:
#     cPickle.dump(countvect, fid)

with open(os.path.join('models', 'vocab'), 'w') as fid:
    cPickle.dump(tfidfvect, fid)



#Now tfidfvectorizer is ready .. Go ahead and train for each tag
print "Tfidf created"

acc = 0.0
for curTag in xrange(totalNumTags):
	print "Curtag "+str(curTag)
	posFileName = dataloc+str(curTag)+"pos"
	negFileName = dataloc+str(curTag)+"neg"
	with open(posFileName,'r') as ff:
		posAll = cPickle.load(ff)
		tagname = cPickle.load(ff)
	with open(negFileName,'r') as ff: 
		negAll = cPickle.load(ff)
	# assert(len(posAll) == trainingPosEg+testingPosEg)
	# assert(len(negAll) == trainingNegEg+testingNegEg)

	posAll2 = [p.preprocessTitle(xx[0])+" "+ p.preprocessBody(xx[1]) for xx in posAll]
	negAll2 = [p.preprocessTitle(xx[0])+" "+ p.preprocessBody(xx[1]) for xx in negAll]

	# for i in xrange(len(posAll2)):
	# 	posAll2[i] = posAll2[i] + (posAll3[i])
	# 	negAll2[i] = posAll2[i] + (negAll3[i])

	random.shuffle(posAll2)
	random.shuffle(negAll2)
	
	posTrainData = posAll2[:-15]
	posTestData = posAll2[-15:]

	negTrainData = negAll2[:-15]
	negTestData = negAll2[-15:]

	trainingPosEg = len(posAll)-15
	trainingNegEg = len(negAll)-15
	testingPosEg = 15
	testingNegEg = 15

	indList = range(trainingPosEg+trainingNegEg)
	random.shuffle(indList)
	# Label is 1 for positive and 0 for negative
	trainLabel = [(lambda x: 1 if x<trainingPosEg else 0)(x) for x in indList]

	trainData2 = posTrainData + negTrainData
	trainData = [trainData2[i] for i in indList]
	# traind2vfeature = [d2vdata.infer_vector(utils.to_unicode(xx).split()) for xx in trainData]
	#Have the trainData and trainLabel now ready

	indList = range(testingPosEg+testingNegEg)
	random.shuffle(indList)
	testGoldLabel = [(lambda x: 1 if x<testingPosEg else 0)(x) for x in indList]

	testData2 = posTestData+negTestData
	testData = [testData2[i] for i in indList]
	# testd2vfeature = [d2vdata.infer_vector(utils.to_unicode(xx).split()) for xx in testData]
	#Have the testData and testGoldLabel now ready


	# Now train classifier
	trainDataVect = tfidfvect.transform(trainData)
	# clf=SVC(
	# 	class_weight='balanced',
	# 	kernel = 'linear',
	# 	probability=True
	# 	).fit(trainDataVect, trainLabel)
	# trainDataVect = sparse.hstack([trainDataVect2, sparse.csr_matrix(traind2vfeature)])

	clf = LogisticRegression(
					 class_weight = 'balanced'
					 # , solver='sag'
					).fit(
						trainDataVect
						, trainLabel
						# , sample_weight=sample_weights
						)
	with open(os.path.join('models', str(curTag)+'clf'), 'w') as fid:
		cPickle.dump(clf, fid)
		cPickle.dump(tagname,fid)

	testDataVect = tfidfvect.transform(testData)
	# testDataVect = sparse.hstack([testDataVect2, sparse.csr_matrix(testd2vfeature)])
	predicted = clf.predict(testDataVect)
	curacc = np.mean(predicted == testGoldLabel)
	print list(predicted)
	print testGoldLabel
	print tagname
	print "--------------------"
	acc += curacc

acc = acc/totalNumTags
print acc