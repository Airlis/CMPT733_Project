import random, csv, cPickle, os, re
import pandas
import preprocessing as p

random.seed(2)
fname = "data/Train.csv"
n = 6034196 # This includes header #145447257 lines
s = 100000
numtags = 300
keep = sorted(random.sample(xrange(1,n),s))
print "keep created"
exist = {}
for i in keep:
	exist[i]=1

# stopreg = re.compile(p.stopsused)


# with open(os.path.join('data', 'kMeansClusters'), 'r') as fil:
#     clusterids = cPickle.load(fil)

with open(os.path.join('data', 'stackSyn'), 'r') as fil:
    stackSyn = cPickle.load(fil)

# print keep
randomdata = []
randomdatacount=0
codecount = 0
titles = []
with open(fname, 'r') as f:
    csv_reader = csv.reader(f)
    i=-1
    for row in csv_reader:
        if i%1000==0:
            print i
    	i=i+1
    	if exist.has_key(i):
            if( ((len(row[1])+len(row[2])) >500)  and len(row)>3):
                if row[1] in titles:
                    continue
                else:
                    titles.append(row[1])
                row[1] = re.sub('\s+', ' ', p.stopwordhandle(p.linksreg.sub(' LINK ',p.rep.replace(row[1].lower()))))
                # l = row[1].split()
                # for w in l:
                #     if clusterids.has_key(w):
                #         row[1] = row[1] + ' ' + str(clusterids[w])
                row[2] = re.sub('\s+', ' ', p.stopwordhandle(p.linksreg.sub(' LINK ',p.rep.replace(row[2].lower()))))
                # l = row[2].split()
                # for w in l:
                #     if clusterids.has_key(w):
                #         row[2] = row[2] + ' ' + str(clusterids[w])
                tags = row[3].split()
                for j in xrange(len(tags)):
                    if tags[j] in stackSyn.keys():
                        tags[j] = stackSyn[tags[j]]
                row[3] = ' '.join(tags)
                randomdata.append(row)
    	else:
    		continue

# print "ques with code count", str(codecount)
print len(randomdata)
print "random data created"
# print randomdata
tags_count = {}
tag_names = {}
i=-1
for row in randomdata:
    i = i+1
    tags = row[3].split(' ')
    for tag in tags:
		if(tags_count.has_key(tag)):
			tags_count[tag].append(i)
		else:
			tags_count[tag]=[i]

with open(os.path.join('data', 'alltags'), 'w') as fid:
    cPickle.dump(tags_count, fid)

with open(os.path.join('data', 'numtags'), 'w') as fid:
    cPickle.dump(numtags, fid)

sorted_tags = sorted(((len(v), k) for k, v in tags_count.items()), reverse=True)
sorted_tags = sorted_tags[:numtags]
print "Min count ",sorted_tags[-1][0]

freq_tags = [b for (a,b) in sorted_tags]

for key in tags_count.keys():
    if (key) in freq_tags:
        continue
    else:
        tags_count.pop(key, None)

# with open(os.path.join('data', 'tagsmall'), 'w') as fid:
#     cPickle.dump(tags_count, fid)

with open(os.path.join('data', 'trainsmall'), 'w') as fid:
    cPickle.dump(randomdata, fid)
    # cPickle.dump(len(tags_count), fid)

# print len(tags_count)
print "written tags and data"

datacount = len(randomdata)
total = set(range(datacount))
test = set(random.sample(range(datacount),10000))
train = total - test
j=0
for v,k in sorted_tags:
    print j
    pos = set(tags_count[k])
    neg = train - pos
    l = random.sample(pos,min(3000,len(pos)))
    m = random.sample(neg, max( min(len(neg),len(l)) , 300 ))
    # train = train | pos | set(m)
    positive = [randomdata[i][1:4] for i in l]
    negative = [randomdata[i][1:4] for i in m]
    with open(os.path.join('data', str(j)+"pos"), 'w') as fid:
        cPickle.dump(positive, fid)
        cPickle.dump(k,fid)

    with open(os.path.join('data', str(j)+"neg"), 'w') as fid:
        cPickle.dump(negative, fid)
    j = j + 1

print "written data for each tag"

# testindices = total-train
# print "Test indices lenth ", len(testindices)
# testsize = min(10000,len(testindices))
# test = random.sample(testindices,testsize)
testdata = [randomdata[i][1:4] for i in test]
with open(os.path.join('data', "testdata"), 'w') as fid:
    cPickle.dump(testdata, fid)

print "Test Data written"

# sink = open(os.path.join('data', 'Trainsmall.csv'),"w")
# writer = csv.writer(sink, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
# for row in randomdata:
# 	writer.writerow(row)
# sink.close()
# print "written data"











#######

# Data -  - #Tags - 30044
# tags with ques 2000- 72
# tags with ques 1000 - 198

