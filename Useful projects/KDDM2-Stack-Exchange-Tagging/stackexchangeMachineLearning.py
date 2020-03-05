from lxml import etree as etree
import pandas as pd
import numpy as np
import re
#nltk.download('stopwords')
import pickle
from sklearn.datasets import load_files
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk
#nltk.download()
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def plotDistribution(counterPosts, counterTags, bars, title):
    plt.rcdefaults()
    fig, ax = plt.subplots()

    y_pos = np.arange(len(bars))
    numberOfPosts = [counterPosts-counterTags,counterTags]

    for i, v in enumerate(numberOfPosts):
        ax.text(v+2, i, str(v),va='center')
    ax.barh(y_pos, numberOfPosts, 0.2, 0.2, align='center', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bars, fontsize=16)
    ax.invert_yaxis() 
    ax.set_xlabel('Number of posts', fontsize=18)
    ax.set_title(title, fontsize=18)
    plt.show()

def createDistributionPlot():
    parsedXml = etree.parse("Tags.xml")
    dfcols = ['tagName', 'count']
    df_xml = pd.DataFrame(columns=dfcols)
    for node in parsedXml.getroot(): 
        tagName = node.attrib.get('TagName')
        count = int(node.attrib.get('Count'))
        df_xml = df_xml.append(pd.Series([(tagName), (count)], index = dfcols), ignore_index = True)

    df_xml['count'] = df_xml['count'].astype(int)
    df_xml.sort_values(by=['count'], inplace=True,ascending= False)
    df = df_xml.nlargest(10, ('count'))
    
    plt.bar(np.arange(10), df['count'], align='center', alpha=0.5)
    plt.xticks(np.arange(10), df['tagName'])
    plt.title('Distribution of tags in travel.stackexchange.com', fontsize=18)
    plt.xlabel('Tags', fontsize=16)
    plt.ylabel ('Number of posts', fontsize=16)
    plt.show()
    return list(df['tagName'])

def numberOfTagsDistribution(numberOfTagsCounters):
    plt.bar(np.arange(len(numberOfTagsCounters)), numberOfTagsCounters, align='center', alpha=0.5)
    for i, v in enumerate(numberOfTagsCounters):
        plt.text(i-0.075, v+150, str(v),va='center')
    plt.xticks(np.arange(len(numberOfTagsCounters)), (1,2,3,4,5))
    plt.title('Distribution of number of tags per post', fontsize=22)
    plt.xlabel('Number of tags', fontsize=20)
    plt.ylabel ('Number of posts', fontsize=20)
    plt.show()

def cleanDataset(topTags):
    parsedXml = etree.parse("Posts.xml")
    root = etree.Element("posts")
    numberOfTagsCounters = [0, 0, 0, 0, 0]
    counterPosts = 0
    counterTop10Tags = 0
    counterTags = 0
    for node in parsedXml.getroot(): 
        counterPosts = counterPosts + 1
        _id = node.attrib.get('Id')
        tags = node.attrib.get('Tags')
        if(tags != None):
            filteredTags = tags.replace("<","").replace(">",",")
            filteredTags = filteredTags[:-1]
            listOfTags = filteredTags.split(",")
            tagsInTop10 = intersection(topTags, listOfTags)
        title = node.attrib.get('Title')
        if(tags != None and title !=None and _id != None) : 
            counterTags = counterTags + 1
            if(len(tagsInTop10)):
                numberOfTagsCounters[len(listOfTags)-1] = numberOfTagsCounters[len(listOfTags)-1] + 1
                row = etree.SubElement(root,'row')
                row.set('Tags', ','.join(tagsInTop10))
                row.set('Title', title)
                row.set('Id',_id)
                counterTop10Tags = counterTop10Tags + 1

    plotDistribution(counterPosts, counterTags,('Without tags', 'With tags'), 'Distribution of posts with and without tags')
    plotDistribution(counterTags,counterTop10Tags, ('Without top 10 tags', 'With top 10 tags'), 'Distribution of top 10 tags in posts')
    numberOfTagsDistribution(numberOfTagsCounters)
    tree = etree.ElementTree(root)    
    tree.write("PostsCleaned.xml")    

def preprocessSupervisedTrainingData(topTags):
    parsedXml = etree.parse("PostsCleaned.xml")
    dfcols = ['id', 'title']
    dfcols = dfcols + topTags
    data = pd.DataFrame(columns=dfcols)
    stemmer = WordNetLemmatizer()
    for node in parsedXml.getroot(): 
        _id = node.attrib.get('Id')
        tags = node.attrib.get('Tags')
        title = node.attrib.get('Title')
        document = re.sub(r'\W',' ', title)
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        visas = str(int('visas' in tags))
        usa = str(int('usa' in tags))
        uk = str(int('uk' in tags))
        airTravel = str(int('air-travel' in tags))
        schengen = str(int('schengen' in tags))
        customsAndImmigration = str(int('customs-and-immigration' in tags))
        transit = str(int('transit' in tags))
        passports = str(int('passports' in tags))
        indianCitizens = str(int('indian-citizens' in tags))
        trains = str(int('trains' in tags))
        data = data.append(pd.Series([(_id), (document),(visas),(usa),(uk),(airTravel),(schengen),(customsAndImmigration),(transit),(passports),(indianCitizens),(trains)], index = dfcols), ignore_index = True)
    
    return data, True
    
def supervisedLearning(topTags,data,mix):
    
    if(mix == True):
        data, mix = preprocessSupervisedTrainingData(topTags)

    categories = topTags
    train, test = train_test_split(data, random_state=42, test_size=0.3, shuffle=mix)
    X_train = train.title
    X_test = test.title
    NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
    for category in categories:
        print('... Processing {}'.format(category))
        # train the model using X_dtm & y
        NB_pipeline.fit(X_train, train[category])
        # compute the testing accuracy
        prediction = NB_pipeline.predict(X_test)
        print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))
    
def preprocessSemiSupervisedTrainingData(topTags):
    parsedXml = etree.parse("PostsCleaned.xml")
    dfcols = ['id', 'title']
    dfcols = dfcols + topTags
    data = pd.DataFrame(columns=dfcols)
    stemmer = WordNetLemmatizer()
    number = 0
    for node in parsedXml.getroot(): 
        number = number + 1
        _id = node.attrib.get('Id')
        tags = node.attrib.get('Tags')
        title = node.attrib.get('Title')
        document = re.sub(r'\W',' ', title)
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = re.sub(r'^b\s+', '', document)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        if(number < 850):
            visas = str(int('visas' in tags))
            usa = str(int('usa' in tags))
            uk = str(int('uk' in tags))
            airTravel = str(int('air-travel' in tags))
            schengen = str(int('schengen' in tags))
            customsAndImmigration = str(int('customs-and-immigration' in tags))
            transit = str(int('transit' in tags))
            passports = str(int('passports' in tags))
            indianCitizens = str(int('indian-citizens' in tags))
            trains = str(int('trains' in tags))
        else:
            visas = '0'
            usa = '0'
            uk = '0'
            airTravel = '0'
            schengen = '0'
            customsAndImmigration = '0'
            transit = '0'
            passports = '0'
            indianCitizens = '0'
            trains = '0'
        data = data.append(pd.Series([(_id), (document),(visas),(usa),(uk),(airTravel),(schengen),(customsAndImmigration),(transit),(passports),(indianCitizens),(trains)], index = dfcols), ignore_index = True)
    
    return data

def semiSupervisedLearning(topTags):

    data = preprocessSemiSupervisedTrainingData(topTags)
    categories = topTags
    train = data[:850]
    test = data[850:]
    
    X_train = train.title
    X_test = test.title
    SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

    for category in categories:
        SVC_pipeline.fit(X_train, train[category])
        prediction = SVC_pipeline.predict(X_test)
        for index, row in enumerate(test.iterrows()):
            row[1][category] = prediction[index]

    data[:17000] = pd.concat([test,train])     

    supervisedLearning(topTags,data,False)
    
def preprocessUnsupervisedTrainingData(topTags):
    parsedXml = etree.parse("PostsCleaned.xml")
    dfcols = ['id', 'title']
    data = pd.DataFrame(columns=dfcols)
    stemmer = WordNetLemmatizer()
    tagList = []
    for node in parsedXml.getroot(): 
        _id = node.attrib.get('Id')
        tags = node.attrib.get('Tags')
        tagList.append(tags.split(','))
        title = node.attrib.get('Title')
        document = re.sub(r'\W',' ', title)
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        data = data.append(pd.Series([(_id), (document)], index = dfcols), ignore_index = True)
    return data, tagList

def unsupervisedLearning(topTags):
    
    data, tagList = preprocessUnsupervisedTrainingData(topTags)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data[:17000]['title'])
    modelkmeans = KMeans(n_clusters=10, init='k-means++', max_iter=400, n_init=10)
    modelkmeans.fit(X)

    Test = vectorizer.transform(data[17000:]['title'])
    tagList_cleaned = tagList[17000:]
    predicted_labels_kmeans = modelkmeans.predict(Test)
    number = 0
    for i in range(len(predicted_labels_kmeans)):
        
        if(topTags[predicted_labels_kmeans[i]] in tagList_cleaned[i]):
            number = number + 1
    print("Test accuracy is {}".format(number/len(predicted_labels_kmeans)))
    #print('Test accuracy is {}'.format(accuracy_score(, predicted_labels_kmeans)))

def main():

    topTags = createDistributionPlot()
    cleanDataset(topTags)
    data = pd.DataFrame()
    print("Supervised learning:")
    supervisedLearning(topTags, data, True)   
    print("Semi-supervised learning:") 
    semiSupervisedLearning(topTags)
    print("Unsupervised learning:")
    unsupervisedLearning(topTags)

main()