import re
import numpy as np
import nltk

###Preprocess tweets
def processTweet2(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

###get stopword list
def getStopWordList():
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    # fp = open(stopWordListFileName, 'r')
    # line = fp.readline()
    # while line:
    #     word = line.strip()
    #     stopWords.append(word)
    #     line = fp.readline()
    # fp.close()
    return stopWords

# stopWords = []

# st = open('stopwords.txt', 'r')
stopWords = getStopWordList()


def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

def getFeatureVector(tweet):
    featureVector = []
    # split tweet into words
    words = tweet.split()
    for w in words:
        # replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if the word starts with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        # ignore if it is a stop word
        if (w in stopWords): # or val is None
            continue
        else:
            featureVector.append(w.lower())
    return featureVector




def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


# end


if __name__=="__main__":
    ###load sentiment training data

    traindata = np.loadtxt('train.csv',dtype=str,comments='#',delimiter=',')
    tweets = []
    featureList = []
    traindata = np.delete(traindata, 0, 0)
    for i in range(len(traindata)):
        sentiment = traindata[i][1]
        tweet = traindata[i][2]
        processedTweet = processTweet2(tweet)
        featureVector = getFeatureVector(processedTweet)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))
        # print tweet
        # print (featureVector, sentiment)
    ### Remove featureList duplicates
    featureList = list(set(featureList))
    print featureList


    ######## NOT TESTED #########
    training_set = nltk.classify.util.apply_features(extract_features, tweets)

    # # Train the classifier Naive Bayes Classifier
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    ## Test
    testdata = np.loadtxt('test.csv', dtype=str, comments='#', delimiter=',')
    predicted_sentiments = testdata[:][2].apply(lambda tweet: NBClassifier.classify(extract_features(getFeatureVector(processTweet2(tweet)))))
     