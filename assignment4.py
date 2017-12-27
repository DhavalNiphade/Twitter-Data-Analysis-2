'''Assignment 4 (Version 1.1)

Please add your code where indicated. You may conduct a superficial test of
your code by executing this file in a python interpreter.

The documentation strings ("docstrings") for each function tells you what the
function should accomplish. If docstrings are unfamiliar to you, consult the
Python Tutorial section on "Documentation Strings".

This assignment requires the following packages:

- numpy
- pandas
- requests

All these packages should be installed if you are using Anaconda.

'''

import os
import pandas as pd
import requests
import tweepy
from sklearn.feature_extraction import text
from sklearn.naive_bayes import GaussianNB

TWITTER_DATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'united-states-congress-house-twitter-2016-grouped-tweets-train.csv')
TWITTER_DATA_FILENAME_TEST = os.path.join(os.path.dirname(__file__), 'data', 'united-states-congress-house-twitter-2016-grouped-tweets-test.csv')
WIKIPEDIA_API_ENDPOINT = 'https://en.wikipedia.org/w/api.php'

ACCESS_TOKEN = "1220909827-X6n5MabisHf7a1hktJ93JPLOwkenN2TU5Nisu8G"
ACCESS_TOKEN_SECRET = "F0fieS5oixKZ3W97FDhV38RhR9PYgoXFpr5yGH4mSa0ku"
CONSUMER_KEY= "mSi6hbX1lRZiWkN3cGUjaGtOR"
CONSUMER_SECRET = "fXZHAzs9j1QzoLuOFSnE7mCdAIzifBgCNUmGOUfzt6SWnAiNe9"

def count_user_mentions(tweet_list):
    """Count the number of user mentions (aka at-mentions, @-mentions) in each tweet in a list.

    This function operates on a list of tweets rather than one tweet at a time.
    This exercise is intended to provide more practice with lists, functions,
    and regular expressions.

    Arguments:
        tweet_list (list of str): List of Tweets.

    Returns:
        list of int: List of user mention counts.

    """
    # YOUR CODE HERE
    # print(tweet_list)
    count=0
    retval=[]
    for tweets in tweet_list:
        for tokens in tweets.strip().split():
            if '@' in tokens:
                count+=1
        retval.append(count)
        count=0
    return retval

def page_ids(titles):
    """Look up the Wikipedia page ids by title.

    For example, the Wikipedia page id of "Albert Einstein" is 736. (This page
    has a small page id because it is one of the very first pages on
    Wikipedia.)

    A useful reference for the Mediawiki API is this page:
    https://www.mediawiki.org/wiki/API:Info

    Arguments:
        titles (list of str): List of Wikipedia page titles.

    Returns:
        list of int: List of Wikipedia page ids.

    """
    # The following lines of code (before `YOUR CODE HERE`) are suggestions
    params = {
        'action': 'query',
        'prop': 'info',
        'titles': '|'.join(titles),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
    }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    # YOUR CODE HERE
    # print(payload)
    retval = []
    for title in titles:
        # retval.append(payload['query']['pages'][0]['pageid']) - Default case for one
        temp = payload['query']['pages']
        for val in temp:
            retval.append(val['pageid'])
    # print(retval)
    return retval

def page_lengths(ids):
    """Find the length of a page according to the Mediawiki API.

    A page's length is measured in bytes which, for English-language pages, is
    approximately equal to the number of characters in the page's source.

    A useful reference for the Mediawiki API is this page:
    https://www.mediawiki.org/wiki/API:Info

    Arguments:
        ids (list of str): List of Wikipedia page ids.

    Returns:
        list of int: List of page lengths.

    """
    # YOUR CODE HERE
    params = {
        'action': 'query',
        'prop': 'info',
        'pageids': "|".join(map(str,[id for id in ids])),
        'format': 'json',
        'formatversion': 2,  # version 2 is easier to work with
    }
    payload = requests.get(WIKIPEDIA_API_ENDPOINT, params=params).json()
    # print(payload)
    retval = []
    for id in ids:
        temp = payload['query']['pages']
        for pages in temp:
            retval.append(pages['length'])

    # print(retval)
    return retval


def tweet_text_by_id(id, consumer_key=None, consumer_secret=None, access_token=None, access_token_secret=None):
    """Get the text of a tweet by its id.

    You may assume that valid credentials (``consumer_key``,
    ``consumer_secret``, ``access_token``, ``access_token_secret``) are passed.
    You are, however, free to ignore them and retrieve the full text of the
    tweet by some other means. It is possible to retrieve a tweet without using
    the API. You could parse the HTML of a normal HTTP response, for instance.

    Feel free to use ``requests_oauth``. You may assume it is installed.

    Arguments:
        id (int): Tweet id.
        consumer_key (str): Twitter API Consumer Key
        consumer_secret (str): Twitter API Consumer Secret
        access_token (str): Twitter API Access Token
        access_token_secret (str): Twitter API Access Token Secret

    Returns:
        str: The text of the specified tweet.

    """
    # YOUR CODE HERE
    auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    tweet = api.get_status(id)
    # print(tweet.text)
    return tweet.text  #COMMENTING THIS IN ORDER TO GET THE TEST CASE RUNNING. UPON USING AN APPROPRIATE KEY AND TOKEN, THE FRAMEWORK WILL WORK



# additional practice with two-mode networks / bipartite graphs

def incidence_matrix_from_user_mentions(user_mentions_list):
    """Construct an incidence matrix from user mentions.

    Given user mentions (aka "@-mentions" or "at-mentions") associated with users,
    `user_mentions_list`, construct an incidence matrix.

    Recall from Newman 6.6 that an incidence matrix is the equivalent of an
    adjacency matrix for a bipartite network. It's a rectangular matrix with
    shape `g` x `n`, where `g` is the number of groups and `n` is the the
    number of participants in the network. Here `g` is the number of unique
    targets of user mentions and `n` is the number of elements (lists) in the
    list, `user_mentions_list`.

    An incidence matrix is different from an adjacency matrix. For example,
    suppose `user_mentions_list` is a list of the following lists:

    - [@nytimes]
    - [@nytimes, @washtimes]
    - [@foxandfriends]
    - [@foxandfriends]

    One would expect as a result the following incidence matrix:

        [[ 0.,  0.,  1.,  1.],
         [ 1.,  1.,  0.,  0.],
         [ 0.,  1.,  0.,  0.]]

    (The first row corresponds to '@foxandfriends', the second row corresponds
    to '@nytimes', and the third row corresponds to '@washtimes'.)

    This exercise should be easier than ``mentions_adjacency_matrix`` (from a
    previous assignment).  If you encountered difficulty with that problem,
    give this one a try.

    Arguments:

        user_mentions_list (list of list of str): List of user mentions

    Returns:
        array: Incidence matrix. Groups should be sorted name.

    """
    # YOUR CODE HERE
    n = len(user_mentions_list)
    uniqueMentions = set([items for mentionLists in user_mentions_list for items in mentionLists])
    g = len(set([items for mentionLists in user_mentions_list for items in mentionLists]))
    uniqueMentions = sorted(uniqueMentions)
    # print(set([items for mentionLists in user_mentions_list for items in mentionLists]))
    # print(n,g)
    retval = [[0 for j in range(n)] for i in range(g)]
    # print(retval)
    for i,mentions in enumerate(uniqueMentions):
        for j,lists in enumerate(user_mentions_list):
            if mentions in lists:
                retval[i][j]=1
            else:
                retval[i][j]=0

    # print(retval)
    return np.array(retval)

# challenging problems


# `load_twitter_corpus` is a data loading function; do not modify
def load_twitter_corpus():
    """Load US Congress Twitter corpus.

    Returns a pandas DataFrame with the following columns:

    - ``screen_name`` Member of Congress' Twitter handle (unused)
    - ``party`` Party affiliation. 0 is Democratic, 1 is Republican
    - ``tweets`` Text of five tweets concatenated together.

    Each record contains multiple tweets connected by a space. Grouping tweets
    together is not strictly necessary.

    Returns:

        pandas.DataFrame: DataFrame with tweets.

    """
    # data loading function; do not modify
    return pd.read_csv(TWITTER_DATA_FILENAME)


def predict_party(tweet, twitter_corpus):
    """Predict the party affiliation given the text of a tweet.

    ``tweet`` may be a group of tweets concatenated together. ``twitter_corpus``
    is passed as an argument to avoid having to load it over and over again
    when `predict_party` is called repeatedly.

    The precise strategy used for prediction is left entirely up to you.
    Nearest neighbors is a perfectly valid strategy. You might also consider
    the @-mentions in the tweet as a possible input to your model.

    Since members of US Congress tweet in very distinctive ways depending on
    party affiliation, a predictive model might be expected to achieve
    out-of-sample accuracy as high as 90% or even 95%. Not all classification
    tasks will be this easy.

    Arguments:

        tweet (str): a tweet (or more than one tweet) from a member of Congress as a Python string.
        twitter_corpus (pandas.DataFrame): DataFrame returned by ``load_twitter_corpus``.

    Returns:
        int: 0 is Democratic, 1 is Republican.

    """

    inputTweets = []
    if type(tweet) is list:
        for elems in tweet:
            inputTweets.append(elems)
    else:
        inputTweets.append(tweet)


    # YOUR CODE HERE
    listOfClasses = [c for c in twitter_corpus.party]

    # Handle training dataset
    vec = text.CountVectorizer(min_df=2)
    dtm = vec.fit_transform(twitter_corpus['tweets']).toarray()
    # vocab = vec.get_feature_names()
    # dtm = pd.DataFrame(dtm,index=twitter_corpus.index,columns=vocab)
    # featureVectors = dtm.values
    listOfClasses = np.array(listOfClasses)

    # Handle the single tweet
    inputArray = pd.DataFrame(inputTweets)
    newVectors = vec.transform(inputArray[0]).toarray()


    # Handle testing dataset - FOR THE TEST.CSV FILE
    ''' testDF = pd.read_csv(TWITTER_DATA_FILENAME)
    inputArray = vec.fit_transform(testDF['tweets']).toarray()
    vocab = vec.get_feature_names()
    inputArray = pd.DataFrame(dtm, index=testDF.index, columns=vocab)
    inputArray = inputArray.values '''

    # Naive Bayesian Classifier
    model = GaussianNB()
    model.fit(dtm,listOfClasses)
    predicted = list(model.predict(newVectors))
    return predicted if len(predicted)>1 else predicted[0]

def predict_party_proba(tweet, twitter_corpus):
    """Predict the probability of Republican party affiliation given the text of a tweet.

    See ``predict_party`` for details. This function differs from ``predict_party`` in
    that it returns the probability of a tweet being from a Republican member of congress.

    Arguments:

        tweet (str): a tweet (or more than one tweet) from a member of Congress as a Python string.
        twitter_corpus (pandas.DataFrame): DataFrame returned by ``load_twitter_corpus``.

    Returns:
        float: probability between 0 and 1 of tweet being authored by a Republican.

    """
    # YOUR CODE HERE

    inputTweets = []
    if type(tweet) is list:
        for elems in tweet:
            inputTweets.append(elems)
    else:
        inputTweets.append(tweet)

    # Build set of classes
    listOfClasses = [c for c in twitter_corpus.party]

    # Handle training dataset
    vec = text.CountVectorizer(min_df=15)
    dtm = vec.fit_transform(twitter_corpus['tweets']).toarray()
    # vocab = vec.get_feature_names()
    # dtm = pd.DataFrame(dtm, index=twitter_corpus.index, columns=vocab)
    # featureVectors = dtm.values
    listOfClasses = np.array(listOfClasses)


    # Handle Single Tweet
    inputArray = pd.DataFrame(inputTweets)
    newVectors = vec.transform(inputArray[0]).toarray()

    '''
    # Handle testing dataset
    testDF = pd.read_csv(TWITTER_DATA_FILENAME)
    inputArray = vec.fit_transform(testDF['tweets']).toarray()
    vocab = vec.get_feature_names()
    inputArray = pd.DataFrame(dtm, index=testDF.index, columns=vocab)
    inputArray = inputArray.values
    '''

    # Naive Bayesian Classifier
    model = GaussianNB()
    model.fit(dtm, listOfClasses)
    predicted = list(model.predict_proba(newVectors))

    return predicted if len(predicted)>1 else predicted[0][0]

# DO NOT EDIT CODE BELOW THIS LINE

import unittest

import numpy as np


class TestAssignment4(unittest.TestCase):

    def test_count_user_mentions1(self):
        tweets = [
          """RT @HouseGOP: The #StateOfTheUnion isn't strong for the 8.7 million Americans out of work.""",
          """@HouseGOP Good morning.""",
        ]
        self.assertEqual(count_user_mentions(tweets), [1, 1])


    def test_page_ids1(self):
        titles = ['Albert Einstein']
        ids = [736]
        self.assertEqual(page_ids(titles), ids)


    def test_page_lengths1(self):
        ids = [736]
        expected = [137000]  # NOTE: this number changes over time
        lengths = page_lengths(ids)
        self.assertEqual(len(lengths), 1)
        self.assertGreater(lengths[0], 0)
        self.assertGreater(lengths[0], expected[0] >> 3)
        self.assertLess(lengths[0], expected[0] << 3)


    def test_tweet_text_by_id1(self):
        id = 685423981671936001
        expected = """Getting ready to go live"""
        text = tweet_text_by_id(
            id,
            consumer_key=os.environ.get('CONSUMER_KEY'),
            consumer_secret=os.environ.get('CONSUMER_SECRET'),
            access_token=os.environ.get('ACCESS_TOKEN'),
            access_token_secret=os.environ.get('ACCESS_TOKEN_SECRET'),
        )
        # NOTE: skipping this test here since it will fail without
        # the appropriate values. If you understand what is happening
        # here, feel free to uncomment the following line:
        # self.assertEqual(text[:len(expected)], expected)


    def test_incidence_matrix_from_user_mentions1(self):
        user_mentions = [
            ['@nytimes'],
            ['@nytimes', '@washtimes'],
            ['@foxandfriends'],
            ['@foxandfriends'],
        ]
        B = incidence_matrix_from_user_mentions(user_mentions)
        self.assertEqual(B.shape, (3, 4))


    def test_load_twitter_corpus1(self):
        df = load_twitter_corpus()
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 100)


    def test_predict_party1(self):
        df = load_twitter_corpus()
        party = predict_party("""RT @HouseGOP: The #StateOfTheUnion is strong.""", df)
        self.assertIn(party, {0, 1})


    def test_predict_party_proba1(self):
        df = load_twitter_corpus()
        party_proba = predict_party_proba("""RT @HouseGOP: The #StateOfTheUnion is strong.""", df)
        self.assertGreaterEqual(party_proba, 0)
        self.assertLessEqual(party_proba, 1)


if __name__ == '__main__':
    unittest.main()
