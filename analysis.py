import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data = pd.read_csv('DJIAnewsedit.csv')
train = pd.read_csv('DJIAnewstrain.csv')  # read train dataset
test = pd.read_csv('DJIAnewstest.csv')    # read test dataset
#print test
#print data.shape
#print train.shape
#print test.shape


from collections import Counter
import string

def get_tokens():
   with open('DJIAnewsedit.csv', 'r') as shakes:
    text = shakes.read()
    lowers = text.lower()
#remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

tokens = get_tokens()
#print tokens
count = Counter(tokens)
#print count.most_common(50)




from nltk.corpus import stopwords              #remove english stop words

tokens = get_tokens()
filtered = [w for w in tokens if not w in stopwords.words('english')]
count = Counter(filtered)
#print count.most_common(20)


import nltk
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer


token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


batch_size = 100
nb_classes = 10


trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:3]))



# top 10000 frequency words using TfidfVectorizer
advancedvectorizer = TfidfVectorizer(tokenizer=tokenize,max_features = 10000,stop_words='english')


advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print advancedtrain


testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:3]))
advancedtest = advancedvectorizer.transform(testheadlines)
#print(advancedtrain.shape)
X_train = advancedtrain.toarray()
X_test = advancedtest.toarray()
#print X_train



print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(train["Label"])
y_test = np.array(test["Labe"])

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)




input_dim = X_train.shape[1]


# Here's a MLP 
model = Sequential()

model.add(Dense(256, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))


model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# optimize with crossentropy loss function and rmsprop optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'] )

print("Training...")

#setting of epochs and batch size 

model.fit(X_train, Y_train, nb_epoch=100, batch_size=100,validation_data=(X_test, Y_test))

print("Generating test predictions...")
preds14 = model.predict_classes(X_test, verbose=0)
acc14 = accuracy_score(test["Labe"], preds14)
print preds14
print acc14



output = pd.DataFrame( data={"Date":test["Date"], "Label":preds14} )

# Use pandas to write the comma-separated output file
output.to_csv( "18marchout1.csv", index=False, quoting=3 )
