import pandas as pd
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score 


def train():
    df = pd.read_csv(r"D:\ozmenta\final\final\data\train.csv", index_col=0)
    print("readed file")
    corpus = []
    df = df.dropna()

    x = df.drop('label', axis=1)

    y=  df['label']
    voc_size = 5000
    sent_length = 20
    print("preprocess started")

    corpus = preprocess(x)

    print("preprocess ended, one hot started")
    one_hot_rep = [one_hot(words,voc_size)for words in corpus]
    embedded_docs=pad_sequences(one_hot_rep,padding='pre',maxlen=sent_length)
    X_final=np.array(embedded_docs)
    y_final=np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

    embedding_vector_features=40
    model=Sequential()
    model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
    model.add(LSTM(100))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
    y_pred = model.predict(X_test)
    pred = []
    for i in range(len(y_pred)):
        if (y_pred[i]) > 0.5:
            pred.append(1)
        else:
            pred.append(0)

    print(confusion_matrix(y_test, pred))
    print(accuracy_score(y_test, pred))

    model.save(r"model.hdf5")

def preprocess(m):
    ps = PorterStemmer()
    corpus = []
    messages = m.copy()
    messages.reset_index(inplace=True)

    for i in range(0, len(messages)):

        review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

if __name__ == "__main__":
    train()