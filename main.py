# importing the libraries
import numpy as np
import pandas as pd
import re
import nltk
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Sequential
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Loading data.
test = pd.read_csv("data/test.txt", sep=";", names=["text", "sentiment"])
train = pd.read_csv("data/train.txt", ";", names=["text", "sentiment"])
val = pd.read_csv("data/val.txt", ";", names=["text", "sentiment"])

# Pre-processing funciton.


def data_cleaning(data):

    data = data.lower()
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"i'll", "i will", data)
    data = re.sub(r"'ll", "will", data)
    data = re.sub(r"have't", "have not", data)
    data = re.sub(r"she's", "she is", data)
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"'d", " would", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"what's", "what is", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"'ve", " have", data)
    data = re.sub(r"'re", " are", data)
    data = re.sub(r"'d", " would", data)
    data = re.sub(r"'ve", " have", data)
    data = re.sub(r"won't", "will not", data)
    data = re.sub(r"don't", "do not", data)
    data = re.sub(r"did't", "did not", data)
    data = re.sub(r"can't", "can not", data)
    data = re.sub(r"it's", "it is", data)
    data = re.sub(r"couldn't", "could not", data)
    data = re.sub("[.,;:'\"/)(]", "", data)
    data = word_tokenize(data)
    data = [re.sub("[*!@#$%^&*()_+{\};'\"\`><?]*", "", x) for x in data]
    stop = stopwords.words("english")
    data = [x for x in data if x not in stop]
    data = nltk.pos_tag(data)
    data = " ".join([x[0] + " " + x[1] for x in data])

    return data


# Cleaning the datasets.
train["text"] = train["text"].map(data_cleaning)
test["text"] = test["text"].map(data_cleaning)
val["text"] = val["text"].map(data_cleaning)

train_data = train["text"].tolist()
test_data = test["text"].tolist()
val_data = val["text"].tolist()

# Converting the labels to numbers for easy representation.
dict_feel = {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
train["sentiment"] = train["sentiment"].replace(
    {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
)
test["sentiment"] = test["sentiment"].replace(
    {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
)
val["sentiment"] = val["sentiment"].replace(
    {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
)

train_label = to_categorical(train["sentiment"])
test_label = to_categorical(test["sentiment"])
val_label = to_categorical(val["sentiment"])

# Creating tokenizer object with oov_token as "NA" rather than default zero.
tokenizer = Tokenizer(16000, oov_token="NA")
tokenizer.fit_on_texts(train_data)
train_data = tokenizer.texts_to_sequences(train_data)
test_data = tokenizer.texts_to_sequences(test_data)
val_data = tokenizer.texts_to_sequences(val_data)

# Making the length of sequences to fixed size.
train_data = pad_sequences(train_data, 71, padding="post")
test_data = pad_sequences(test_data, 71, padding="post")
val_data = pad_sequences(val_data, 71, padding="post")



""""
# Building CNN Configuration.
# And adding additional feature learned from CNN to the testing data.

model = Sequential()
model.add(Embedding(16000, 64, input_length=81))
model.add(Dropout(0.5))
model.add(Conv1D(10, kernel_size=3, activation="relu"))
model.add(MaxPooling1D())
model.add(Conv1D(10, kernel_size=3, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(6, activation="sigmoid"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_data, train_label, epochs=10, validation_data=(val_data, val_label))


# Predicting additional feature using CNN and adding it to test data.
temp = model.predict(test_data)
temp1 = [np.argmax(x) for x in temp]
pred_cnn = np.array(temp1)
test_data = test_data.tolist()
for i in range(len(pred_cnn)):
    test_data[i] = list(test_data[i])
for i in pred_cnn:
    count = 0
    test_data[count][-1] = i
    count += 1
test_data = np.array(test_data)
# Adding feature to test data finished
"""

# Building LSTM model.
model = Sequential()
model.add(Embedding(16000, 64, input_length=71))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Dense(6, activation="softmax"))
checkpoint = ModelCheckpoint(
    "sentiment_analysis.h5", monitor="accuracy", verbose=1, save_best_only=True
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

stat = model.fit(
    train_data,
    train_label,
    epochs=15,
    validation_data=(val_data, val_label),
    callbacks=[checkpoint],
)


model.evaluate(test_data, test_label)
"""
# Saving the trained model to a pickle file

# open a file, where you ant to store the data
file = open('model', 'wb')

# dump information to that file
pickle.dump(model, file)

# close the file
file.close()
# open a file, where you stored the pickled data
file = open('model', 'rb')

# dump information to that file
loaded_model = pickle.load(file)

# close the file
file.close()
"""


# Predicting the sentiment using user input

def Predict_Next_Words(model, tokenizer, text):
    text = " ".join(text)
    text = data_cleaning(text)
    text = np.array(text)
    text = text.tolist()
    sequence = tokenizer.texts_to_sequences([text])
    sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, 71, padding="post"
    )
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in dict_feel.items():
        if value == preds:
            predicted_word = key
            break

    print(predicted_word)
    return predicted_word


while True:
    text = input("Enter your line(0 to exit): ")

    if text == "0":
        print("Execution finished")
        break

    else:
        try:
            text = text.split(" ")
            text = text[-69:]
            print(text)

            Predict_Next_Words(model, tokenizer, text)

        except Exception as e:
            print("Error occurred: ", e)
            continue
