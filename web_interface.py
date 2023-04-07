import numpy as np
import re
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from flask import Flask, request, render_template
from tensorflow import keras
# import pandas as pd
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import Sequential
# from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dropout
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Loading data.
# test = pd.read_csv("data/test.txt", sep=";", names=["text", "sentiment"])
# train = pd.read_csv("data/train.txt", ";", names=["text", "sentiment"])
# val = pd.read_csv("data/val.txt", ";", names=["text", "sentiment"])


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


# # Cleaning the datasets.
# train["text"] = train["text"].map(data_cleaning)
# test["text"] = test["text"].map(data_cleaning)
# val["text"] = val["text"].map(data_cleaning)

# train_data = train["text"].tolist()
# test_data = test["text"].tolist()
# val_data = val["text"].tolist()

# # Converting the labels to numbers for easy representation.
# dict_feel = {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
# train["sentiment"] = train["sentiment"].replace(
#     {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
# )
# test["sentiment"] = test["sentiment"].replace(
#     {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
# )
# val["sentiment"] = val["sentiment"].replace(
#     {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}
# )

# train_label = to_categorical(train["sentiment"])
# test_label = to_categorical(test["sentiment"])
# val_label = to_categorical(val["sentiment"])

# # Creating tokenizer object with oov_token as "NA" rather than default zero.
# tokenizer = Tokenizer(16000, oov_token="NA")
# tokenizer.fit_on_texts(train_data)
# train_data = tokenizer.texts_to_sequences(train_data)
# test_data = tokenizer.texts_to_sequences(test_data)
# val_data = tokenizer.texts_to_sequences(val_data)

# # Making the length of sequences to fixed size.
# train_data = pad_sequences(train_data, 71, padding="post")
# test_data = pad_sequences(test_data, 71, padding="post")
# val_data = pad_sequences(val_data, 71, padding="post")


# # Building LSTM model.
# model = Sequential()
# model.add(Embedding(16000, 64, input_length=71))
# model.add(Dropout(0.3))
# model.add(Bidirectional(LSTM(128, return_sequences=True)))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(0.3))
# model.add(Dense(128))
# model.add(Dense(6, activation="softmax"))
# checkpoint = ModelCheckpoint(
#     "sentiment_analysis.h5", monitor="accuracy", verbose=1, save_best_only=True
# )
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# print(model.summary())

# stat = model.fit(
#     train_data,
#     train_label,
#     epochs=15,
#     validation_data=(val_data, val_label),
#     callbacks=[checkpoint],
# )


# model.evaluate(test_data, test_label)


# # 1st type of saving the model
# # # Saving the trained model to a pickle file

# # open a file, where you ant to store the data
# file = open('model.pkl', 'wb')

# # dump information to that file
# pickle.dump(model, file)

# # close the file
# file.close()
# # end of 1st type of model saving

# # 2nd type of saving the model
# model.save('C:\StFx_Courses_Data\\PBDAI_3rdSem\\Machine_Learning\\project\\NLP\\project\\')
model = keras.models.load_model('C:\StFx_Courses_Data\\PBDAI_3rdSem\\Machine_Learning\\project\\NLP\\project\\')
# # end of 2nd type of modelsaving


# # 3rd type of saving:
# # saving using JSON nad weights:
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# # later...

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
# # end of 3rd type of model saving


tokenizer = pickle.load(open('tokenizer', 'rb'))

dict_feel = {"joy": 0, "love": 1, "anger": 2, "sadness": 3, "surprise": 4, "fear": 5}


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


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text = request.form["text"]
    text = text.split(" ")
    text = text[-69:]

    output = Predict_Next_Words(model, tokenizer, text)

    return render_template('index.html', prediction_text='The sentence feeling infer: $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=False)