import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from sklearn import metrics
import pickle


def load_data(file_path):
    return pd.read_csv(file_path)


def get_main_feature(data):
    list_features = ["MEAN_RR", "MEDIAN_RR", "SDRR", "RMSSD", "SDSD", "SDRR_RMSSD", "HR", "pNN50",
                     "SD1", "SD2", "VLF", "LF", "LF_NU", "HF", "HF_NU", "TP", "LF_HF", "HF_LF", 'condition']

    new_data = data[list_features]
    label = data['condition']
    le = preprocessing.LabelEncoder()
    le.fit(label)
    train_labels = le.transform(label)
    train_labels = to_categorical(train_labels)

    feature = new_data.drop(['condition'], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        feature, train_labels, test_size=0.3, random_state=1000)
    print(np.shape(X_train))
    print('DONE GET FEATURES')
    return np.asarray(X_train), np.asarray(X_val), np.asarray(y_train), np.asarray(y_val)


def BaselineModel():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        64, input_dim=18, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    def myprint(s):
        with open('modelsummary.txt','w+') as f:
            print(s, file=f)
    plot_model(model, to_file='model.png')
    model.summary(print_fn =myprint)
    return  model


def train(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train, epochs=30, batch_size=128, validation_data=(X_val, y_val),verbose=1)
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save_weights("model.h5")
    print("Saved model to disk")


def predict(data, model_path):
    # loaded_model = pickle.load(open(model_path, 'rb'))
    # predict_label = loaded_model.predict(data)
    return predict_label


if __name__ == '__main__':
    data = load_data('hrv_dataset/data/final/train.csv')
    X_train, X_val, y_train, y_val = get_main_feature(data)
    model  =  BaselineModel()
    train(model, X_train, X_val, y_train, y_val)

    # evaluation_model(X_train,y_train)
    # predict(data, 'model.pkl')