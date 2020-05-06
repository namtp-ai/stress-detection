import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import pickle


def  load_data(file_path ):
    return  pd.read_csv(file_path)

def get_main_feature(data):
    list_features = ["MEAN_RR", "MEDIAN_RR", "SDRR","RMSSD","SDSD","SDRR_RMSSD"
                 ,"HR","pNN50","SD1","SD2","VLF","LF","LF_NU","HF","HF_NU"
                 ,"TP","LF_HF","HF_LF",'condition']

    new_data  =  data[list_features]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    label  = data['condition']
    feature =  new_data.drop(['condition'], axis =1)
    scaler.fit(feature)
    feature = scaler.transform(feature)
    X_train, X_test, y_train, y_test = train_test_split( feature, label, test_size=0.2, random_state=100)
    print('DONE GET FEATURES')
    return X_train, X_test, y_train, y_test 

def evaluation_model(X_train,y_train):
    models = [
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
        SVC(C=20, kernel='rbf'),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    print("DONE LOAD MODEL")
    entries = []
    for model in models:
        print("Processing model", model)
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
            print('Fold_idx',fold_idx, '  Accuracy:', accuracy)

    mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
    std_accuracy = cv_df.groupby('model_name').accuracy.std()

    acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
            ignore_index=True)
    acc.columns = ['Mean Accuracy', 'Standard deviation']
    print(acc)


def make_model_machine(model, x_train,X_test , y_train, y_test, filename_model):
    model.fit(x_train , y_train)
    text_pre = model.predict(X_test)
    acc  =  accuracy_score(y_test ,text_pre )
    print("ACC :" , acc)
    pickle.dump(model, open(filename_model, 'wb'))
    print('DONE MODEL')

def predict(data, model_path):
    loaded_model = pickle.load(open(model_path, 'rb'))
    print("DONE load model")
    predict_label  =  loaded_model.predict(data)
    return predict_label


if __name__ == '__main__':
    data  = load_data('hrv_dataset/data/final/train.csv')
    X_train, X_test, y_train, y_test   =  get_main_feature(data)
    model  =  RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    make_model_machine(model,X_train, X_test, y_train, y_test , 'model.pkl')

    # evaluation_model(X_train,y_train)
    # text_pre =predict(X_test, 'model.pkl')
    # acc  =  accuracy_score(y_test ,text_pre )
    # print("ACC :" , acc)
    
    
