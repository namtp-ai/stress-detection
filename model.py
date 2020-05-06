import pandas as pd
import numpy as np
import os
import sklearn.pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from firebase.firebase import FirebaseAuthentication,FirebaseApplication
import time
from hrvanalysis import get_frequency_domain_features,get_time_domain_features, get_poincare_plot_features

def root_directory():
    current_path = os.path.abspath(os.getcwd())
    return os.path.abspath(os.path.join(current_path, os.pardir))
def data_directory():
    return os.path.join(root_directory(), "hrv dataset/data")

def load_train_set():
    #Loading a hdf5 file is much much faster
    in_file = os.path.join(data_directory(), "final",  "train.csv")
    return pd.read_csv(in_file)
def load_test_set():
    #Loading a hdf5 file is much much faster
    in_file = os.path.join(data_directory(), "final",  "data_user.csv")
    return pd.read_csv(in_file)
def load_test(pipeline, hrv_features):
    test = load_test_set()
    X_test = test[hrv_features]
    X_test = scaler.transform(X_test)
    y_prediction = pipeline.predict(X_test)
    return y_prediction[-1]
def RR_to_features(RR_interval):
    feautures_1 = get_poincare_plot_features(RR_interval)
    SD1 = feautures_1['sd1']
    SD2 = feautures_1['sd2']
    feautures_2 = get_frequency_domain_features(RR_interval)
    LF = feautures_2['lf']
    HF = feautures_2['hf']
    LF_HF = feautures_2['lf_hf_ratio']
    HF_LF = 1/LF_HF
    LF_NU = feautures_2['lfnu']
    HF_NU = feautures_2['hfnu']
    TP = feautures_2['total_power']
    VLF = feautures_2['vlf']
    feautures_3 = get_time_domain_features(RR_interval)
    pNN50 = feautures_3['pnni_50']
    RMSSD = feautures_3['rmssd']
    MEAN_RR = feautures_3['mean_nni']
    MEDIAN_RR = feautures_3['median_nni']
    HR = feautures_3['mean_hr']
    SDRR = feautures_3['sdnn']
    SDRR_RMSSD = SDRR/RMSSD
    SDSD = feautures_3['sdsd']
    import csv
    row_list = [["MEAN_RR", "MEDIAN_RR", "SDRR","RMSSD","SDSD","SDRR_RMSSD"
                 ,"HR","pNN50","SD1","SD2","VLF","LF","LF_NU","HF","HF_NU"
                 ,"TP","LF_HF","HF_LF"],
             [MEAN_RR,MEDIAN_RR,SDRR,RMSSD,SDSD,SDRR_RMSSD,HR,pNN50,SD1,SD2
              ,VLF,LF,LF_NU,HF,HF_NU,TP,LF_HF,HF_LF]]
    with open('data/final/data_user.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)                    
    
def diagnose():
    app = FirebaseApplication('https://smartmachine-ed87d.firebaseio.com/')
    list_data =[]
    while True:
        heart_data = app.get('/heart_rate',None)
        if heart_data['10'] != 0:
            for i in range(10,0,-1):
                heart_data['%d'%i] = 60/heart_data['%d'%i]
                list_data.append(heart_data['%d'%i]*1000)
            heart_data = dict.fromkeys(heart_data, 0)
            print(list_data)
            app.put('/','heart_rate',heart_data)
            if(len(list_data)%30==0):
                stress = RR_to_features(list_data)
                list_data.clear()

if __name__ == '__main__':
    select = SelectKBest(k=10)
    train =load_train_set()
    
    target = 'condition'
    hrv_features = list(train)
    hrv_features = [x for x in hrv_features if x not in [target]]
    X_train= train[hrv_features]
    y_train= train[target]
    print("load data ok")
    classifiers = [
                    RandomForestClassifier(n_estimators=100, max_features='log2', n_jobs=-1),
                    SVC(C=20, kernel='rbf'),   
                 ]
    for clf in classifiers:
        name = str(clf).split('(')[0]
        if 'svc' == name.lower():
            # Normalize the attribute values to mean=0 and variance=1
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
        clf = RandomForestClassifier()
        steps = [('feature_selection', select),
             ('model', clf)]
        pipeline = sklearn.pipeline.Pipeline(steps)
        print("load fit data")
        pipeline.fit(X_train, y_train)
        print("load fit ok")

   