from sklearn.preprocessing import StandardScaler
import pickle

path_model =  ''
path_csv = ''


def  load_data(file_path ):
    return  pd.read_csv(file_path)

def get_main_feature():
    data  = load_data(path_csv)
    list_features = ["MEAN_RR", "MEDIAN_RR", "SDRR","RMSSD","SDSD","SDRR_RMSSD"
                 ,"HR","pNN50","SD1","SD2","VLF","LF","LF_NU","HF","HF_NU"
                 ,"TP","LF_HF","HF_LF",'condition']

    new_data  =  data[list_features]
    scaler = StandardScaler()
    label  = data['condition']
    feature =  new_data.drop(['condition'], axis =1)
    scaler.fit(feature)
    return scaler


MODEL  = pickle.load(open(path_model, 'rb'))
print("MODEL LOADED !")

SCALER =  get_main_feature()
print ('FEATURE FITED !')

print('MODEL READY FOR PREDICT !')