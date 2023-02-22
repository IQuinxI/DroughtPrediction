import pandas as pd
from sklearn import preprocessing, tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from scipy import stats
import numpy as np
import matplotlib as plt
from  joblib  import dump, load
from tqdm import tqdm
import dask.dataframe as dd
import os 
# df = pd.read_csv('test_timeseries.csv', nrows=1000,usecols=lambda column: column not in ["date"])
# df = pd.read_csv('train_timeseries.csv', nrows=10000, usecols=lambda column: column not in ["date"])
# print(df)
feature_names = ['fips','PRECTOT','PS','QV2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN','T2M_RANGE','TS','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE','WS50M','WS50M_MAX','WS50M_MIN','WS50M_RANGE', 'score']
# df = pd.DataFrame(train)

def Processor(data):
    # data = pd.DataFrame(inp)
    print("removing duplicates")
    data = data.drop_duplicates()
    print("finished removing duplicates")

    # ========= replaces empty data slots in the database by the mean =========
    print("Imputing data")
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    print("finished Imputing data")

    # ========= removing outliers with a z score above 3 =========
    # z =  np.abs(stats.zscore(data))
    # data = data[(z<3).all(axis=1)]

    # ========= Normalize data between -1 and 1 =========
    
    
    max_abs_scaler = preprocessing.MaxAbsScaler()
    

    print("saving scaler")
    dump(max_abs_scaler, 'scaler.joblib')
    print("finished saving scaler")

    print("Normalizing data")
    data = max_abs_scaler.fit_transform(data)
    print("Finished Normalizing data")
    # print(len(data))
    return pd.DataFrame(data, columns=feature_names)

def DecisionTree(inp):
    X_train = inp.drop(columns="score").copy()
    Y_train = inp["score"]
    
    print(X_train)
    print(Y_train)

    
    clf = tree.DecisionTreeRegressor()

    # pipe = Pipeline(steps=[('dec_tree', clf)])
    # criterion = ['squared_error']
    # # max_depth = [9000000,15000000,19000000]
    # # 19000000, 
    # max_depth = [19300680, 19300000, 19000000]

    # parameters = dict(dec_tree__criterion=criterion,
    #                     dec_tree__max_depth=max_depth)
    

    # clf = GridSearchCV(pipe, parameters)

    print("Fitting ...")
    clf.fit(X_train, Y_train)

    # print('Best Criterion:', clf.best_estimator_.get_params()['dec_tree__criterion'])
    # print('Best max_depth:', clf.best_estimator_.get_params()['dec_tree__max_depth'])
    # clf.fit(X_train, Y_train)
    
    print("Dumping")
    dump(clf, 'DecisionTree.joblib')
    print("Finished dumping")
    
    return clf

def Predict(input):
    if not os.path.exists('./scaler.joblib'):
        print("Error: no scaler file was found")
        exit()
    elif not os.path.exists('./DecisionTree.joblib'):
        print("Error: no model file was found")
        exit()

    clf = load('DecisionTree.joblib')

    
    predict_input = pd.DataFrame(input, columns=feature_names[:len(feature_names)-1])

    scaler = load('scaler.joblib')              
    scaler.fit_transform(predict_input)
    original = scaler.inverse_transform(clf.predict(predict_input))
    print(original)

def SaveDataFrame(): 
    chunks = pd.DataFrame({})
    i = 1
    for chunk in tqdm(pd.read_csv('train_timeseries.csv', chunksize=10000, usecols=lambda column: column not in ["date"])):        
        # print(f"retrieving chunk {i}")        
        chunks=chunks.append(chunk)
        # chunks = pd.concat([chunks, chunk])
        i+=1
    print("Data Retrieved")    
    tqdm(dump(chunks, 'DataFrame.joblib'))
    return chunks
    # df = pd.DataFrame(chunks)
    # print("Initialised DataFrame")
    # 
    # 

def Train(df):
    data = Processor(df)
    DecisionTree(data)
# Train(df)


def PredictFunc():
    # df = pd.read_csv('train_timeseries.csv', nrows=1000000,usecols=lambda column: column not in ["date"])
    predict_input = [[1001,0.0,99.87,15.16,26.01,20.61,20.47,33.0,19.39,13.62,26.08,1.84,2.34,1.41,0.93,4.07,6.43,2.38,4.05]]
    
    Predict(predict_input)

def Train():
    print("Loading ...")
    df = load('DataFrame.joblib')
    print('DataFrame Loaded')

    # print(df.head())
    # print(feature_names)
    print('Processing')
    pdf = Processor(df)
    print('Finished processing')

    print("Training Decision tree")
    DecisionTree(pdf)
    print("finished training")

    # print("Number of cores:", os.cpu_count())
PredictFunc()