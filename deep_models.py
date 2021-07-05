import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.metrics import roc_auc_score


def CreateTestTrain_Dense(data_std, Y, window):
    r,x = data_std.shape
    
    dat = np.zeros((r-window, window, x))
    

    for i in range(r-window):
        dat[i,:,:] = data_std.iloc[i:i+window,:].values



    train = dat[:14000, :,:]
    val = dat[14000:15500, :,:]
    test = dat[15500:, :,:]

    Y_train = Y[window:14005]
    Y_val = Y[14000 + window:15500 + window]
    Y_test = Y[15500 + window:]
    
    return train, val, test, Y_train, Y_val, Y_test
    
    


def Dense_Model_1(train,val,test,Y_train,Y_val, Y_test, EP, window, x):
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[window, x]),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])
    
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train, Y_train, epochs=EP, validation_data = (val,Y_val), verbose=0)
    
    return history, model


def Dense_Model_2(train,val,test,Y_train,Y_val, Y_test, EP, window, x):
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[window, x]),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train, Y_train, epochs=EP, validation_data = (val,Y_val), verbose=0)
    
    return history, model
    

def Dense_Model_3(train,val,test,Y_train,Y_val, Y_test, EP, window, x):
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[window, x]),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train, Y_train, epochs=EP, validation_data = (val,Y_val), verbose=0)
    
    return history, model


def Run_Dense(data_std, Y, name, window = 5, EP = 50):
    r,x = data_std.shape
    
    train, val, test, Y_train, Y_val, Y_test = CreateTestTrain_Dense(data_std, Y, window)
    
    His1, Mod1 = Dense_Model_1(train,val,test,Y_train,Y_val, Y_test, EP, window, x)
    His2, Mod2 = Dense_Model_2(train,val,test,Y_train,Y_val, Y_test, EP, window, x)
    His3, Mod3 = Dense_Model_3(train,val,test,Y_train,Y_val, Y_test, EP, window, x)
    
    
    return test, Y_test, His1, His2, His3, Mod1, Mod2, Mod3


#def plot(his):
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])    


def Eval(mod, test, Y_test):
    pred = np.argmax(mod.predict(test), axis=-1)
    n = pred.shape[0]

    acc = sum(pred == Y_test.iloc[:,0]) / n

    pred = mod.predict(test)
    ROC = roc_auc_score(pd.get_dummies(Y_test), pred, multi_class="ovr")

    print("Accuray = " + str(acc) + ", ROC = " + str(ROC))




def CreateData_LSTM(data_std, Y):
    r,x = data_std.shape
    data = data_std.values.reshape((1, r, x))
    Y_std = Y.values.reshape((1,r,1))
    
    train_lstm = data[:,:14000,:]
    #val_lstm = data[:,14000:15500,:]
    val_lstm = data[:,:15500,:]
    #test_lstm = data[:,15500:,:]
    test_lstm = data[:,:,:]
    
    Y_train_lstm = Y_std[:,:14000,:]
    #Y_val_lstm = Y_std[:,14000:15500,:]
    Y_val_lstm = Y_std[:,:15500,:]
    #Y_test_lstm = Y_std[:,15500:,:]
    Y_test_lstm = Y_std[:,:,:]


    return train_lstm, val_lstm, test_lstm, Y_train_lstm, Y_val_lstm, Y_test_lstm
    

def LSTM_Model_1(train_lstm, Y_train_lstm, val_lstm, Y_val_lstm, x, EP = 25):
    model = Sequential()
    model.add(LSTM(16, input_shape=(None, x), return_sequences = True))
    model.add(Dense(3, activation = "softmax"))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    history = model.fit(train_lstm, Y_train_lstm, epochs=25, validation_data = (val_lstm, Y_val_lstm), verbose = 0)


    return history, model



def Run_LSTM(data_std, Y, name, EP = 25):
    r,x = data_std.shape
    
    train_lstm, val_lstm, test_lstm, Y_train_lstm, Y_val_lstm, Y_test_lstm = CreateData_LSTM(data_std, Y)
    
    His1, Mod1 = LSTM_Model_1(train_lstm,Y_train_lstm,val_lstm,Y_val_lstm, x)
    
    
    return test_lstm, Y_test_lstm, His1, Mod1




def Eval_LSTM(mod, test_lstm, Y_test_lstm):
    pred = np.argmax(mod.predict(test_lstm), axis=-1)
    acc = sum(pred[0,15500:] == Y_test_lstm[0,15500:,0])/1544
    pred = mod.predict(test_lstm)
    ROC = roc_auc_score(pd.get_dummies(Y_test_lstm[:,15500:,:].reshape(1544)), pred[:,15500:,:].reshape((1544,3)), multi_class="ovr")

    print("Accuray = " + str(acc) + ", ROC = " + str(ROC))

    

