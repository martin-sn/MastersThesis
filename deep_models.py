import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import sklearn
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tabulate import tabulate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report




callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode = "min", restore_best_weights = True)



def CreateTestTrain_Dense(data_std, Y, window, val_split, test_split):
    r,x = data_std.shape
    
    dat = np.zeros((r-window, window, x))
    

    for i in range(r-window):
        dat[i,:,:] = data_std.iloc[i:i+window,:].values



    train = dat[:val_split, :,:]
    val = dat[val_split:test_split, :,:]
    test = dat[test_split:, :,:]

    Y_train = Y[window:val_split + window]
    Y_val = Y[val_split + window:test_split + window]
    Y_test = Y[test_split + window:]
    
    return train, val, test, Y_train, Y_val, Y_test
    
    


def Dense_Model_1(train,val,test,Y_train,Y_val, Y_test, EP, window, x):


    a,b,c = train.shape

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[window, x]),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train, Y_train, epochs=EP, validation_data = (val,Y_val), verbose=0, batch_size = a, callbacks=[callback])
    
    return history, model


def Dense_Model_2(train,val,test,Y_train,Y_val, Y_test, EP, window, x):

    a,b,c = train.shape


    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[window, x]),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train, Y_train, epochs=EP, validation_data = (val,Y_val), verbose=0, batch_size = a, callbacks=[callback])
    
    return history, model
    

def Dense_Model_3(train,val,test,Y_train,Y_val, Y_test, EP, window, x):

    a,b,c = train.shape

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[window, x]),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
    ])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(train, Y_train, epochs=EP, validation_data = (val,Y_val), verbose=0, batch_size = a, callbacks=[callback])
    
    return history, model


def Run_Dense(data_std, Y, name, val_split, test_split, windows = [1,5,10], EP = 250):

    test_sets = []

    def SelectBest(His,Mod):


        if (min(His[1].history["val_loss"]) < min(His[2].history["val_loss"])) & (min(His[1].history["val_loss"]) < min(His[3].history["val_loss"])):
            BestHis = His[1]
            BestMod = Mod[1]
            print("Window of size " + str(windows[0]) + " was best") 


        if (min(His[2].history["val_loss"]) < min(His[1].history["val_loss"])) & (min(His[2].history["val_loss"]) < min(His[3].history["val_loss"])):
            BestHis = His[2]
            BestMod = Mod[2]
            print("Window of size " + str(windows[1]) + " was best") 


        if (min(His[3].history["val_loss"]) < min(His[1].history["val_loss"])) & (min(His[3].history["val_loss"]) < min(His[2].history["val_loss"])):
            BestHis = His[3]
            BestMod = Mod[3]
            print("Window of size " + str(windows[2]) + " was best") 

        return BestHis, BestMod



    r,x = data_std.shape
    

    His = ["x"]
    Mod = ["x"]

    test_sets = []
    y_sets = []

    for window in windows: 

        train, val, test, Y_train, Y_val, Y_test = CreateTestTrain_Dense(data_std, Y, window, val_split, test_split)

        test_sets.append(test)
        y_sets.append(Y_test)

        His1, Mod1 = Dense_Model_1(train,val,test,Y_train,Y_val, Y_test, EP, window, x)

        His.append(His1)
        Mod.append(Mod1)


    His1, Mod1 = SelectBest(His, Mod)

    His = ["x"]
    Mod = ["x"]

    for window in windows: 

        train, val, test, Y_train, Y_val, Y_test = CreateTestTrain_Dense(data_std, Y, window, val_split, test_split)


        His2, Mod2 = Dense_Model_2(train,val,test,Y_train,Y_val, Y_test, EP, window, x)

        His.append(His2)
        Mod.append(Mod2)


    His2, Mod2 = SelectBest(His, Mod)

    His = ["x"]
    Mod = ["x"]

    for window in windows: 

        train, val, test, Y_train, Y_val, Y_test = CreateTestTrain_Dense(data_std, Y, window, val_split, test_split)


        His3, Mod3 = Dense_Model_3(train,val,test,Y_train,Y_val, Y_test, EP, window, x)
        His.append(His3)
        Mod.append(Mod3)


    His3, Mod3 = SelectBest(His, Mod)


    
    return test_sets, y_sets, His1, His2, His3, Mod1, Mod2, Mod3


#def plot(his):
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])    



#def ConfMatrix(True,Pred):
#
#
 #   table = [['Pred Down', 'Pred Neutral', 'Pred Up'],['1 Name', 'Last Name', 'Age'], ['John', 'Smith', 39], ['Mary', 'Jane', 25], ['Jennifer', 'Doe', 28]]







def Eval(mod, test, Y_test):
    pred = np.argmax(mod.predict(test), axis=-1)
    n = pred.shape[0]

    #acc = sum(pred == Y_test.iloc[:,0]) / n
    acc = accuracy_score(Y_test.iloc[:,0], pred)

    # Sensitivity (Recall)

    True_zero = sum(Y_test.iloc[:,0] == 0) 
    True_one = sum(Y_test.iloc[:,0] == 1) 
    True_two = sum(Y_test.iloc[:,0] == 2) 

    Pred_zero = sum(pred == 0) 
    Pred_one = sum(pred == 1) 
    Pred_two = sum(pred == 2) 

    TP_one = sum((Y_test.iloc[:,0] == 1) & (pred == 1))
    TP_two = sum((Y_test.iloc[:,0] == 2) & (pred == 2))

    TN_one = sum((Y_test.iloc[:,0] != 1) & (pred != 1))
    TN_two = sum((Y_test.iloc[:,0] != 2) & (pred != 2))

    TPR_one = TP_one/True_one
    TPR_two = TP_two/True_two

    TNR_one = TN_one / (True_two + True_zero)
    TNR_two = TN_two / (True_one + True_zero)

    print("Class 1:")
    print("Sens = " + str(TPR_one))
    print("Spec = " + str(TNR_one))

    print("Class 2:")
    print("Sens = " + str(TPR_two))
    print("Spec = " + str(TNR_two))

    pred = mod.predict(test)
    ROC = roc_auc_score(pd.get_dummies(Y_test), pred, multi_class="ovr")

    print("Accuray = " + str(acc) + ", ROC = " + str(ROC))

    pred = np.argmax(mod.predict(test), axis=-1)
    print("Confusion matrix: (row = true, column = predicted)")
    print(confusion_matrix(Y_test.iloc[:,0], pred))


    class_rep = classification_report(Y_test.iloc[:,0], pred)
    print("Classifcation Report")
    print(class_rep)




def CreateData_LSTM(data_std, Y, val_split, test_split):
    r,x = data_std.shape
    data = data_std.values.reshape((1, r, x))
    Y_std = Y.values.reshape((1,r,1))
    
    train_lstm = data[:,:val_split,:]
    #val_lstm = data[:,14000:15500,:]
    val_lstm = data[:,:test_split,:]
    #test_lstm = data[:,15500:,:]
    test_lstm = data[:,:,:]
    
    Y_train_lstm = Y_std[:,:val_split,:]
    #Y_val_lstm = Y_std[:,14000:15500,:]
    Y_val_lstm = Y_std[:,:test_split,:]
    #Y_test_lstm = Y_std[:,15500:,:]
    Y_test_lstm = Y_std[:,:,:]


    return train_lstm, val_lstm, test_lstm, Y_train_lstm, Y_val_lstm, Y_test_lstm
    

def LSTM_Model_1(train_lstm, Y_train_lstm, val_lstm, Y_val_lstm, x, seed, EP):

    model = Sequential()
    model.add(LSTM(4, input_shape=(None, x), return_sequences = True))
    model.add(Dense(3, activation = "softmax"))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    history = model.fit(train_lstm, Y_train_lstm, epochs=EP, validation_data = (val_lstm, Y_val_lstm), verbose = 0, callbacks=[callback])


    return history, model


def LSTM_Model_2(train_lstm, Y_train_lstm, val_lstm, Y_val_lstm, x, seed, EP):

    model = Sequential()
    model.add(LSTM(8, input_shape=(None, x), return_sequences = True))
    model.add(LSTM(4, return_sequences = True))
    model.add(Dense(3, activation = "softmax"))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    history = model.fit(train_lstm, Y_train_lstm, epochs=EP, validation_data = (val_lstm, Y_val_lstm), verbose = 0, callbacks=[callback])


    return history, model


def LSTM_Model_3(train_lstm, Y_train_lstm, val_lstm, Y_val_lstm, x, seed, EP):

    model = Sequential()
    model.add(LSTM(16, input_shape=(None, x), return_sequences = True))
    model.add(LSTM(8, return_sequences = True))
    model.add(LSTM(4, return_sequences = True))
    model.add(Dense(3, activation = "softmax"))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    history = model.fit(train_lstm, Y_train_lstm, epochs=EP, validation_data = (val_lstm, Y_val_lstm), verbose = 0, callbacks=[callback])


    return history, model




def Run_LSTM(data_std, Y, name, val_split, test_split, EP = 100):


    #seeds = [1, 10, 100, 123, 300, 500, 555, 999, 1000, 123456]

    seeds = [1, 123, 999]


    r,x = data_std.shape



    train_lstm, val_lstm, test_lstm, Y_train_lstm, Y_val_lstm, Y_test_lstm = CreateData_LSTM(data_std, Y, val_split, test_split)

    Hiss1 = []
    Mods1 = []
    Hiss2 = []
    Mods2 = [] 
    Hiss3 = []
    Mods3 = []

    for seed in seeds:
        His1, Mod1 = LSTM_Model_1(train_lstm,Y_train_lstm,val_lstm,Y_val_lstm, x, seed, EP)
        His2, Mod2 = LSTM_Model_2(train_lstm,Y_train_lstm,val_lstm,Y_val_lstm, x, seed, EP)
        His3, Mod3 = LSTM_Model_3(train_lstm,Y_train_lstm,val_lstm,Y_val_lstm, x, seed, EP)


        Hiss1.append(His1)
        Mods1.append(Mod1)

        Hiss2.append(His2)
        Mods2.append(Mod2)

        Hiss3.append(His3)
        Mods3.append(Mod3)

        
    def SelectBest(His,Mod):


        if (min(His[0].history["val_loss"]) < min(His[1].history["val_loss"])) & (min(His[0].history["val_loss"]) < min(His[2].history["val_loss"])):
            BestHis = His[0]
            BestMod = Mod[0]
            print("Seed " + str(seeds[0]) + " was best") 


        if (min(His[1].history["val_loss"]) < min(His[0].history["val_loss"])) & (min(His[1].history["val_loss"]) < min(His[2].history["val_loss"])):
            BestHis = His[1]
            BestMod = Mod[1]
            print("Seed " + str(seeds[1]) + " was best") 


        if (min(His[2].history["val_loss"]) < min(His[0].history["val_loss"])) & (min(His[2].history["val_loss"]) < min(His[1].history["val_loss"])):
            BestHis = His[2]
            BestMod = Mod[2]
            print("Seed " + str(seeds[2]) + " was best") 

        return BestHis, BestMod

    His1, Mod1 = SelectBest(Hiss1, Mods1)
    His2, Mod2 = SelectBest(Hiss2, Mods2)
    His3, Mod3 = SelectBest(Hiss3, Mods3)


    
    return test_lstm, Y_test_lstm, His1, Mod1, His2, Mod2, His3, Mod3




def Eval_LSTM(mod, test_lstm, Y_test_lstm, split):

    a,b,c = test_lstm.shape

    n = b - split

    pred = np.argmax(mod.predict(test_lstm), axis=-1)
    acc = sum(pred[0,split:] == Y_test_lstm[0,split:,0])/n

    True_zero = sum(Y_test_lstm[0,split:,0] == 0) 
    True_one = sum(Y_test_lstm[0,split:,0] == 1) 
    True_two = sum(Y_test_lstm[0,split:,0] == 2) 


    Pred_zero = sum(pred[0,split:] == 0) 
    Pred_one = sum(pred[0,split:] == 1) 
    Pred_two = sum(pred[0,split:] == 2) 

    TP_one = sum((Y_test_lstm[0,split:,0] == 1) & (pred[0,split:] == 1))
    TP_two = sum((Y_test_lstm[0,split:,0] == 2) & (pred[0,split:] == 2))

    TN_one = sum((Y_test_lstm[0,split:,0] != 1) & (pred[0,split:] != 1))
    TN_two = sum((Y_test_lstm[0,split:,0] != 2) & (pred[0,split:] != 2))

    TPR_one = TP_one/True_one
    TPR_two = TP_two/True_two

    TNR_one = TN_one / (True_two + True_zero)
    TNR_two = TN_two / (True_one + True_zero)

    print("Class 1:")
    print("Sens = " + str(TPR_one))
    print("Spec = " + str(TNR_one))

    print("Class 2:")
    print("Sens = " + str(TPR_two))
    print("Spec = " + str(TNR_two))

    pred = mod.predict(test_lstm)
    ROC = roc_auc_score(pd.get_dummies(Y_test_lstm[:,split:,:].reshape(n)), pred[:,split:,:].reshape((n,3)), multi_class="ovr")

    print("Accuracy = " + str(acc) + ", ROC = " + str(ROC))

    pred = np.argmax(mod.predict(test_lstm), axis=-1)
    print("Confusion matrix: (row = true, column = predicted)")
    print(confusion_matrix(Y_test_lstm[0,split:,0], pred[0,split:]))

    class_rep = classification_report(Y_test_lstm[0,split:,0], pred[0,split:])
    print("Classifcation Report")
    print(class_rep)

    

