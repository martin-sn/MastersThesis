import pandas as pd
import numpy as np
import matplotlib as plt
from tqdm.notebook import tqdm, trange
import re
import os
pd.options.mode.chained_assignment = None

freq = 300



def get_list(folder,crypto):
    file_list = os.listdir(folder)
    r = re.compile(crypto)
    crypto_list = list(filter(r.match, file_list))
    return(crypto_list)

def read_files(folder,crypto):
    list_of_files = get_list(folder,crypto)
    df = pd.read_csv((folder+'/'+list_of_files[0]))
    for i in (list_of_files[1:]):
        df = df.append(pd.read_csv((folder+'/'+i)))
    df = df.sort_values('system_time')
    df = df.reset_index(drop=True)
    return(df)

folder = 'data/raw/5min'


BTC_df = read_files(folder,'BTC')
ETH_df = read_files(folder, 'ETH')
ADA_df = read_files(folder, 'ADA')


def check_for_missing(df, freq):
    time = df['system_time']
    t = pd.to_datetime(time[:(len(time)-1)]).reset_index(drop=True)
    t_1 = pd.to_datetime(time[1:]).reset_index(drop=True)
    diff = (t_1-t) / freq
    
    errors = np.sum(diff != '0 days 00:00:01')
    
    
        
    return(errors)


print("MISSING BTC: " + str(check_for_missing(BTC_df, freq = freq)))
print("MISSING ETH: " + str(check_for_missing(ETH_df, freq = freq)))
print("MISSING ADA: " + str(check_for_missing(ADA_df, freq = freq)))


def Returns(df):
    df["Return"] = np.nan
    #df["Return"] = np.diff(df["midpoint"]) / df["midpoint"][:(len(df)-1)]
    rt = np.diff(np.log(df["midpoint"]))
    df = df[:-1]
    df["Return"] = rt
    
    return(df)


def RealizedVolatility(df, freq):
    df["RV"] = np.nan
    for i in trange((len(df)-freq)):
        df['RV'][(i+freq)] = sum(df['Return'][i:(freq+i)]**2)
    return(df)


def RealizedSemiVariance(df,freq):
    df["RSV_pos"] = np.nan
    df["RSV_neg"] = np.nan
    
    for i in trange((len(df)) - freq): 
        X = df['Return'][i:(freq+i)].reset_index(drop = True)
        
        X_pos = X[X > 0]
        df["RSV_pos"][(i+1+freq)] = sum(X_pos**2)
        
        X_neg = X[X < 0]
        df["RSV_neg"][(i+1+freq)]  = sum(X_neg**2)
        
        
    return(df)




def BipowerVariation(df,freq):
    df["BiV"] = np.nan
    c = np.pi/2
    for i in trange((len(df)) - freq): 
        df["BiV"][(i+1+freq)] = c*sum(np.abs(df['Return'][i:(freq+i)])
                                    *np.abs(df['Return'][(i+1):(freq+i+1)]))
    return(df)




def BipowerVariation_lag2(df,freq):
    df["BiV_2"] = np.nan
    c = np.pi/2
    for i in trange((len(df)) - freq): 
        df["BiV_2"][(i+2+freq)] = c*sum(np.abs(df['Return'][i:(freq+i)])
                                    *np.abs(df['Return'][(i+2):(freq+i+2)]))
    return(df)



def BipowerSemiVariance(df,freq):
    df["BiV_pos"] = np.nan
    df["BiV_neg"] = np.nan
    

    c = np.pi/2
    for i in trange((len(df)) - freq): 
        X = df['Return'][i:(freq+i)].reset_index(drop = True)
        X_1 = df['Return'][(i+1):(freq+i+1)].reset_index(drop=True)
        X_pos = X[X_1 > 0]
        X_1_pos = X_1[X_1 > 0]
        df["BiV_pos"][(i+1+freq)] = c*sum(np.abs(X_pos)*np.abs(X_1_pos))
        X_neg = X[X_1 < 0]
        X_1_neg = X_1[X_1 < 0]
        df["BiV_neg"][(i+1+freq)]  = c*sum(np.abs(X_neg)*np.abs(X_1_neg))

    return(df)
    

def RealizedQuarticity(df, freq):
    df["RQ"] = np.nan
    for i in trange((len(df)-freq)):
        df['RQ'][(i+freq)] = sum(np.abs(np.sqrt(freq)*df['Return'][i:(freq+i)])**4)
    return(df)


def RealizedQuarticityTri(df,freq):
    mu_p = 0.8308478**(-3) # Approximation
    n = freq
    p = 4/3
    r = df["Return"]
    
    df["RQTri"] = np.nan
    

    
    for i in trange(len(df)-freq-2):
        lag_2 = r[i:freq+i]
        lag_1 = r[i+1:freq+1+i]
        lag_0 = r[i+2:freq+2+i]
        
        df["RQTri"][i+freq+2] = n*mu_p*sum(np.abs(lag_0)**p*np.abs(lag_1)**p
                                            *np.abs(lag_2)**p)    
        
    return(df)



def RealizedQuarticityQuad(df, freq):
    mu_p = 0.7978469**(-4) # Approximation
    n = freq
    r = df["Return"]
    df["RQQuad"] = np.nan
    
    
    for i in trange(len(df)-freq-3):
        lag_3 = r[i:freq+i].reset_index(drop = True)
        lag_2 = r[i+1:freq+1+i].reset_index(drop = True)
        lag_1 = r[i+2:freq+2+i].reset_index(drop = True)
        lag_0 = r[i+3:freq+3+i].reset_index(drop = True)
        
        df["RQQuad"][i+freq+3] = n*mu_p*sum(np.abs(lag_0)*np.abs(lag_1)
                                            *np.abs(lag_2)*np.abs(lag_3))   
    return(df)
    


def RealizedKernel(df, H, kernel_type,freq):
    df["RK"] = np.nan

    def kernel(x, kernel_type):
        if kernel_type == "bartlett":
            return(1-x)
        if kernel_type == "cubic":
            return(1-3*x**2+2*x**3)
        if kernel_type == "parzen":
            if 0 <= x <= 0.5:
                return(1 - 6*x**2 + 6*x**3)
            elif 0.5 < x <= 1:
                return(2*(1-x)**3)
            else: 
                print("ERROR")
        
    Y = df["midpoint"]

    def gamma_h(Y,h):
        Y = Y.reset_index(drop = True)
        gamma = np.zeros(len(Y)-h-1)
        for i in range(len(Y)-h-1):
            gamma[i] = (Y[i+h+1] - Y[i+h])*(Y[i+1]-Y[i])
        return(sum(gamma))


    def RK_n(Y):
        gamma_0 = gamma_h(Y,0)
        k_gamma_n = np.zeros(H-2)
        for i in range(H-2):
            k_gamma_n[i] = kernel((i+1)/H, kernel_type)*gamma_h(Y,i+1)
        RK_n = gamma_0 + 2*sum(k_gamma_n)
        return(RK_n)


    for i in trange(len(df)-freq):
            df["RK"][i+freq] = RK_n(Y[i:i+freq])

    return(df)


def PreAvg(df, freq, k):
    
    K = k
    df["PreAvg"] = np.nan
    
    Y = df["Return"]
    
    def kernel(x):
        m = min(x,1-x)
        return(m)
    
    def Pre_n(Y,k):
        
        r = Y 
        
        n = len(r)
        r_pa = 0
        
        for i in range(K-1):
            r_pa = r_pa + kernel(i/k)*r[i:(len(r)+i-(k-1))]
    
        idx = np.array(range(k))/k
        
        g = np.zeros(shape=len(idx))
        
        for i in range(len(idx)):
            g[i] = kernel(idx[i])
        
        dg = np.diff(g)
        psi1 = sum(dg**2)*k
        psi2 = sum(g**2)/k
        
        P = sum(r_pa**2)
        P = P*n/((n-k+2)*k*psi2)
        P = P-sum(r**2)*psi1/(2*psi2*k**2)
        
        return(P)
    
    for i in trange(len(df)-freq):
        df["PreAvg"][i+freq] = Pre_n(Y[i:i+freq],k) #### We only use 60 observations in total, maybe use more?
        
    return(df)




def JumpVariation(df):
    df["JV"] = df["RV"]-df["BiV"] # Can this be nagative?
    df["JV"][df["JV"] < 0] = 0 # Truncate negative values to 0
    df["PJ"] = 1- df["BiV"] / df["RV"] # Proportion of jumps
    df["PJ"][df["PJ"]<0] = 0
    return(df)


def SpotVolatility(df,freq):
    df["SV"] = df["Return"]**2
    
    df["ASV"] = np.nan
    
    for i in trange(len(df)):
        df["ASV"][i+freq] = np.mean(df["SV"][i:(i+freq)])
        
    return(df)


def NoiseVariance(df,freq):
    df["NV"] = np.nan
    Y = df["midpoint"].reset_index(drop = True)
    
    for i in trange(len(df)-freq-1):
        Y_0 = Y[i+1:i+freq+1].reset_index(drop=True)
        Y_1 = Y[i:i+freq].reset_index(drop=True)
        df["NV"][i+freq+1] = 1/(2*freq)*sum((Y_0-Y_1)**2)
    
    return(df)


def GenerateFeatures(df, freq):
    # Statistical Features
    
    df = Returns(df)
    print("Returns Done")
    

    
    # Volatility Measures
    df = RealizedVolatility(df, freq)
    print("RV Done")
    df = RealizedSemiVariance(df,freq)
    print("RSV Done")
    df = BipowerVariation(df, freq) # Maybe 59 in freq here? 
    print("BV Done")
    df = RealizedKernel(df, 10, "cubic", freq)
    print("RK Done")
    df = PreAvg(df, freq, 10)
    print("PreAvg DOne")
    df = JumpVariation(df)
    print("Jump Var Done")
    df = SpotVolatility(df, freq)
    print("Spot Var Done")
    
    
    
    # Noise and uncertainty measures
    df = RealizedQuarticity(df, freq)
    print("RQ Done")
    DF = RealizedQuarticityTri(df,freq)
    print("RQTri Done")
    df = RealizedQuarticityQuad(df,freq)
    print("RQQuad Done")
    df = NoiseVariance(df, freq)
    print("Noise Variance Done")
    
    df = df.dropna()
    
   # df = df.drop("system_time", axis = 1)
    
    return(df)

Feature_df = GenerateFeatures(BTC_df,18)
Feature_df.to_csv("BTC.csv", index = False)
Feature_df = GenerateFeatures(ETH_df,18)
Feature_df.to_csv("ETH.csv", index = False)
Feature_df = GenerateFeatures(ADA_df,18)
Feature_df.to_csv("ADA.csv", index = False)