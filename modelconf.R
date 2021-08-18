devtools::install_github("nielsaka/modelconf")

library(tidyverse)


library(modelconf)

library(Rcpp)

library(MCS)

?MCS

?MCSprocedure

df = read_csv("mastersthesis/conf_test.csv") %>% 
  select(-X1) %>% 
  na.omit()

estMCS(df, "t.max", B = 10000, l = 10)

MCSprocedure(df, alpha = 0.05)

ETH = read_csv("mastersthesis/ConfSets/ETH_1min_conf.csv") %>% 
  select(-X1) %>% 
  na.omit()

estMCS(ETH, "t.range", B = 10000, l = 10)

MCSprocedure(ETH, alpha = 0.05)

colMeans(ETH)


ADA = read_csv("mastersthesis/ConfSets/ADA_1min_conf.csv") %>% 
  select(-X1) %>% 
  na.omit()

estMCS(ADA, test = "t.range")
MCSprocedure(ADA, alpha = 0.05)

colMeans(ADA)


BTC5M = read_csv("mastersthesis/ConfSets/BTC_5min_conf.csv") %>% 
  select(-X1) %>% 
  na.omit()

estMCS(BTC5M, test = "t.range")
MCSprocedure(BTC5M, alpha = 0.05)

colMeans(BTC5M)


ETH5M = read_csv("mastersthesis/ConfSets/ETH5M_5min_conf.csv") %>% 
  select(-X1) %>% 
  na.omit()

estMCS(ETH5M, test = "t.range")
MCSprocedure(ETH5M, alpha = 0.05)

colMeans(ETH5M)


ETH1M5M = read_csv("mastersthesis/ConfSets/ETH1M5M_1min5min_conf.csv") %>% 
  select(-X1) %>% 
  na.omit()

MCSprocedure(ETH1M5M, alpha = 0.05)

ETH1M5M_5 = read_csv("mastersthesis/ConfSets/ETH1M5M_1min5min_conf.csv") %>% 
  select(-X1) %>% 
  slice(which(row_number() %% 5 == 1)) %>% 
  slice(which(row_number() > 19)) %>% 
  na.omit()
  
  
  
ETH_two <- cbind(ETH1M5M_5, ETH5M)

MCSprocedure(ETH_two, alpha = 0.15)


  
ETH1M5M_5

colnames(ETH1M5M_5) <- c("BA_DENSE_1_1min", "BA_DENSE_2_1min", "BA_DENSE_3_1min", "BA_LSTM_1_1min",  "BA_LSTM_2_1min",  "BA_LSTM_3_1min",  "OF_DENSE_1_1min",
"OF_DENSE_2_1min", "OF_DENSE_3_1min", "OF_LSTM_1_1min",  "OF_LSTM_2_1min",  "OF_LSTM_3_1min" , "HF_DENSE_1_1min", "HF_DENSE_2_1min",
 "HF_DENSE_3_1min", "HF_LSTM_1_1min",  "HF_LSTM_2_1min",  "HF_LSTM_3_1min")
  


  

