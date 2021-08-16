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
