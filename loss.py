from deep_models import * 

import os

##### BTC 1 Min ###### 


def LossDF(cryp, freq, v_split, t_split):

	#cols = ["BA_DENSE_1", "BA_DENSE_2", "BA_DENSE_3", "BA_LSTM_1", "BA_LSTM_2", "BA_LSTM_3",
	#	"OF_DENSE_1", "OF_DENSE_2", "OF_DENSE_3", "OF_LSTM_1", "OF_LSTM_2", "OF_LSTM_3",
	#	"HF_DENSE_1", "HF_DENSE_2", "HF_DENSE_3", "HF_LSTM_1", "HF_LSTM_2", "HF_LSTM_3"]

	#df = pd.DataFrame(np.nan, index = range(1440), columns = cols)


	dic = {}

	Y = pd.read_csv(f"data/final/{freq}/{cryp}_Y_std.csv")

	#test_sets, y_sets, His1, His2, His3, Mod1, Mod2, Mod3, Win1, Win2, Win3 = Run_Dense(data_std, Y, "test",v_split, t_split, EP = 10000)

	for feat in ["BA", "OF", "HF"]:

		if feat == "BA":
			data_std = pd.read_csv(f"data/final/{freq}/BidAsk/{cryp}_std.csv")

		if feat == "OF":
			data_std = pd.read_csv(f"data/final/{freq}/OrderFlow/{cryp}_std.csv")

		if feat == "HF":
			data_std = pd.read_csv(f"data/final/{freq}/HF/{cryp}_std.csv")


		test_sets, y_sets, His1, His2, His3, Mod1, Mod2, Mod3, Win1, Win2, Win3 = Run_Dense(data_std, Y, "test",v_split, t_split, EP = 10000)




		#df.loc[f"{feat}_DENSE_1"] = GetLossDense(Mod1, data_std, Y, Win1, v_split, t_split)
		#df.loc[f"{feat}_DENSE_2"] = GetLossDense(Mod2, data_std, Y, Win2, v_split, t_split)
		#df.loc[f"{feat}_DENSE_3"] = GetLossDense(Mod3, data_std, Y, Win3, v_split, t_split)

		dic[f"{feat}_DENSE_1"] = {}
		dic[f"{feat}_DENSE_1"]["loss"] = GetLossDense(Mod1, data_std, Y, Win1, v_split, t_split)
		dic[f"{feat}_DENSE_1"]["win"] = Win1

		dic[f"{feat}_DENSE_2"] = {}
		dic[f"{feat}_DENSE_2"]["loss"] = GetLossDense(Mod2, data_std, Y, Win2, v_split, t_split)
		dic[f"{feat}_DENSE_2"]["win"] = Win2

		dic[f"{feat}_DENSE_3"] = {}
		dic[f"{feat}_DENSE_3"]["loss"] = GetLossDense(Mod3, data_std, Y, Win3, v_split, t_split)
		dic[f"{feat}_DENSE_3"]["win"] = Win3

		mod_num = 0
		for mod in sorted(os.listdir(f"models/lstm/{cryp}_{feat}_{freq}/")):
			mod_num += 1
			model = keras.models.load_model(f"models/lstm/{cryp}_{feat}_{freq}/{mod}")
			loss = GetLoss(model, data_std, Y, v_split, t_split, "LSTM")[0,t_split:]
			dic[f"{feat}_LSTM_{mod_num}"] = {}
			dic[f"{feat}_LSTM_{mod_num}"]["win"] = 0
			dic[f"{feat}_LSTM_{mod_num}"]["loss"] = loss





	#return df
	return dic


#print(sorted(os.listdir("models/lstm/BA/")))

#BTC1M = LossDF("BTC")


#print(BTC1M)
