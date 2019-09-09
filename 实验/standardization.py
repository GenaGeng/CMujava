from sklearn.preprocessing import MinMaxScaler
import pandas as pd

x = pd.read_csv('data.csv', header=None)
x = x.iloc[:, :].values

sc = MinMaxScaler(feature_range=(0,1))

x_data = sc.fit_transform(x)

file = open('standar_data.txt', 'w')
file.write(str(x_data))
file.close()

