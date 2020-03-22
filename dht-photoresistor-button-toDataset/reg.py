import numpy as np
import pandas as pd

#load csv, separate x and y data into target features
df = pd.read_csv('/Users/damurphy/Desktop/snippets/dht-photoresistor-button-toDataset/esp8266_readings - Sheet1.csv')
x = df['Dig Button on/off'].values
y = df['Photoresistor'].values
print(x.shape == y.shape) # Shape of y == x , good!

x = x.reshape(-1, 1)
print(x)
