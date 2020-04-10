import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

st.title("Support Vector Classifier for Predicting whether I am in my room or not")

"""
The data used to predict whether I am in my room or not includes: 
"""
"""
1. **Normalized photoresistor** data that reads the amount of light in my room, 
"""
"""
2. **Normalized DHT-sensor** data that reads the temperature and humidity in my room, 
"""
"""
3. **Digital sensor** data (1's and 0's... 1 = im in my room, 0 = im not in my room), and 
"""
"""
4. **One-hot-encoded hours (0-23)** (used to incorporate time series into the model)
"""

"""
Here's a glimpse of the data **pre-clean**:
"""

df = pd.read_csv("/Users/danielmurphy/Desktop/snippets/dht-photoresistor-button-toDataset/esp8266_readings - Sheet1.csv")
df = pd.DataFrame(df)
df
r, c = df.shape
st.text("Shape of Initial Data: {}".format(df.shape) + ". There are " + str(r) + " rows and " + str(c) + " columns.")
"""
Kind of messy... Here it is **post-clean**:
"""

# rename columns
df = df.rename(columns={
    'Event Name': 'Event_Name', 
    'Value1': 'Digital_Button', 
    'Value2':'Photoresistor', 
    'Value3':'Temp; Humidity'
    })
# drop rows where humidity sensor was not activated
df = df.drop([0, 1, 2, 3, 4, 5, 6, 7, 8])

#clean temp: humidity column
df[['Temp', 'Humidity']] = df['Temp; Humidity'].str.split(';', expand=True)
df = df.drop(columns="Temp; Humidity")

#clean date column
date_df = pd.DataFrame(df['Date'])
date_df = date_df['Date'].str.split('at', expand = True)


# convert from 12 hr to 24 hr time
def hourConverter():
    times = []
    for val in date_df[1].iteritems():
        if str(val[1][-2:]) == 'AM':
            times.append(int(val[1][0:3]))
        elif str(val[1][-2:]) == 'PM':
            times.append(int(val[1][0:3]) + 12)
    return times

time_df = pd.DataFrame(hourConverter())
time_df = time_df.rename(columns={0: 'Hour'})
# OHE hours
time_df = pd.get_dummies(time_df.astype(str))
#rearrange indeces and concat time to initial df
df.index = np.arange(0, len(df))
df = pd.concat([df, time_df], axis =1)
#drop unnecessary columns
df = df.drop(columns=['Date'])

#normalization
df['Photoresistor'] = df['Photoresistor']/df['Photoresistor'].max()
df['Temp'] = df['Temp'].astype(float)/df['Temp'].astype(float).max()
df['Humidity'] = df['Humidity'].astype(float)/df['Humidity'].astype(float).max()
df
r, c = df.shape
st.text("Shape of Cleaned Data: {}".format(df.shape) + ". There are " + str(r) + " rows and " + str(c) + " columns.")
"""
**Now, let's build the Support Vector Classifier Model:**
"""

from sklearn.svm import SVC
x = df.drop(columns=["Digital_Button", "Event_Name"]).values
y = df['Digital_Button'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state = 42)


svm_clf = SVC(kernel="linear")
svm_clf.fit(x_train, y_train)

y_preds = svm_clf.predict(x_test)

results = pd.DataFrame({
    'Actual': y_test,
    'Predictions': y_preds
})
st.text("Here, we can see the actual results vs. our model's predictions... not too bad!")
results
kernels = ["linear", "poly", 'rbf', 'sigmoid']
def computeAccuracy(kern):
    try:
        svm_clf = SVC(kernel = kern)
        svm_clf.fit(x_train, y_train)
        y_preds = svm_clf.predict(x_test)
    except:
       svm_clf = SVC(kernel = kern)
       svm_clf.fit(x_train, y_train)
       y_preds = svm_clf.predict(x_test)
       return "Error with model, returning linear accuracy: {}".format(svm_clf.score(x_test, y_test))
    return svm_clf.score(x_test, y_test)
option = st.selectbox(
    "Which Kernel Would You Like to Choose for the Model?",
    kernels
)
st.text("The accuracy of your model is: {}".format(computeAccuracy("{}".format(option))))
accuracy_records = {}
for kern in kernels:
    accuracy_records["{}".format(kern)] = computeAccuracy("{}".format(kern))
st.text("The kernel with the highest accuracy is: " + str(max(accuracy_records, key=accuracy_records.get)))



