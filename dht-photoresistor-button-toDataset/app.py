import streamlit as st
import numpy as np
import pandas as pd
import pandas.plotting as pdplt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import plotly.express as pl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
st.title("Support Vector Classifier for Predicting whether I am in my room or not")

st.header("**Part 1. Data Pre-Clean vs. Post-Clean**")
"""
The data used to predict whether I am in my room or not includes: 
"""
"""
1. **_Normalized photoresistor_** data that reads the amount of light in my room
"""
"""
2. **_Normalized DHT-sensor_** data that reads the temperature and humidity in my room
"""
"""
3. **_Digital sensor_** data (1's and 0's... 1 = im in my room, 0 = im not in my room) and 
"""
"""
4. **_One-hot-encoded hours (0-23)_** (used to incorporate time series into the model)
"""
st.subheader("**_Part A. Pre-Clean_**")
df = pd.read_csv("esp8266_readings - Sheet1.csv")
df = pd.DataFrame(df)
df
r, c = df.shape
st.text("Shape of Initial Data: {}".format(df.shape) + ". There are " + str(r) + " rows and " + str(c) + " columns.")
"""
Kind of messy... we need to: \n
1. Rename Columns. What are Value1? Value2? etc... \n
2. By taking a look at Value3, we see that the first 9 rows are different from the following 5382. We'll want to remove these or determine a placeholder for the missing values that is representative of the data we are working with. \n
3. Normalize columns 3 (Value1), 4 (Value2), and 5 (Value3) \n
3. Split the "Date" column at 'at' and retrieve the hour for each row (then we can one-hot-encode the hour values)
"""

st.subheader("**_Part B. Post-Clean_**")
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
cols = time_df.columns.tolist() # get name of columns in a list
# now, reorganize those names 
cols = ['Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 'Hour_23']
# now, use the reorganized column names to reorganize your columns
time_df = time_df[cols]
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


st.header("**Part 2. Create Your Own Machine Learning Model:**")
x = df.drop(columns=["Digital_Button", "Event_Name"])
y = df['Digital_Button']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state = 42)

st.subheader("First, test out a variety of models: ")
model = st.selectbox(
    "choose your model:",
    ["K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Logistic Regression"]
    )
st.subheader("Second, choose your parameters:")
if model == "K-Nearest Neighbors":
    neighbors = st.selectbox(
        "number of neighbors to use:",
        [5, 10, 15, 20, 25]
    )
    algo = st.selectbox(
        "the algorithm you want to use to compute the nearest neighbors: ",
        ["ball_tree", "kd_tree", "brute", "auto"]
    )
    n = st.selectbox(
        "the number of jobs (-1 is recommended):",
        [1, -1]
    )
    met = st.selectbox(
        "Euclidean (2) or Manhattan Distance (1)? This is the power parameter for the Minkowski metric.",
        ["1", "2"]
    )
    knn_clf = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algo, n_jobs=n, p = int(met))
    knn_clf.fit(x_train, y_train)
    preds_knn = knn_clf.predict(x_test)
    results = pd.DataFrame(
       { "Actual": y_test,
        "Predicted": preds_knn
       }
    )
    score = knn_clf.score(x_test, y_test)
    st.write(score)
elif model == "Support Vector Machine":
    kern = st.selectbox(
        "the kernel type you want to be used in the algorithm:",
        ["linear", "poly", "rbf", "sigmoid"]
    )
    deg = st.selectbox(
        "the degree of the polynomial kernel function (default = 3): ",
        ["2", "3", "4"]
    )
    rand = st.text_input(
        "set a random state (strongly recommended... suggestions are 42 or 1234)",
        value= "42"
    )
    svc_clf = SVC(kernel= kern, degree = int(deg), random_state= int(rand))
    svc_clf.fit(x_train, y_train)
    y_preds = svc_clf.predict(x_test)
    results = pd.DataFrame({
        "Acutal": y_test,
        "Predicted": y_preds
    })
    score = svc_clf.score(x_test, y_test)
    st.write(score)
elif model == "Decision Tree":
    criteria =st.selectbox(
        "the function you want to use to measure the quality of a split:",
        ["gini", "entropy"]
    )
    split = st.selectbox(
        "the strategy used to choose the split at each node:",
        ["best", "random"]
    )
    feats = st.selectbox(
        "the number of features to consider when looking for the best split:",
        ["auto", "sqrt", "log2", None]
    )
    rand = st.text_input(
        "set a random state (strongly recommended... suggestions are 42 or 1234)",
        value = "42"
    )
    dt_clf = DecisionTreeClassifier(criterion=criteria, splitter=split, max_features=feats, random_state=int(rand))
    dt_clf.fit(x_train, y_train)
    preds= dt_clf.predict(x_test)
    results = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": preds
        }
    )
    score = dt_clf.score(x_test, y_test)
    st.write(score)
elif model == "Logistic Regression":
    pen = st.selectbox(
        "Choose the penalty, which is used to specify the norm used in the penalization (for now, only l2 is supported)",
        ["l2"]
    )
    rand_state = st.text_input(label="Enter the random state (strongly suggested for uniformity across models):", value="1234")
    sol = st.selectbox(
        "Choose the Algorithm to use in the optimization (default is lbfgs)",
        ["lbfgs", "liblinear", "sag", "saga"]
    )
    warm = st.selectbox(
        "If true, reuse the solution of the previous call to fit as initialization. Otherwise, erase previous.",
        ["True", "False"]
    )
    jobs = st.selectbox(
        "Number of jobs (if -1, all CPU cores used.",
        [-1, 1]
    )
    log_clf = LogisticRegression(penalty=pen, random_state=int(rand_state), solver= sol, warm_start=warm, n_jobs=jobs)
    log_clf.fit(x_train, y_train)
    y_preds_log = log_clf.predict(x_test)
    results = pd.DataFrame(
        {
            "Predicted": y_preds_log,
            "Actual": y_test
        }
    )
    score = log_clf.score(x_test, y_test)
    st.write(score)
else:
    unknown_input = []
    unknown_input.append(model)
    st.write("{}".format(model) + " is not available yet, coming soon (:")

st.subheader("Lastly, choose the visualizations to build:")
def return_visualization():
    if model == "K-Nearest Neighbors":
        viz_selector = st.selectbox(
            "Choose your visualization:",
            ["Boxplot", "Test Accuracy Vs. # of Neighbors", "Confusion Matrix"]
        )
        if viz_selector == "Boxplot":
            fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=[8, 6])
            plt1 = sns.boxplot(data= df, y="Photoresistor", ax = ax1)
            plt2 = sns.boxplot(data= df, y="Temp", ax=ax2)
            plt3 = sns.boxplot(data= df, y="Humidity", ax=ax3)
            plt4 = sns.boxplot(data=df, y="Digital_Button", ax=ax4)

            return st.plotly_chart(fig)
        elif viz_selector=="Test Accuracy Vs. # of Neighbors":
            num_neighbors = st.selectbox(
                "Choose Range for Number of Neighbors",
                [[5, 10, 15, 20, 25], [10, 20,30, 40, 50]]
            )
            accuracy_scores = []
            for i in num_neighbors:
                knn_clf2 = KNeighborsClassifier(n_neighbors=i, algorithm=algo, n_jobs=n, p= int(met))
                knn_clf2.fit(x_train, y_train)
                preds = knn_clf2.predict(x_test)
                score = knn_clf2.score(x_test, y_test)
                accuracy_scores.append(score)
            
            fig = pl.line(x=num_neighbors, y=accuracy_scores, hover_name = accuracy_scores)
            fig.update_layout(title="Accuracy vs. {} Neighbors".format(num_neighbors), xaxis_title = "Number of Neighbors", yaxis_title = "Accuracy (0 - 1.0)")
            return st.plotly_chart(fig)
        elif viz_selector=="Confusion Matrix":
            c_mat = pd.crosstab(preds_knn, y_test, rownames=["True"], colnames=["False"])
            st.write(c_mat)
            fig = pl.imshow(c_mat)
            return st.plotly_chart(fig)

    if model == "Logistic Regression":
        viz_selector = st.selectbox(
            "Choose your visualization for {}: ".format(model),
            ["Confusion Matrix"]
        )
        c_mat = pd.crosstab(y_preds_log, y_test, rownames=["True"], colnames=["False"])
        st.write(c_mat)
        fig = plt.subplots(1, 1, figsize=[8, 6])
        fig = pl.imshow(confusion_matrix(y_test, y_preds_log))
        return st.plotly_chart(fig)


return_visualization()

