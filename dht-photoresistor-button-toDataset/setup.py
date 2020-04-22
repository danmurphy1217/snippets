from plotly.graph_objs import *
import streamlit as st
import numpy as np
import pandas as pd
import pandas.plotting as pdplt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import plotly.express as pl
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import webbrowser
import time

st.title("Machine Learning Models predicting on self-collected data")
url = "https://docs.google.com/spreadsheets/d/1F-UPZf3je1x4M8ryp34OCe29E-CagCrUn9mFzsyfmJE/edit?usp=sharing"
st.write("A sample of the data can be found here: \n")
if st.button('Check out the dataset'):
    webbrowser.open_new_tab(url)

st.header("**Part 1. Data Pre-Clean vs. Post-Clean**")
"""
The data used to predict whether I am in my room or not includes: 
"""
"""
1. **_Photoresistor_** data that reports the amount of light in the surrounding area (the higher the luminosity, the lower the resistance)
"""
"""
2. **_DHT-sensor_** data that reports the temperature and humidity in the room
"""
"""
3. **_Digital sensor_** data (1's and 0's... 1 = im in my room, 0 = im not in my room) and 
"""
"""
4. **_Dates_** (used to incorporate time series into the model)
"""
st.subheader("**_Part A. Pre-Clean_**")
df = pd.read_csv("esp8266_readings - Sheet1.csv")
df = pd.DataFrame(df)
st.dataframe(df.style.highlight_max(axis=0))
r, c = df.shape
st.text("Shape of Initial Data: {}".format(df.shape) + ". There are " + str(r) + " rows and " + str(c) + " columns.")
"""
Kind of messy... we need to: \n
1. Rename Columns. What are Value1? Value2? etc... \n
2. By taking a look at Value3, we see that the first 9 rows are different from the following 5382. We'll want to remove these or determine a representative placeholder for the missing values. \n
3. Normalize columns 3 (Value1), 4 (Value2), and 5 (Value3) \n
3. Split the "Date" column at 'at' and retrieve the hour for each row. Then, we will one-hot-encode the hour values.
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
cols = ['Hour_1', 'Hour_2', 'Hour_3', 'Hour_4', 'Hour_5', 'Hour_6', 'Hour_7', 'Hour_8', 'Hour_9', 'Hour_10', 'Hour_11', 'Hour_12', 'Hour_13', 'Hour_14', 'Hour_15', 'Hour_16', 'Hour_17', 'Hour_18', 'Hour_19', 'Hour_20', 'Hour_21', 'Hour_22', 'Hour_23', 'Hour_24']
# now, use the reorganized column names to reorganize your columns
time_df = time_df[cols]
#rearrange indeces and concat time to initial df
df.index = np.arange(0, r-9)
df = pd.concat([df, time_df], axis =1)
#drop unnecessary columns
df = df.drop(columns=['Date'])

#normalization
df['Photoresistor'] = df['Photoresistor'].astype(int)/df['Photoresistor'].max().astype(int)
df['Temp'] = df['Temp'].astype(float)/df['Temp'].astype(float).max()
df['Humidity'] = df['Humidity'].astype(float)/df['Humidity'].astype(float).max()
r, c = df.shape
st.dataframe(df)
st.text("Shape of Cleaned Data: {}".format(df.shape) + ". There are " + str(r) + " rows and " + str(c) + " columns.")


st.header("**Part 2. Use and Analyze Various Machine Learning Models:**")
x = df.drop(columns=["Digital_Button", "Event_Name"])
y = df['Digital_Button']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state = 42)

st.subheader("First, Choose Your Model: ")
model = st.selectbox(
    "Model Selection:",
    ["K-Nearest Neighbors", "Support Vector Classifier", "Decision Tree", "Logistic Regression"]
    )
st.subheader("Second, Choose Your Parameters:")
st.text("If you need help, click here: ")

# @st.cache(show_spinner=True, suppress_st_warning=True, hash_funcs={st.DeltaGenerator.DeltaGenerator: lambda _ : None})
def run_grid_search(model_name):
    if model_name == "K-Nearest Neighbors":
        param_grid = {
            "n_neighbors" : [5, 10, 15],
            "weights" : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brue']
        }
        gs_KNN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        gs_KNN.fit(x_train, y_train)
        best_parameters = gs_KNN.best_params_
        st.write("The parameters that lead to the highest accuracy are: ")
        return st.table(pd.DataFrame(best_parameters, index=[0]))

    elif model_name == "Support Vector Classifier":
        param_grid = {
            'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']

        }
        gs_SVC = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1)
        gs_SVC.fit(x_train, y_train)
        best_parameters = gs_SVC.best_params_
        st.write("The parameters that lead to the highest accuracy are: ")
        return st.table(pd.DataFrame(best_parameters, index=[0]))

    elif model_name == "Decision Tree":
        param_grid = {
            'criterion':['gini', 'entropy'],
            'splitter':['best', 'random'],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        gs_DT = GridSearchCV(estimator= DecisionTreeClassifier(), param_grid= param_grid, scoring="accuracy")
        gs_DT.fit(x_train, y_train)
        best_parameters = gs_DT.best_params_
        st.write("The parameters that lead to the highest accuracy are: ")
        return st.table(pd.DataFrame(best_parameters, index=[0]))

    elif model_name == "Logistic Regression":
        param_grid = {
            'penalty':['l1', 'l2'],
            'solver':['lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter':[100, 150]

        }
        gs_LOG = GridSearchCV(estimator = LogisticRegression(), param_grid=param_grid, scoring='accuracy')
        gs_LOG.fit(x_train, y_train)
        best_parameters = gs_LOG.best_params_
        st.write("The parameters that lead to the highest accuracy are: ")
        return st.table(pd.DataFrame(best_parameters, index=[0]))



if st.button("Run a Grid Search"):
   run_grid_search(model)

if model == "K-Nearest Neighbors":
    algo = st.selectbox(
        "Algorithm to computer nearest neighbor: ",
        ["ball_tree", "kd_tree", "brute", "auto"]
    )
    neighbors = st.selectbox(
        "Number of Neighbors to use:",
        [5, 10, 15, 20, 25]
    )
    w = st.selectbox(
        "Weight function",
        ["uniform", "distance"]
    )
    knn_clf = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algo, n_jobs=-1, weights=w)
    knn_clf.fit(x_train, y_train)
    preds_knn = knn_clf.predict(x_test)
    results = pd.DataFrame(
       { "Actual": y_test,
        "Predicted": preds_knn
       }
    )
    score = knn_clf.score(x_test, y_test)
    st.write(score)
elif model == "Support Vector Classifier":
    deg = st.selectbox(
        "Degree of Polynomial: ",
        ["2","3", "4"]
    )
    gam = st.selectbox(
        "Gamma, the Kernel Coefficient",
        ["scale", "auto"]
    )
    kern = st.selectbox(
        "Kernel Type used in Algorithm:",
        ["linear", "poly", "rbf", "sigmoid"]
    )

    svc_clf = SVC(kernel= kern, degree = int(deg), random_state= 42, gamma=gam)
    svc_clf.fit(x_train, y_train)
    y_preds_svm = svc_clf.predict(x_test)
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_preds_svm
    })
    score = svc_clf.score(x_test, y_test)
    st.write(score)
elif model == "Decision Tree":
    criteria =st.selectbox(
        "Function to measure the split quality:",
        ["gini", "entropy"]
    )
    feats = st.selectbox(
        "Max Features considered for best split:",
        ["auto", "sqrt", "log2", None]
    )
    split = st.selectbox(
        "Splitter (the strategy used to choose the split at each node):",
        ["best", "random"]
    )

    dt_clf = DecisionTreeClassifier(criterion=criteria, splitter=split, max_features=feats, random_state=42)
    dt_clf.fit(x_train, y_train)
    y_preds_dt = dt_clf.predict(x_test)
    results = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_preds_dt
        }
    )
    score = dt_clf.score(x_test, y_test)
    st.write(score)
elif model == "Logistic Regression":
    iters = st.selectbox(
        "Max iterations for solvers to converge: ",
        [50, 100, 150, 200]
    )
    pen = st.selectbox(
        "Penalty (l1 or l2)",
        ["l2", "l1"]
    )
    sol = st.selectbox(
        "Algorithm to use in the optimization",
        ["lbfgs", "liblinear", "sag", "saga"]
    )

    log_clf = LogisticRegression(penalty=pen, random_state=42, solver= sol, n_jobs=-1, max_iter = iters)
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
            ["Confusion Matrix", "Test Accuracy Vs. # of Neighbors"]
        )
        # if viz_selector == "Boxplot":
        #     fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=[8, 6])
        #     plt1 = sns.boxplot(data= df, y="Photoresistor", ax = ax1)
        #     plt2 = sns.boxplot(data= df, y="Temp", ax=ax2)
        #     plt3 = sns.boxplot(data= df, y="Humidity", ax=ax3)
        #     plt4 = sns.boxplot(data=df, y="Digital_Button", ax=ax4)

        #     return st.plotly_chart(fig)
        if viz_selector=="Test Accuracy Vs. # of Neighbors":
            num_neighbors = st.selectbox(
                "Choose Range for Number of Neighbors",
                [[5, 10, 15, 20, 25], [10, 20,30, 40, 50]]
            )
            accuracy_scores = []
            for i in num_neighbors:
                knn_clf2 = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
                knn_clf2.fit(x_train, y_train)
                preds = knn_clf2.predict(x_test)
                score = knn_clf2.score(x_test, y_test)
                accuracy_scores.append(score)
            
            fig = pl.line(x=num_neighbors, y=accuracy_scores, hover_name = accuracy_scores)
            fig.update_layout(title="Accuracy for {} Neighbors".format(num_neighbors), xaxis_title = "Number of Neighbors", yaxis_title = "Accuracy (0 - 1.0)")
            return st.plotly_chart(fig)
        elif viz_selector=="Confusion Matrix":
            c_mat = pd.crosstab(preds_knn, y_test, rownames=["True"], colnames=["False"])
            st.write(c_mat)
            fig = pl.imshow(c_mat, labels=dict(
                x = 'Testing Data',
                y = 'Predicted Data'
            ), x= ["False", "True"], y=["False", "True"])
            fig.update_xaxes(side = "top")
            return st.plotly_chart(fig)
    if model == "Support Vector Classifier":
        viz_selector = st.selectbox(
            "Choose your visualization for {}: ".format(model),
            ["Confusion Matrix", "Test Accuracy Vs. Kernel",]
        )
        if viz_selector == "Confusion Matrix":
            c_mat = pd.crosstab(y_preds_svm, y_test, colnames= ['False'], rownames=['True'])
            st.write(c_mat)
            fig = pl.imshow(c_mat, 
            labels=dict(x="Testing Data", y="Predicted Data"),
                    x=['False', 'True'],
                    y=['False', 'True']
            )
            fig.update_xaxes(side = "top")

            return st.plotly_chart(fig) 
        elif viz_selector == "Test Accuracy Vs. Kernel":
            svc_kern1 = SVC(kernel='linear')
            svc_kern2 = SVC(kernel='poly')
            svc_kern3 = SVC(kernel='rbf')
            svc_kern4 = SVC(kernel='sigmoid')


            fit_m1 = svc_kern1.fit(x_train, y_train)
            fit_m2 = svc_kern2.fit(x_train, y_train)
            fit_m3 = svc_kern3.fit(x_train, y_train)
            fit_m4 = svc_kern4.fit(x_train, y_train)

            data_for_chart = dict(
                {
                    'linear': svc_kern1.score(x_test, y_test),
                    'poly': svc_kern2.score(x_test, y_test),
                    'rbf': svc_kern3.score(x_test, y_test),
                    'sigmoid': svc_kern4.score(x_test, y_test)
                }
            )
            df = pd.DataFrame(data_for_chart, index=[0], columns= ['linear', 'poly', 'rbf', 'sigmoid'])
            fig = pl.bar(df, x = ['linear', 'rbf', 'sigmoid', 'poly'], y = [svc_kern1.score(x_test, y_test), 
            svc_kern3.score(x_test, y_test), svc_kern4.score(x_test, y_test), svc_kern2.score(x_test, y_test)], 
            color=[svc_kern1.score(x_test, y_test), svc_kern3.score(x_test, y_test), svc_kern4.score(x_test, y_test), svc_kern2.score(x_test, y_test)],
            labels={
                'x':'Kernel Type',
                'y': 'Accuray (estimator.score())'
            })
            fig.update_layout(title="Accuracy Score vs. Kernel Type")
            return st.plotly_chart(fig)


    if model == "Logistic Regression":
        viz_selector = st.selectbox(
            "Choose your visualization for {}: ".format(model),
            ["Confusion Matrix", "Test Accuracy vs. Optimization Algorithm Used"]
        )
        if viz_selector == "Confusion Matrix":
            c_mat = pd.crosstab(y_preds_log, y_test, rownames=["True"], colnames=["False"])
            st.write(c_mat)
            fig = plt.subplots(1, 1, figsize=[8, 6])
            fig = pl.imshow(confusion_matrix(y_preds_log, y_test), labels=dict(x="Testing Data", y="Predicted Data"),
                        x=['False', 'True'],
                        y=['False', 'True']
                )
            fig.update_xaxes(side = "top")
            return st.plotly_chart(fig)
        elif viz_selector == "Test Accuracy vs. Optimization Algorithm Used":
            log_lbfgs = LogisticRegression(solver='lbfgs')
            fit_model1 = log_lbfgs.fit(x_train, y_train)

            log_liblinear = LogisticRegression(solver='liblinear')
            fit_model2 =  log_liblinear.fit(x_train, y_train)
            
            log_sag = LogisticRegression(solver='sag')
            fit_model3 = log_sag.fit(x_train, y_train)

            log_saga = LogisticRegression(solver='saga')
            fit_model4 = log_saga.fit(x_train, y_train)

            data_for_chart = dict({
                    "lbfgs": fit_model1.score(x_test, y_test),
                    'liblinear' : fit_model2.score(x_test, y_test),
                    'sag' : fit_model3.score(x_test, y_test),
                    'saga' : fit_model4.score(x_test, y_test)

                })
            df = pd.DataFrame(data_for_chart, index=[0])
            fig = pl.bar(df, x = ['lbfgs', 'liblinear', 'sag', 'saga'], y = [fit_model1.score(x_test, y_test), fit_model2.score(x_test, y_test), fit_model3.score(x_test, y_test), fit_model4.score(x_test, y_test)], color = [fit_model1.score(x_test, y_test), fit_model2.score(x_test, y_test), fit_model3.score(x_test, y_test), fit_model4.score(x_test, y_test)])
            return st.plotly_chart(fig)
            
    if model == "Decision Tree":
        viz_selector = st.selectbox(
            "Choose your visualization for {}: ".format(model),
            ["Confusion Matrix", "Accuracy vs. Function"]
        )
        if viz_selector == "Confusion Matrix":
            c_mat = pd.crosstab(y_preds_dt, y_test, rownames=['True'], colnames=['False'])
            st.write(c_mat)
            fig = pl.imshow(c_mat
            , 
            labels=dict(x="Testing Data", y="Predicted Data"),
                    x=['False', 'True'],
                    y=['False', 'True']
            )
            fig.update_xaxes(side = "top")
            return st.plotly_chart(fig)
        elif viz_selector == "Accuracy vs. Function":
            dt_gini = DecisionTreeClassifier(criterion='gini')
            dt_gini.fit(x_train, y_train)
            score_gini = dt_gini.score(x_test, y_test)

            dt_entropy = DecisionTreeClassifier(criterion='entropy')
            dt_entropy.fit(x_train, y_train)
            score_entropy = dt_entropy.score(x_test, y_test)

            data_for_chart = dict(
                {
                    'gini': score_gini,
                    'entropy': score_entropy
                }
            )
            df = pd.DataFrame(data_for_chart, index=[0], columns= ['gini', 'entropy'])
            fig = pl.bar(df, x = ['gini', 'entropy'], y = [score_gini, score_entropy], 
            color=[score_gini, score_entropy],
            labels={
                'x':'Function Type',
                'y': 'Accuray (estimator.score())'
            })
            fig.update_layout(title="Function Type vs. Accuracy")
            return st.plotly_chart(fig)

return_visualization()
st.header("**Part 3. Input your own data and check out the results:**")
st.write("It is recommended you look at the values in the cleaned dataframe to aid in helping you select your inputs.")

st.subheader("First, Input your data:")


photo_val = st.text_input("Photoresistor Value: ", value=".4541")
temp_val = st.text_input("Temperature Value: ", value='.7368')
humid_val = st.text_input("Humidity Value: ", value='.8923')
hour_num = st.selectbox(
    "Hour Value: ",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
)
hour_timeframe = st.selectbox(
    "AM or PM?",
    ["AM", "PM"]
)
st.subheader("Now, select your model:")
model_selector = st.selectbox(
    "Model Selector:",
    ["K-Nearest Neighbors", "Support Vector Classifier", "Decision Tree", "Logistic Regression"]
)
def hourConverterTwo(timeframe, hour):
    if timeframe == "PM":
        final_time = int(hour) + 12
        return int(final_time)
    else:
        return int(hour)

@st.cache(show_spinner=True)
def transform_user_input(photo, temp, humidity, hour):
    hour_val = [0 if i!= hourConverterTwo(hour_timeframe, hour_num) else 1 for i in range(1, 25)]
    input_data_array = np.array([float(photo_val), float(temp_val), float(humid_val)]+[i for i in hour_val])
    return input_data_array

confirmation_button = st.button("Run Model and Predict")

@st.cache(show_spinner=True, hash_funcs={st.DeltaGenerator.DeltaGenerator: lambda _: None})
def buildModel(model_name, input_data):
    if model_name == "K-Nearest Neighbors":
        knn_clf = KNeighborsClassifier(algorithm='auto', n_neighbors=5, weights='distance', n_jobs=-1)
        knn_clf.fit(x_train, y_train)
        preds = knn_clf.predict(input_data.reshape(1, -1))
    elif model_name == "Support Vector Classifier":
        svc_clf = SVC(degree=4, gamma='scale', kernel='poly')
        svc_clf.fit(x_train, y_train)
        preds = svc_clf.predict(input_data.reshape(1, -1))
        
    elif model_name=="Decision Tree":
        dt_clf = DecisionTreeClassifier(criterion='entropy', max_features='log2', splitter='random')
        dt_clf.fit(x_train, y_train)
        preds = dt_clf.predict(input_data.reshape(1, -1))
    elif model_name == "Logistic Regression":
        log_clf = LogisticRegression(penalty='l1', max_iter=150, solver='saga', warm_start=0, n_jobs=-1)
        log_clf.fit(x_train, y_train)
        preds = log_clf.predict(input_data.reshape(1, -1))
    return st.dataframe(preds)
if confirmation_button:
    buildModel(model_selector, transform_user_input(photo= photo_val, temp=temp_val, humidity= humid_val, hour=hour_num))

