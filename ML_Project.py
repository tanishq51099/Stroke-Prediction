import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


image = Image.open('Stroke.jpeg')

df = pd.read_csv('healthcare.csv')

df.drop('id',axis=1,inplace=True) # Dropping Id as it is not of any use

st.title('ML on Healthcare Stroke dataset', anchor=None)

st.image(image)

st.markdown("""
Knowing the signs of a stroke is the first step in stroke prevention. 
A stroke, sometimes called a "brain attack," occurs when blood flow to an area in the brain is cut off. 
The brain cells, deprived of the oxygen and glucose needed to survive, die. If a stroke is not caught early, permanent brain damage or death can result.
This app is designed to predict the probability and likeliness of getting a stroke to a person by performing **Classification** on the Stroke data.
It is said that up to 50% of all strokes are preventable and many risk factors can be controlled before they cause problems if detected early.
* **Python libraries:** Pandas, Streamlit, Numpy, Matplotlib, Scikit Learn
* **Data source:** [Stroke Prediction Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
""")

from sklearn.impute import KNNImputer
my_imp = KNNImputer(missing_values=np.NaN,n_neighbors=1000)
fixed_df = pd.DataFrame(my_imp.fit_transform(df.drop(columns=["avg_glucose_level","gender","hypertension","heart_disease","ever_married","work_type","Residence_type","smoking_status","stroke"])))
fixed_df["gender"] = df["gender"]
fixed_df["hypertension"] = df["hypertension"]
fixed_df["heart_disease"] = df["heart_disease"]
fixed_df["ever_married"] = df["ever_married"]
fixed_df["work_type"] = df["work_type"]
fixed_df["Residence_type"] = df["Residence_type"]
fixed_df["avg_glucose_level"] = df["avg_glucose_level"]
fixed_df["smoking_status"] = df["smoking_status"]
fixed_df["stroke"] = df["stroke"]

df=fixed_df.rename(columns={0: 'age', 1: 'bmi'})

st.markdown("""
    For the missingness of the dataset KNN imputer was used. 
    There was some missingness in the Body Mass Index (bmi) column of the dataset and upon observing the relationship with other columns 
    it was found that bmi had some correlation with the age of patient. 
    So, the KNN imputer was used considering only the age of the patient.
    The data was also scaled using the Standard Scalar function. One hot encoding was also used for categorical features.
    """)

st.text("")
st.text("")
check = st.checkbox('Show the dataset as table data')

if check:
    st.dataframe(df)

st.text("")
st.text("")

st.sidebar.header('Input your information for prediction')

def user_input_features():
    age = st.sidebar.slider('Enter your Age', 0, 82, 60, step=1)
    bmi = st.sidebar.number_input('Enter your Body Mass Index', 5.0, 97.6, 36.6, step=0.1)
    eval_bmi  = st.sidebar.checkbox("Don't know your bmi? Click here to know.")

    if eval_bmi:
    	weight = st.sidebar.number_input('Enter your weight (in Kgs)',0.0,200.0,1.0)
    	height = st.sidebar.number_input('Enter your height (in meters)',0.0,200.0,0.5)
    	final_bmi = weight/(height**2)
    	if weight > 1 and height > 0.51:
    		st.sidebar.metric(label="Your bmi", value=final_bmi)
    heart = st.sidebar.radio("Do you have a heart disease?",("Yes","No"))
    if heart == "Yes":
    	h = 1
    else:
    	h = 0
    temp = st.sidebar.radio("Do you have a High Blood Pressure?",("Yes","No"))
    if temp == "Yes":
    	hypertension = 1
    else:
    	hypertension = 0    
    temp2 = st.sidebar.radio("Are you Married?",("Yes","No"))
    if temp2 == "Yes":
    	m = 1
    else:
    	m = 0

    data = {'age': age,
            'bmi': bmi,
            'heart_disease':h,
            'hypertension': hypertension,
            'ever_married':m}

    features = pd.DataFrame(data, index=[1])
    return features

df_new = user_input_features()

temp_df = df.loc[(fixed_df["stroke"]==1)]
a = df
for i in range(17):
    a = a.append(temp_df)
le = LabelEncoder()
# dfle = a
a.gender = le.fit_transform(a.gender)
a.ever_married = le.fit_transform(a.ever_married)
a.work_type = le.fit_transform(a.work_type)
a.Residence_type = le.fit_transform(a.Residence_type)
X = a[["age","bmi","heart_disease","hypertension","ever_married"]]
y = a[["stroke"]]

my_scaler = StandardScaler()
my_scaler.fit(X)
X_scaled = my_scaler.transform(X)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["KNN Classifier", "SVM Classifier", "Random Forest Classifier", "Decision Tree", "Gaussian Naive Bayes","MLP Classifier"])

with tab1:
	st.header("K Neighbors Classifier")
	predict = st.checkbox("Want to predict? Click here.",key=1000)
	if predict:
   		# st.subheader('User Input parameters')
   		# st.write(df_new)
   		knn = neighbors.KNeighborsClassifier(n_neighbors=2, weights='uniform')
   		knn.fit(X_scaled, y)
   		prediction = knn.predict(df_new)
   		prediction_proba = knn.predict_proba(df_new)
   		st.subheader('Prediction Probability')
   		st.write(prediction_proba)
   		st.subheader('Prediction')
   		if int(prediction) == 0:
   			st.write("**Unlikely to get a stroke.**")
   		else:
   			st.write("**You are prone to getting a Stroke. Consult a doctor.**")
	if st.checkbox("Want to train and check how good the selected model is? Click here.",key=1):
   		st.subheader('Training Model')
   		features1 = st.multiselect('Enter the input features with which you want to train the model',("age","avg_glucose_level","bmi","hypertension","heart_disease","gender","ever_married","work_type","Residence_type"),default=['age','avg_glucose_level','ever_married'],key=10)
   		# st.write(features1)
   		lst2=[]
   		lst3 = []
   		if 'gender' in features1:
   			lst2.append(features1.index('gender'))
   			lst3.append('gender')
   		if 'ever_married' in features1:
   			lst2.append(features1.index('ever_married'))
   			lst3.append('ever_married')
   			# st.write("YES")
   		if 'work_type' in features1:
   			lst2.append(features1.index('work_type'))
   			lst3.append('work_type')
   		if 'Residence_type' in features1:
   			lst2.append(features1.index('Residence_type'))
   			lst3.append('Residence_type')
   		f = tuple(lst3)
   		X_temp = a[features1].values
   		y = a.stroke.values
   		ct = ColumnTransformer([(f, OneHotEncoder(), lst2)], remainder = 'passthrough')
   		X_temp = ct.fit_transform(X_temp)
   		# st.write(X_temp)

   		X1 = X_temp
   		# st.write(X1)

   		my_scaler = StandardScaler()
   		my_scaler.fit(X1)
   		X_scaled1 = my_scaler.transform(X1)
   		size = st.number_input('Enter the train and test proportion ratio.', value =0.2, min_value=0.1, max_value=0.9, step=0.1,key=100)
   		X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled1,y, test_size=size, random_state = 42)
   		nbr = st.number_input('Enter the number of neighbors for training',value =2, min_value=1, step=1)
   		w = st.radio("Select the weights type for training",("Uniform","Distance"))
   		if w=="Uniform":
   			weigh = 'uniform'
   		else:
   			weigh = 'distance'
   		model_1 = neighbors.KNeighborsClassifier(n_neighbors=int(nbr), weights=weigh)
   		model_1.fit(X_train1, y_train1)
   		prediction_test1 = model_1.predict(X_test1)

   		st.subheader("Scores")
   		st.write("Select the score you want to see.")
   		acc1 = st.checkbox("Accuracy")
   		if acc1:
   			accuracy1 = metrics.accuracy_score(y_test1, prediction_test1)
   			st.metric('The accuracy is:', accuracy1)
   		f1sc1 = st.checkbox("F1 score")
   		if f1sc1:
   			f11 = f1_score(y_test1, prediction_test1, average='weighted')
   			st.metric('F1 score is:',f11)
   		p1 = st.checkbox("Precision score")
   		if p1:
   			ps1 = precision_score(y_test1, prediction_test1, average='weighted')
   			st.metric('Precision score is:',ps1)
   		r1 = st.checkbox("Recall score")
   		if r1:
   			rs1 = recall_score(y_test1, prediction_test1, average='weighted')
   			st.metric('Recall score is:',rs1)
   		c_val = st.checkbox("Cross Validation",key=111111)
   		if c_val:
   			kfold = st.slider("Please specify the number of folds.",2,50,5)
   			scores = cross_val_score(model_1, X_scaled1, y, cv=kfold)
   			st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

   		st.text("")
   		st.text("")
   		st.subheader("Confusion Matrix")

   		conf = ConfusionMatrixDisplay.from_estimator(model_1, X_test1, y_test1)
   		st.set_option('deprecation.showPyplotGlobalUse', False)
   		# plt.figure(figsize=(20, 20))
   		st.pyplot(plt.show())

with tab2:
   	st.header("SVM Classifier")
   	predict = st.checkbox("Want to predict? Click here.",key=2000)
   	if predict:
   		# st.subheader('User Input parameters')
   		# st.write(df_new)
   		svm_model = svm.SVC(probability = True)
   		svm_model.fit(X_scaled, y)
   		prediction2 = svm_model.predict(df_new)
   		prediction_proba2 = svm_model.predict_proba(df_new)
   		st.subheader('Prediction Probability')
   		st.write(prediction_proba2)
   		st.subheader('Prediction')
   		if int(prediction2) == 0:
   			st.write("**Unlikely to get a stroke.**")
   		else:
   			st.write("**You are prone to getting a Stroke. Consult a doctor.**")

   	click2 = st.checkbox("Want to train and check how good the selected model is? Click here.",key=2)
   	if click2:
   		st.subheader('Training Model')
   		features2 = st.multiselect('Enter the input features with which you want to train the model',("age","avg_glucose_level","bmi","hypertension","heart_disease","gender","ever_married","work_type","Residence_type"),default=['age','avg_glucose_level','ever_married'],key=20)
   		lst2=[]
   		lst3 = []
   		if 'gender' in features2:
   			lst2.append(features2.index('gender'))
   			lst3.append('gender')
   		if 'ever_married' in features2:
   			lst2.append(features2.index('ever_married'))
   			lst3.append('ever_married')
   		if 'work_type' in features2:
   			lst2.append(features2.index('work_type'))
   			lst3.append('work_type')
   		if 'Residence_type' in features2:
   			lst2.append(features2.index('Residence_type'))
   			lst3.append('Residence_type')
   		f = tuple(lst3)
   		X_temp = a[features2].values
   		y = a.stroke.values
   		ct = ColumnTransformer([(f, OneHotEncoder(), lst2)], remainder = 'passthrough')
   		X_temp = ct.fit_transform(X_temp)

   		X2 = X_temp
   		my_scaler = StandardScaler()
   		my_scaler.fit(X2)
   		X_scaled2 = my_scaler.transform(X2)
   		size = st.number_input('Enter the train and test proportion ratio.', value =0.2, min_value=0.1, max_value=0.9, step=0.1,key=200)
   		X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled2,y, test_size=size, random_state = 42)
   		model_2 = svm.SVC(probability = True)
   		model_2.fit(X_train2, y_train2)
   		prediction_test2 = model_2.predict(X_test2)

   		st.subheader("Scores")
   		st.write("Select the score you want to see.")
   		acc2 = st.checkbox("Accuracy",key=22)
   		if acc2:
   			accuracy2 = metrics.accuracy_score(y_test2, prediction_test2)
   			st.metric('The accuracy is:', accuracy2)
   		f1sc2 = st.checkbox("F1 score",key=222)
   		if f1sc2:
   			f12 = f1_score(y_test2, prediction_test2, average='weighted')
   			st.metric('F1 score is:',f12)
   		p2 = st.checkbox("Precision score",key=2222)
   		if p2:
   			ps2 = precision_score(y_test2, prediction_test2, average='weighted')
   			st.metric('Precision score is:',ps2)
   		r2 = st.checkbox("Recall score",key=22222)
   		if r2:
   			rs2 = recall_score(y_test2, prediction_test2, average='weighted')
   			st.metric('Recall score is:',rs2)
   		c_val = st.checkbox("Cross Validation",key=222222)
   		if c_val:
   			kfold = st.slider("Please specify the number of folds.",2,50,5)
   			scores = cross_val_score(model_2, X_scaled2, y, cv=kfold)
   			st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

   		st.text("")
   		st.text("")
   		st.subheader("Confusion Matrix")

   		conf = ConfusionMatrixDisplay.from_estimator(model_2, X_test2, y_test2)
   		st.set_option('deprecation.showPyplotGlobalUse', False)
   		st.pyplot(plt.show())


with tab3:
	st.header("Random Forest Classifier")
	predict = st.checkbox("Want to predict? Click here.",key=3000)
	if predict:
		# st.subheader('User Input parameters')
		# st.write(df_new)
		model3 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
		model3.fit(X_scaled, y)
		prediction3 = model3.predict(df_new)
		prediction_proba3 = model3.predict_proba(df_new)
		st.subheader('Prediction Probability')
		st.write(prediction_proba3)
		st.subheader('Prediction')
		if int(prediction3) == 0:
			st.write("**Unlikely to get a stroke.**")
		else:
			st.write("**You are prone to getting a Stroke. Consult a doctor.**")
	click = st.checkbox("Want to train and check how good the selected model is? Click here.",key = 3)
	if click:
   		st.subheader('Training Model')
   		features = st.multiselect('Enter the input features with which you want to train the model',("age","avg_glucose_level","bmi","hypertension","heart_disease","gender","ever_married","work_type","Residence_type"),default=['age','avg_glucose_level','ever_married'],key=30)
   		lst2=[]
   		lst3 = []
   		if 'gender' in features:
   			lst2.append(features.index('gender'))
   			lst3.append('gender')
   		if 'ever_married' in features:
   			lst2.append(features.index('ever_married'))
   			lst3.append('ever_married')
   		if 'work_type' in features:
   			lst2.append(features.index('work_type'))
   			lst3.append('work_type')
   		if 'Residence_type' in features:
   			lst2.append(features.index('Residence_type'))
   			lst3.append('Residence_type')
   		f = tuple(lst3)
   		X_temp = a[features].values
   		y = a.stroke.values
   		ct = ColumnTransformer([(f, OneHotEncoder(), lst2)], remainder = 'passthrough')
   		X_temp = ct.fit_transform(X_temp)

   		X3 = X_temp
   		my_scaler = StandardScaler()
   		my_scaler.fit(X3)
   		X_scaled3 = my_scaler.transform(X3)
   		size = st.number_input('Enter the train and test proportion ratio.', value =0.2, min_value=0.1, max_value=0.9, step=0.1,key=300)
   		X_train3, X_test3, y_train3, y_test3 = train_test_split(X_scaled3,y, test_size=size, random_state = 42)
   		nst = st.number_input('Enter the number of trees in the forest for training',value =10, min_value=1, step=1)

   		model_3 = RandomForestClassifier(max_depth=5, n_estimators=int(nst), max_features=1)
   		model_3.fit(X_train3, y_train3)
   		prediction_test3 = model_3.predict(X_test3)

   		st.subheader("Scores")
   		st.write("Select the score you want to see.")
   		acc3 = st.checkbox("Accuracy",key=33)
   		if acc3:
   			accuracy3 = metrics.accuracy_score(y_test3, prediction_test3)
   			st.metric('The accuracy is:', accuracy3)
   		f1sc3 = st.checkbox("F1 score",key=333)
   		if f1sc3:
   			f13 = f1_score(y_test3, prediction_test3, average='weighted')
   			st.metric('F1 score is:',f13)
   		p3 = st.checkbox("Precision score",key=3333)
   		if p3:
   			ps3 = precision_score(y_test3, prediction_test3, average='weighted')
   			st.metric('Precision score is:',ps3)
   		r3 = st.checkbox("Recall score",key=33333)
   		if r3:
   			rs3 = recall_score(y_test3, prediction_test3, average='weighted')
   			st.metric('Recall score is:',rs3)
   		c_val = st.checkbox("Cross Validation",key=333333)
   		if c_val:
   			kfold = st.slider("Please specify the number of folds.",2,50,5)
   			scores = cross_val_score(model_3, X_scaled3, y, cv=kfold)
   			st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

   		st.text("")
   		st.text("")
   		st.subheader("Confusion Matrix")

   		conf = ConfusionMatrixDisplay.from_estimator(model_3, X_test3, y_test3)
   		st.set_option('deprecation.showPyplotGlobalUse', False)
   		# plt.figure(figsize=(20, 20))
   		st.pyplot(plt.show())

with tab4:
   	st.header("Decision Tree")
   	predict = st.checkbox("Want to predict? Click here.",key=4000)
   	if predict:
   		# st.subheader('User Input parameters')
   		# st.write(df_new)
   		model4 = DecisionTreeClassifier(max_depth=5)
   		model4.fit(X_scaled, y)
   		prediction4 = model4.predict(df_new)
   		prediction_proba4 = model4.predict_proba(df_new)
   		st.subheader('Prediction Probability')
   		st.write(prediction_proba4)
   		st.subheader('Prediction')
   		if int(prediction4) == 0:
   			st.write("**Unlikely to get a stroke.**")
   		else:
   			st.write("**You are prone to getting a Stroke. Consult a doctor.**")

   	click = st.checkbox("Want to train and check how good the selected model is? Click here.",key=4)
   	if click:
   		st.subheader('Training Model')
   		features = st.multiselect('Enter the input features with which you want to train the model',("age","avg_glucose_level","bmi","hypertension","heart_disease","gender","ever_married","work_type","Residence_type"),default=['age','avg_glucose_level','ever_married'],key=40)
   		lst2=[]
   		lst3 = []
   		if 'gender' in features:
   			lst2.append(features.index('gender'))
   			lst3.append('gender')
   		if 'ever_married' in features:
   			lst2.append(features.index('ever_married'))
   			lst3.append('ever_married')
   		if 'work_type' in features:
   			lst2.append(features.index('work_type'))
   			lst3.append('work_type')
   		if 'Residence_type' in features:
   			lst2.append(features.index('Residence_type'))
   			lst3.append('Residence_type')
   		f = tuple(lst3)
   		X_temp = a[features].values
   		y = a.stroke.values
   		ct = ColumnTransformer([(f, OneHotEncoder(), lst2)], remainder = 'passthrough')
   		X_temp = ct.fit_transform(X_temp)
   		
   		X4 = X_temp
   		my_scaler = StandardScaler()
   		my_scaler.fit(X4)
   		X_scaled4 = my_scaler.transform(X4)
   		size = st.number_input('Enter the train and test proportion ratio.', value =0.2, min_value=0.1, max_value=0.9, step=0.1,key=400)
   		X_train4, X_test4, y_train4, y_test4 = train_test_split(X_scaled4,y, test_size=size, random_state = 42)
   		md = st.slider("Set the maximum depth of the tree.",3,100,5)
   		model_4 = DecisionTreeClassifier(max_depth=int(md))
   		model_4.fit(X_train4, y_train4)
   		prediction_test4 = model_4.predict(X_test4)

   		st.subheader("Scores")
   		st.write("Select the score you want to see.")
   		acc4 = st.checkbox("Accuracy",key=44)
   		if acc4:
   			accuracy4 = metrics.accuracy_score(y_test4, prediction_test4)
   			st.metric('The accuracy is:', accuracy4)
   		f1sc4 = st.checkbox("F1 score",key=444)
   		if f1sc4:
   			f14 = f1_score(y_test4, prediction_test4, average='weighted')
   			st.metric('F1 score is:',f14)
   		p4 = st.checkbox("Precision score",key=4444)
   		if p4:
   			ps4 = precision_score(y_test4, prediction_test4, average='weighted')
   			st.metric('Precision score is:',ps4)
   		r4 = st.checkbox("Recall score",key=44444)
   		if r4:
   			rs4 = recall_score(y_test4, prediction_test4, average='weighted')
   			st.metric('Recall score is:',rs4)
   		c_val = st.checkbox("Cross Validation",key=444444)
   		if c_val:
   			kfold = st.slider("Please specify the number of folds.",2,50,5)
   			scores = cross_val_score(model_4, X_scaled4, y, cv=kfold)
   			st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

   		st.text("")
   		st.text("")

   		st.subheader("Confusion Matrix")

   		conf = ConfusionMatrixDisplay.from_estimator(model_4, X_test4, y_test4)
   		st.set_option('deprecation.showPyplotGlobalUse', False)
   		# plt.figure(figsize=(20, 20))
   		st.pyplot(plt.show())

with tab5:
	st.header("Gaussian Naive Bayes")
	predict = st.checkbox("Want to predict? Click here.",key=5000)
	if predict:
   		# st.subheader('User Input parameters')
   		# st.write(df_new)
   		model5 = GaussianNB()
   		model5.fit(X_scaled, y)
   		prediction5 = model5.predict(df_new)
   		prediction_proba5= model5.predict_proba(df_new)
   		st.subheader('Prediction Probability')
   		st.write(prediction_proba5)
   		st.subheader('Prediction')
   		if int(prediction5) == 0:
   			st.write("**Unlikely to get a stroke.**")
   		else:
   			st.write("**You are prone to getting a Stroke. Consult a doctor.**")

	click = st.checkbox("Want to train and check how good the selected model is? Click here.",key=5)
	if click:
   		st.subheader('Training Model')
   		features = st.multiselect('Enter the input features with which you want to train the model',("age","avg_glucose_level","bmi","hypertension","heart_disease","gender","ever_married","work_type","Residence_type"),default=['age','avg_glucose_level','ever_married'],key=50)
   		lst2=[]
   		lst3 = []
   		if 'gender' in features:
   			lst2.append(features.index('gender'))
   			lst3.append('gender')
   		if 'ever_married' in features:
   			lst2.append(features.index('ever_married'))
   			lst3.append('ever_married')
   		if 'work_type' in features:
   			lst2.append(features.index('work_type'))
   			lst3.append('work_type')
   		if 'Residence_type' in features:
   			lst2.append(features.index('Residence_type'))
   			lst3.append('Residence_type')
   		f = tuple(lst3)
   		X_temp = a[features].values
   		y = a.stroke.values
   		ct = ColumnTransformer([(f, OneHotEncoder(), lst2)], remainder = 'passthrough')
   		X_temp = ct.fit_transform(X_temp)
   		
   		X5 = X_temp
   		my_scaler = StandardScaler()
   		my_scaler.fit(X5)
   		X_scaled5 = my_scaler.transform(X5)
   		size = st.number_input('Enter the train and test proportion ratio.', value =0.2, min_value=0.1, max_value=0.9, step=0.1,key=500)
   		X_train5, X_test5, y_train5, y_test5 = train_test_split(X_scaled5,y, test_size=size, random_state = 42)

   		model_5 = GaussianNB()
   		model_5.fit(X_train5, y_train5)
   		prediction_test5 = model_5.predict(X_test5)

   		st.subheader("Scores")
   		st.write("Select the score you want to see.")
   		acc5 = st.checkbox("Accuracy",key=55)
   		if acc5:
   			accuracy5 = metrics.accuracy_score(y_test5, prediction_test5)
   			st.metric('The accuracy is:', accuracy5)
   		f1sc5 = st.checkbox("F1 score",key=555)
   		if f1sc5:
   			f15= f1_score(y_test5, prediction_test5, average='weighted')
   			st.metric('F1 score is:',f15)
   		p5 = st.checkbox("Precision score",key=5555)
   		if p5:
   			ps5 = precision_score(y_test5, prediction_test5, average='weighted')
   			st.metric('Precision score is:',ps5)
   		r5 = st.checkbox("Recall score",key=55555)
   		if r5:
   			rs5 = recall_score(y_test5, prediction_test5, average='weighted')
   			st.metric('Recall score is:',rs5)
   		c_val = st.checkbox("Cross Validation",key=555555)
   		if c_val:
   			kfold = st.slider("Please specify the number of folds.",2,50,5)
   			scores = cross_val_score(model_5, X_scaled5, y, cv=kfold)
   			st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

   		st.text("")
   		st.text("")
   		st.subheader("Confusion Matrix")

   		conf = ConfusionMatrixDisplay.from_estimator(model_5, X_test5, y_test5)
   		st.set_option('deprecation.showPyplotGlobalUse', False)
   		# plt.figure(figsize=(20, 20))
   		st.pyplot(plt.show())

with tab6:
   	st.header("Multi-layer Perceptron Classifier")
   	predict = st.checkbox("Want to predict? Click here.",key=6000)
   	if predict:
   		# st.subheader('User Input parameters')
   		# st.write(df_new)
   		model6 = MLPClassifier(alpha=1, max_iter=1000)
   		model6.fit(X_scaled, y)
   		prediction6 = model6.predict(df_new)
   		prediction_proba6 = model6.predict_proba(df_new)
   		st.subheader('Prediction Probability')
   		st.write(prediction_proba6)
   		st.subheader('Prediction')
   		if int(prediction6) == 0:
   			st.write("**Unlikely to get a stroke.**")
   		else:
   			st.write("**You are prone to getting a Stroke. Consult a doctor.**")	

   	click = st.checkbox("Want to train and check how good the selected model is? Click here.",key=6)
   	if click:
   		st.subheader('Training Model')
   		features = st.multiselect('Enter the input features with which you want to train the model',("age","avg_glucose_level","bmi","hypertension","heart_disease","gender","ever_married","work_type","Residence_type"),default=['age','avg_glucose_level','ever_married'],key=30)
   		lst2=[]
   		lst3 = []
   		if 'gender' in features:
   			lst2.append(features.index('gender'))
   			lst3.append('gender')
   		if 'ever_married' in features:
   			lst2.append(features.index('ever_married'))
   			lst3.append('ever_married')
   		if 'work_type' in features:
   			lst2.append(features.index('work_type'))
   			lst3.append('work_type')
   		if 'Residence_type' in features:
   			lst2.append(features.index('Residence_type'))
   			lst3.append('Residence_type')
   		f = tuple(lst3)
   		X_temp = a[features].values
   		y = a.stroke.values
   		ct = ColumnTransformer([(f, OneHotEncoder(), lst2)], remainder = 'passthrough')
   		X_temp = ct.fit_transform(X_temp)
   		
   		X6 = X_temp
   		my_scaler = StandardScaler()
   		my_scaler.fit(X6)
   		X_scaled6 = my_scaler.transform(X6)
   		size = st.number_input('Enter the train and test proportion ratio.', value =0.2, min_value=0.1, max_value=0.9, step=0.1,key=600)
   		X_train6, X_test6, y_train6, y_test6 = train_test_split(X_scaled6,y, test_size=size, random_state = 42)

   		act = st.radio("Select the activation function for training the nueral network",("Identity","Logistic","Tanh","Relu"))
   		if act=="Relu":
   			activation = 'relu'
   		elif act == "Identity":
   			activation = 'identity'
   		elif act=="Logistic":
   			activation = 'logistic'
   		elif act == "Tanh":
   			activation = 'tanh'
   		model_6 = MLPClassifier(alpha=1, max_iter=1000, activation = activation)
   		model_6.fit(X_train6, y_train6)
   		prediction_test6 = model_6.predict(X_test6)

   		st.subheader("Scores")
   		st.write("Select the score you want to see.")
   		acc6 = st.checkbox("Accuracy",key=66)
   		if acc6:
   			accuracy6 = metrics.accuracy_score(y_test6, prediction_test6)
   			st.metric('The accuracy is:', accuracy6)
   		f1sc6 = st.checkbox("F1 score",key=666)
   		if f1sc6:
   			f16 = f1_score(y_test6, prediction_test6, average='weighted')
   			st.metric('F1 score is:',f16)
   		p6 = st.checkbox("Precision score",key=6666)
   		if p6:
   			ps6 = precision_score(y_test6, prediction_test6, average='weighted')
   			st.metric('Precision score is:',ps6)
   		r6 = st.checkbox("Recall score",key=66666)
   		if r6:
   			rs6 = recall_score(y_test6, prediction_test6, average='weighted')
   			st.metric('Recall score is:',rs6)
   		c_val = st.checkbox("Cross Validation",key=666666)
   		if c_val:
   			kfold = st.slider("Please specify the number of folds.",2,50,5)
   			scores = cross_val_score(model_6, X_scaled6, y, cv=kfold)
   			st.write("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

   		st.text("")
   		st.text("")
   		st.subheader("Confusion Matrix")

   		conf = ConfusionMatrixDisplay.from_estimator(model_6, X_test6, y_test6)
   		st.set_option('deprecation.showPyplotGlobalUse', False)
   		# plt.figure(figsize=(20, 20))
   		st.pyplot(plt.show())



