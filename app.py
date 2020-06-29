import streamlit as st
import pandas as pd
import numpy as np
import base64

from sklearn.tree import DecisionTreeClassifier
import webbrowser


def main():
	rtml_temp = """
	<div class='pm-button'><a href='https://www.payumoney.com/paybypayumoney/#/9D1CFC52C70BB8FE8DC0C2A11A3B5CFF'target="_blank"><img src='https://www.payumoney.com/media/images/payby_payumoney/new_buttons/23.png' /></a></div>  
	"""
	html_temp = """
	<a href="https://www.linkedin.com/in/abhishek-vaid-78505811b/" target="_blank"><button>View my Linkedin</button></a>
	</div>
	"""
	gtml_temp = """
	<iframe width="560" height="315" src="https://www.youtube.com/embed/vKjzSq7njLw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
	</div>
	"""
	ftml_temp = """
	<a href="https://abhishekvaid13.github.io/Abhishek_portfolio_3/" target="_blank"><button>View my Portfolio</button></a>
	</div>
	"""
	jtml_temp = """
	<a href="https://www.thelifeyoucansave.org/best-charities/" target="_blank"><button>Go to - Life you can save</button></a>
	</div>
	"""
	ktml_temp = """
	<a href="https://www.facebook.com/aspecialschool/" target="_blank"><button>Go to - Special Mentions</button></a>
	</div>
	"""
	ltml_temp = """
	<a href="https://www.change.org/" target="_blank"><button>Go to Change.org</button></a>
	</div>
	"""
	st.markdown("""
	<style>
		body {
    	color: #DC2F0A;
    	background-image: url(https://mail.google.com/mail/u/0?ui=2&ik=17b7366792&attid=0.1&permmsgid=msg-a:r5961599422475648997&th=172f212637dbf15b&view=fimg&sz=s0-l75-ft&attbid=ANGjdJ8SYmsTP9nSpj8ZWBEEF6wbJllgLQTNXxNodHKnzhr91UvhF4CZIJVog1Vfs4N7uj4OqeWV4qA1rfjR-x8wm3ae-9EJjKKKaMYDEZxvoZ6xoGx-7F-WIfwc-F0&disp=emb&realattid=ii_kbwlnp270);
    	etc.
		}
	</style>
    """, unsafe_allow_html=True)
	st.title("Automated Machine Learning Application")



	st.sidebar.title("Select the type of Machine Learning Algorithm required")
	activities = ["Regression Machine Learning","Classification Machine Learning","Ensembles Machine Learning"]
	choice = st.sidebar.selectbox("click box below",activities)






	if choice == 'Classification Machine Learning':
		st.subheader("Your machine is ready to learn!!!")
		Test_data = st.file_uploader("Upload a Testing Dataset", type=["csv", "txt", "xlsx"])
		if Test_data is not None:
			tf = pd.read_csv(Test_data)
			st.dataframe(tf.head())




		Train_data = st.file_uploader("Upload a Training Dataset", type=["csv", "txt", "xlsx"])
		if Train_data is not None:
			df = pd.read_csv(Train_data)
			st.dataframe(df.head())





			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select type of Classification Algorithm",["SVM","RandomForestClassifier","Naive Bayes Algorithm","K-Nearest Neighbors Classifier","Logistic regression"])
			Train_Data_Columns = st.multiselect("Select Columns To Train",all_columns_names)
			Target_Data_Columns = st.text_input("Select Target Columns","Type Here")
			if st.button("TRAIN DATA"):
				st.success("Generating prediction using {} for {} target column".format(type_of_plot,Target_Data_Columns))
				Target_data = df[Target_Data_Columns]
				Train_data = df[Train_Data_Columns]
				Test_data = tf[Train_Data_Columns]
				cateogry_columns=Train_data.select_dtypes(include=['object']).columns.tolist()
				integer_columns=Train_data.select_dtypes(include=['int64','float64']).columns.tolist()
				for columns in Train_data:
    					if Train_data[columns].isnull().any():
        					if(columns in cateogry_columns):
            						Train_data[columns]=Train_data[columns].fillna(Train_data[columns].mode()[0])
        					else:
            						Train_data[columns]=Train_data[columns].fillna(Train_data[columns].mean())

				cateogry_columns=Test_data.select_dtypes(include=['object']).columns.tolist()
				integer_columns=Test_data.select_dtypes(include=['int64','float64']).columns.tolist()
				Train_data['train']=1
				Test_data['train']=0
				for columns in Test_data:
    					if Test_data[columns].isnull().any():
        					if(columns in cateogry_columns):
            						Test_data[columns]=Test_data[columns].fillna(Test_data[columns].mode()[0])
        					else:
            						Test_data[columns]=Test_data[columns].fillna(Test_data[columns].mean())
				combined = pd.concat([Train_data,Test_data])
				kf = pd.get_dummies(combined[Train_Data_Columns])
				combined = pd.concat([combined['train'],kf],axis=1)
				train_df = combined[combined['train']==1]
				test_df = combined[combined["train"]==0]
				train_df.drop(["train"], axis=1, inplace=True)
				test_df.drop(["train"], axis=1, inplace=True)




				def get_table_download_link(submission):
					"""Generates a link allowing the data in a given panda dataframe to be downloaded
					in:  dataframe
					out: href string
					"""
					csv = submission.to_csv(index=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">***Download csv file***</a>'

				if type_of_plot == 'SVM':
					from sklearn.svm import SVC
					Train_data = train_df
					clf = SVC()
					kz = clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'RandomForestClassifier':
					from sklearn.ensemble import RandomForestClassifier
					Train_data = train_df
					clf = RandomForestClassifier()
					kz = clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Logistic regression':
					from sklearn.linear_model import LogisticRegression
					Train_data = train_df
					clf = LogisticRegression()
					kz = clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Naive Bayes Algorithm':
					from sklearn.naive_bayes import GaussianNB
					Train_data = train_df
					clf = GaussianNB()
					kz = clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'K-Nearest Neighbors Classifier':
					from sklearn.neighbors import KNeighborsClassifier
					import math
					Train_data = train_df
					total_rows = Train_data.shape[0]
					k = int(math.sqrt(total_rows)/2)
					clf = KNeighborsClassifier(n_neighbors=k)
					kz = clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)




	elif choice == 'Regression Machine Learning':
		st.subheader("Your machine is ready to learn!!!")
		Test_data = st.file_uploader("Upload a Testing Dataset", type=["csv", "txt", "xlsx"])
		if Test_data is not None:
			tf = pd.read_csv(Test_data)
			st.dataframe(tf.head())
		Train_data = st.file_uploader("Upload a Training Dataset", type=["csv", "txt", "xlsx"])
		if Train_data is not None:
			df = pd.read_csv(Train_data)
			st.dataframe(df.head())


			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Regression Algorithm",["LinearRegression","Support Vector Regression SVR","Lasso Regression"])
			Train_Data_Columns = st.multiselect("Select Columns To Train",all_columns_names)
			Target_Data_Columns = st.text_input("Select Target Columns","Type Here")
			if st.button("TRAIN DATA"):
				st.success("Generating prediction using {} for {} target column".format(type_of_plot,Target_Data_Columns))
				Target_data = df[Target_Data_Columns]
				Train_data = df[Train_Data_Columns]
				Test_data = tf[Train_Data_Columns]
				cateogry_columns=Train_data.select_dtypes(include=['object']).columns.tolist()
				integer_columns=Train_data.select_dtypes(include=['int64','float64']).columns.tolist()
				for columns in Train_data:
    					if Train_data[columns].isnull().any():
        					if(columns in cateogry_columns):
            						Train_data[columns]=Train_data[columns].fillna(Train_data[columns].mode()[0])
        					else:
            						Train_data[columns]=Train_data[columns].fillna(Train_data[columns].mean())

				cateogry_columns=Test_data.select_dtypes(include=['object']).columns.tolist()
				integer_columns=Test_data.select_dtypes(include=['int64','float64']).columns.tolist()
				Train_data['train']=1
				Test_data['train']=0
				for columns in Test_data:
    					if Test_data[columns].isnull().any():
        					if(columns in cateogry_columns):
            						Test_data[columns]=Test_data[columns].fillna(Test_data[columns].mode()[0])
        					else:
            						Test_data[columns]=Test_data[columns].fillna(Test_data[columns].mean())
				Train_data['train']=1
				Test_data['train']=0
				combined = pd.concat([Train_data,Test_data])
				kf = pd.get_dummies(combined[Train_Data_Columns])
				combined = pd.concat([combined['train'],kf],axis=1)
				train_df = combined[combined['train']==1]
				test_df = combined[combined["train"]==0]
				train_df.drop(["train"], axis=1, inplace=True)
				test_df.drop(["train"], axis=1, inplace=True)


				def get_table_download_link(submission):
					"""Generates a link allowing the data in a given panda dataframe to be downloaded
					in:  dataframe
					out: href string
					"""
					csv = submission.to_csv(index=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">***Download csv file***</a>'


				if type_of_plot == 'LinearRegression':
					from sklearn.linear_model import LinearRegression
					Train_data = train_df
					regressor = LinearRegression()
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Support Vector Regression SVR':
					from sklearn.svm import SVR
					Train_data = train_df
					regressor = SVR()
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Lasso Regression':
					from sklearn.linear_model import Lasso
					Train_data = train_df
					regressor = Lasso()
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)


	elif choice == 'Ensembles Machine Learning':
		st.subheader("Your machine is ready to learn!!!")
		Test_data = st.file_uploader("Upload a Testing Dataset", type=["csv", "txt", "xlsx"])
		if Test_data is not None:
			tf = pd.read_csv(Test_data)
			st.dataframe(tf.head())
		Train_data = st.file_uploader("Upload a Training Dataset", type=["csv", "txt", "xlsx"])
		if Train_data is not None:
			df = pd.read_csv(Train_data)
			st.dataframe(df.head())


			all_columns_names = df.columns.tolist()
			type_of_plot = st.selectbox("Select Type of Ensembles Algorithm",["XGBOOST Regressor","XGBOOST Classifier","Random Forest Classification","AdaBoost Classifier","AdaBoost Regressor","Gradient Boosting Classifier","Gradient Boosting Regressor","CatBoost Classifier","CatBoost Regressor","Bagging Classifier","Bagging Regressor"])
			Train_Data_Columns = st.multiselect("Select Columns To Train",all_columns_names)
			Target_Data_Columns = st.text_input("Select Target Columns","Type Here")
			if st.button("TRAIN DATA"):
				st.success("Generating prediction using {} for {} target column".format(type_of_plot,Target_Data_Columns))
				Target_data = df[Target_Data_Columns]
				Train_data = df[Train_Data_Columns]
				Test_data = tf[Train_Data_Columns]
				cateogry_columns=Train_data.select_dtypes(include=['object']).columns.tolist()
				integer_columns=Train_data.select_dtypes(include=['int64','float64']).columns.tolist()
				for columns in Train_data:
    					if Train_data[columns].isnull().any():
        					if(columns in cateogry_columns):
            						Train_data[columns]=Train_data[columns].fillna(Train_data[columns].mode()[0])
        					else:
            						Train_data[columns]=Train_data[columns].fillna(Train_data[columns].mean())

				cateogry_columns=Test_data.select_dtypes(include=['object']).columns.tolist()
				integer_columns=Test_data.select_dtypes(include=['int64','float64']).columns.tolist()

				for columns in Test_data:
    					if Test_data[columns].isnull().any():
        					if(columns in cateogry_columns):
            						Test_data[columns]=Test_data[columns].fillna(Test_data[columns].mode()[0])
        					else:
            						Test_data[columns]=Test_data[columns].fillna(Test_data[columns].mean())

				Train_data['train']=1
				Test_data['train']=0
				combined = pd.concat([Train_data,Test_data])
				kf = pd.get_dummies(combined[Train_Data_Columns])
				combined = pd.concat([combined['train'],kf],axis=1)
				train_df = combined[combined['train']==1]
				test_df = combined[combined["train"]==0]
				train_df.drop(["train"], axis=1, inplace=True)
				test_df.drop(["train"], axis=1, inplace=True)


				def get_table_download_link(submission):
					"""Generates a link allowing the data in a given panda dataframe to be downloaded
					in:  dataframe
					out: href string
					"""
					csv = submission.to_csv(index=False)
					b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
					return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">***Download csv file***</a>'


				if type_of_plot == 'XGBOOST Regressor':
					import xgboost as xgb
					Train_data = train_df
					regressor = xgb.XGBRegressor()
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'XGBOOST Classifier':
					import xgboost as xgb
					Train_data = train_df
					Clf = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
					kz = Clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Random Forest Classification':
					from sklearn.ensemble import RandomForestClassifier
					Train_data = train_df
					Clf = RandomForestClassifier()
					kz = Clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'AdaBoost Classifier':
					from sklearn.ensemble import AdaBoostClassifier
					Train_data = train_df
					Clf = AdaBoostClassifier(random_state=1)
					kz = Clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'AdaBoost Regressor':
					from sklearn.ensemble import AdaBoostRegressor
					Train_data = train_df
					regressor = AdaBoostRegressor()
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Gradient Boosting Classifier':
					from sklearn.ensemble import GradientBoostingClassifier
					Train_data = train_df
					Clf = GradientBoostingClassifier(learning_rate=0.01,random_state=1)
					kz = Clf.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Gradient Boosting Regressor':
					from sklearn.ensemble import GradientBoostingRegressor
					Train_data = train_df
					regressor = GradientBoostingRegressor()
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'CatBoost Classifier':
					st.subheader('CatBoost takes longer to process, thanks for your patience')
					from catboost import CatBoostClassifier
					Train_data = train_df
					Clf = CatBoostClassifier()
					categorical_var = np.where(Train_data.dtypes != np.float)[0]
					kz = Clf.fit(Train_data, Target_data, cat_features = categorical_var,plot=False)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'CatBoost Regressor':
					st.subheader('CatBoost takes longer to process, thanks for your patience')
					from catboost import CatBoostRegressor
					Train_data = train_df
					regressor = CatBoostRegressor()
					categorical_var = np.where(Train_data.dtypes != np.float)[0]
					kz = regressor.fit(Train_data, Target_data, cat_features = categorical_var,plot=False)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Bagging Classifier':
					from sklearn.ensemble import BaggingClassifier
					from sklearn import tree
					Train_data = train_df
					regressor = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

				if type_of_plot == 'Bagging Regressor':
					from sklearn.ensemble import BaggingRegressor
					from sklearn import tree
					Train_data = train_df
					regressor = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
					kz = regressor.fit(Train_data, Target_data)
					prediction = kz.predict(test_df)
					submission = pd.DataFrame({Target_Data_Columns: prediction})
					st.subheader('Target Data Columns')
					st.write(Target_data)
					st.subheader('Prediction')
					st.write(submission)
					st.markdown(get_table_download_link(submission), unsafe_allow_html=True)

	
	st.sidebar.title("Video explanation to use the tool")
	activities = ["Select an option", "Best way to use this tool"]
	choice = st.sidebar.selectbox("click box below",activities)
	if choice == "Best way to use this tool":
		st.subheader("I have created this tool for people who want to do machine learning, and get to a prediction within a matter of 1-2 minutes and save a lot of time.")
		st.markdown(gtml_temp,unsafe_allow_html=True)
		
	st.sidebar.title("Get the Machine Learning code of this application")
	activities = ["Select an option", "Code for your education"]
	choice = st.sidebar.selectbox("click box below",activities)
	if choice == "Code for your education":
		st.subheader("You can get code of this application for your education for only Rs 499. I would personally send you an email with the complete code within 24 hours of the payment confirmation")
		st.markdown(rtml_temp,unsafe_allow_html=True)
		
	st.sidebar.title("FOR GOOD")
	activities = ["Select an option", "Change.ORG","Donate to your Favourite Charitable Organization"]
	choice = st.sidebar.selectbox("click box below",activities)

	if choice == "Change.ORG":
		st.subheader("Be the Change you want to see in the world")
		st.markdown(ltml_temp,unsafe_allow_html=True)

	if choice == "Donate to your Favourite Charitable Organization":
		st.subheader("Everylife counts, do your bit")
		st.markdown(jtml_temp,unsafe_allow_html=True)
		st.markdown(ktml_temp,unsafe_allow_html=True)


	st.sidebar.title("About the Maker")
	activities = ["Select an option", "My portfolio"]
	choice = st.sidebar.selectbox("click box below",activities)
	if choice == "My portfolio":
		st.subheader("Hi I am Abhishek Vaid, a budding Data Scientist")
		st.markdown(ftml_temp,unsafe_allow_html=True)
		st.markdown(html_temp,unsafe_allow_html=True)
		
	






if __name__ == '__main__':
	main()
