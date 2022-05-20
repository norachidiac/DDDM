#Importing Libraries
from asyncore import write
from ctypes import alignment
from tkinter import CENTER
import streamlit as st
st.set_page_config(layout='wide')
import streamlit as st
import numpy as np
import matplotlib.pylab as plt
from streamlit_option_menu import option_menu
#Creating Sidebar with 3 Options
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home Page", "EDA", "Machine Learning"],
    )
with st.sidebar.header('Upload your CSV data'):
        uploaded_file= st.sidebar.file_uploader('Upload your input CSV file')
#Filling Homepage Dashboard
if selected=="Home Page":

    st.markdown("<h1 style='text-align: left; color: purple;'>Black Friday Sales Prediction</h1>", unsafe_allow_html=True)
    st.image("https://www.redcarpet-fashionawards.com/wp-content/uploads/2020/11/41513-posts.article_md.jpg")
    st.subheader("Objective: To understand the customer purchase behavior (specifically, purchase amount). After analyzing the dataset, we are going to build a model to predict the purchase amount of customers which will help the company to create personalized offers for customers against different products.")


#Filling the EDA Dashboard
if selected=="EDA":
    st.header("Exploratory Data Analysis")
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv('/Users/noura/Desktop/train.csv')
    st.markdown("**1.1 First five headers:**")
    test_df = pd.read_csv('/Users/noura/Desktop/test.csv')
    st.code(df.head())
    st.write("**1.2 Dataset is made up of 12 columns and 550068 rows**")
    st.code(df.shape)
    st.code(df.info())
    st.write("**1.3 Checking for missing values**")
    st.code(df.isnull().sum())
    variables = st.sidebar.multiselect("List of Features", df.columns)
    st.write("You selected these variables", variables)
    st.write("**1.4 Heatmap showing the correlation between the features**")
    num_df = df.copy()

    col = ['Product_ID','Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
    for i in col:
        num_df[i] = num_df[i].astype('category').cat.codes


    corr = num_df.corr(method='pearson')
    fig, ax = plt.subplots()
    plt.figure(figsize=(13,8))
    sns.heatmap(corr, annot=True,center=0)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("**1.5 Visualizations showing the Analysis of Purchase with Gender**")
    def Plot_1(feature): 
    
        fig,axes = plt.subplots(2,2,figsize=(15,12))
        fig.suptitle('Analysis of Purchase with ' + feature)
        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        sns.set_style('darkgrid')

    
        for i in df[feature].unique():
        
        #histogram
            sns.distplot(a=df.loc[df[feature]==i]['Purchase'],ax = axes[0][0],label=i, kde=False)
        
        #kde_plot
            sns.kdeplot(data=df.loc[df[feature]==i]['Purchase'],ax = axes[0][1],label=i)
                
        
        axes[0][0].set_title('Histogram')
        axes[0][1].set_title('KDE plot')
    
    #piechart
        temp_df = df.groupby([feature]).sum().reset_index().sort_values('Purchase',ascending=False)
        colors = sns.color_palette('pastel')
        axes[1][0].pie(temp_df['Purchase'], labels = temp_df[feature], colors = colors, autopct='%.0f%%')
        axes[1][0].set_title('Percentage of Purchases by ' + feature)


    #boxplot
        sns.boxplot(y=df['Purchase'],x=df[feature],ax=axes[1][1])
        axes[1][1].set_title('Box plot of Purchase by '+  feature)

        plt.legend()
    st.pyplot(Plot_1('Gender'))
    st.write("**1.6 Analysis of Purchase with Age**")
    def Plot_2(feature):
    
        fig,axes = plt.subplots(2,2,figsize=(15,12))
        fig.suptitle('Analysis of Purchase with ' + feature)
        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        sns.set_style('darkgrid')


    #sns.barplot
        sns.barplot(x=feature,y='Purchase',data=df.groupby([feature]).sum().reset_index().sort_values(feature),palette='plasma',ax=axes[0][0])
        axes[0][0].set_title('Barplot - Sum')

        sns.barplot(x=feature,y='Purchase',data=df.groupby([feature]).mean().reset_index().sort_values(feature),palette='plasma',ax=axes[0][1])
        axes[0][1].set_title('Barplot - Mean')

    #sns.boxplot
        sns.boxplot(y=df['Purchase'],x=df[feature],ax=axes[1][1])
        axes[1][1].set_title('Boxplot')


    #plt.pie chart!
        temp_df = df.groupby([feature]).sum().reset_index().sort_values(feature,ascending=False)
        colors = sns.color_palette('pastel')#[0:7]
        axes[1][0].pie(temp_df['Purchase'], labels = temp_df[feature], colors = colors, autopct='%.0f%%' ,pctdistance=0.8)
        axes[1][0].set_title('Percentage of Purchases by ' + feature)

    st.pyplot(Plot_2('Age'))

#Filling Machine Learning Dashboard
if selected=="Machine Learning":
    st.header("Pre-Processing")
    import streamlit as st
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv('/Users/noura/Desktop/train.csv')
    test_df = pd.read_csv('/Users/noura/Desktop/test.csv')
    def remove_outlier(df_in, col_name):
        q1 = df_in[col_name].quantile(0.25)
        q3 = df_in[col_name].quantile(0.75)
        iqr = q3-q1    
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
        return df_out
    df_clean = remove_outlier(df,'Purchase')

    df_clean.reset_index(drop=True)
    # Replacing ''P00'' with no value and scaling the ProductID column. 
    df['Product_ID'] = df['Product_ID'].str.replace('P00', '')
    ss = StandardScaler()
    df['Product_ID'] = ss.fit_transform(df['Product_ID'].values.reshape(-1, 1))
    # Checking for missing values
    st.write("**1.1 Removing missing Values**")
    st.code(df.isnull().sum())
    st.write("After removing missing values")
    df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mean())
    # The missing data in the product category 2 column have been imputed using mean.
    df.drop(['Product_Category_3'],axis=1,inplace=True)
    # There are more than 50 percent missing values present in the Product_category_column so we will drop that column.
    st.code(df.isnull().sum())
    # As we can see the missing values have been successfully imputed and now there are no null values present in the dataset.
    # The label encoding technique will now replace all the categorical variables to numeric for easier computation.
    from sklearn.preprocessing import LabelEncoder
    cat_cols=['Gender','City_Category','Age']
    le=LabelEncoder()
    for i in cat_cols:
        df[i]=le.fit_transform(df[i])
    st.write("**1.2 Replacing Categorical variables to Numeric**")
    st.code(df.dtypes)
    df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].replace('4+','4')
    # Values in the Stay_In_Current_City_Years column has been changed from 4+ to 4
    df['Gender']=df['Gender'].astype(int)
    df['Age']=df['Age'].astype(int)
    df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
    # The gender, Age and Stay_In_Current_City_Years values are changed to integer types
    df['City_Category']=df['City_Category'].astype('category')
    # The type of city_category has been changed from int to category.
    st.code(df)
    df['Purchase']=np.log(df['Purchase'])
    # The log transformation will help us transform the data and change the data to normal distribution
    df= pd.get_dummies(df)
    df.head()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # The get_dummies() function is used to convert categorical variable into dummy/indicator variables.
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import metrics
    from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
    st.header("Machine Learning Models")
    X=df.drop(labels=['Purchase'],axis=1)                         
    Y=df['Purchase']
    X.head()
    # The data is split into X and Y where independent and dependent variables have been separated
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
    # The data has been split into Train and test.
    st.write("**2.1 Linear Regression**")
    model=LinearRegression()
    model.fit(X_train,Y_train)
    # Predicting on X_test
    Y_predict=model.predict(X_test)
    score=r2_score(Y_test,Y_predict)
    mae=mean_absolute_error(Y_test,Y_predict)
    mse=mean_squared_error(Y_test,Y_predict)
    rmse=(np.sqrt(mean_squared_error(Y_test,Y_predict)))
    st.write('r2_score: ',score)
    st.write('mean_absolute_error: ',mae)
    st.write('mean_squared_error: ',mse)
    st.write('root_mean_squared_error: ',rmse)
    st.write("The r2_score is only 0.20 and the Root mean squared error is high so the model is not very accurate to predict the purchases or the target column")
    st.write("**2.2 Decision Tree Regressor**")
    DT=DecisionTreeRegressor(max_depth=9)
    DT.fit(X_train,Y_train)
    #predicting train
    train_preds=DT.predict(X_train)
    #predicting on test
    test_preds=DT.predict(X_test)
    RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds)))
    RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds)))
    st.write("RMSE TrainingData = ",str(RMSE_train))
    st.write("RMSE TestData = ",str(RMSE_test))
    st.write('-'*50)
    st.write('RSquared value on train:',DT.score(X_train, Y_train))
    st.write('RSquared value on test:',DT.score(X_test, Y_test))
    st.write("The Decision Tree Regressor is better compared to Linear regression as it can be observed that the root mean square error is less as compared to the previous model and the RSquared value is higher in this model.")
    st.write("**2.3 Random Forest Regressor**")
    RF=RandomForestRegressor().fit(X_train,Y_train)
    #predicting train
    train_preds1=RF.predict(X_train)
    #predicting on test
    test_preds1=RF.predict(X_test)
    RMSE_train=(np.sqrt(metrics.mean_squared_error(Y_train,train_preds1)))
    RMSE_test=(np.sqrt(metrics.mean_squared_error(Y_test,test_preds1)))
    st.write("RMSE TrainingData = ",str(RMSE_train))
    st.write("RMSE TestData = ",str(RMSE_test))
    st.write('-'*50)
    st.write('RSquared value on train:',RF.score(X_train, Y_train))
    st.write('RSquared value on test:',RF.score(X_test, Y_test))
    st.write("The Random Forest regressor model is again better than the previous model as we have a lower root mean square error value and the Rsquared value is higher than the previous model.")
    st.write("**2.4 Model on Test Dataset**")
    test_df
    st.write("Checking for missing values present in the test dataset.")
    st.code(test_df.isnull().sum())
    st.write("As random forest regressor performed very well compared to linear regression and decision tree regressor model. Random forest regressor model has been used to predict on our test dataset.")
   









