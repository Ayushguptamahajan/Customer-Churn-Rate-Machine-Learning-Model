# Customer-Churn-Rate-Machine-Learning-Model
Churn prediction is predicting which customers are at high risk of leaving your company or canceling a subscription to a service, based on their behavior with your product.It is a critical prediction for many businesses because acquiring new clients often costs more than retaining existing ones.

**The summary of the steps executed in ML model are:**

**1. Data collection and overview of the dataframe:**
- The data was extracted from csv file downloaded from kaggle.
- Data consist of 7043 rows and 21 columns.
- Intuition of the data was gained via calling the first five and last five rows.
- There were 0.15% Nan values present in TotalCharges feature of given dataset.
- There were no duplicated value present in the given dataset.
- Datatype of TotalCharges changed to float from string.
- Feature customerID was dropped out since its non informative and there is no particular trend in the customerID feature which may help in model building or prediction.
- Function was created to get value_counts corresponding to unique categories. In many feature categories was present having same meaning. The same featue categories was reduced.

**2. Exploratory Data Analysis:**
- Univariate Analysis:
    - Dist and QQ plot was ploted for numerical columns which depicted none of the numerical columns are normally distributed.
    - Countplot of categorical features was plotted which provide following insights:
        - There are more male having telecom connection compared to female.
        - Most people have opt fibre optic as internet service.
        - Maximum people have not opt Multipleline, OnlineSecurity, device protection and StreamingTV services.
        - Maximum people have month to month connection.
        - Since the churned customer are very less which imply the data is highly imbalanced.The imbalance nature of data can be balanced to some extend by using following method:
            1. Reshuffle the data.
            2. Collect more Data.
            3.Use precision and recall as accuracy metrices.
            
-  Bivariate and Multivariate Analysis:
    - Histplot was plotted of numerical column with bin size of 50 against the targeted feature.Following insights were gained from the plotted histplot:
        - With increase in tenure the count of churned customer was less which implies churned customer have not stayed for long time.
        - When Monthly charge was between 70 to 100, the no of churned customer was more
        - More churned customer has paid very less charges before quiting the telecom service.
    - Countplot was plotted for the categorical feature with respect to target feature (Churn). FOllowing insights were gained from the plotted countplots:
        - Female has churned more compared to male.
        - Partner connection has churned less.
        - Dependents connection has churned more.Reason may be that they were not able to get money from their dependent source to sustain.
        - Customer having Phone Service connection has churned more.
        - MultipleLine connection has churned less 
        - People with Fibre Optic has churned more.
        - People with no online security, online backup, Device protection,Techsupport,StreamingTV/Movies option and has churned more.
        - Month-to-month suscribed customer has churned more.
    - Heat map for correlation matrix of the dataset was viewed. Following insights were gained:
        - Tenure and TotalCharges was having strong correlation.
        - Monthly charges and TotalCharges was having strong correlation. (as the case should be)
    - Pairplot was plotted for numerical feature with respect to target feature (Churn).
    - Box plot was plotted for checking the outlier present in the numerical features.Since the data was non uniform, IQR method was used to check for the outlier present in the numerical columns of the given dataset.(Function for checking programmatically outlier in dataset was also created. The resultant of boxplot and function was that there were no outlier present in the given dataset.
    
**3. Feature Engineering:**
    Pipeline was created for all the feature engineering works for the dataframe for ease of deployment of model.Step of same is as follow:
- **Step1 Missing Value Imputation**: The missing value in the TotalCharges was imputed with Iterative imputation method by creating a function named NumericalImputationMICE. 
- **Step2 Encoding of categorical columns**: Columntransformer object was created for Onehotencoding of 'gender','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaperlessBilling' and 'PaymentMethod'.   
- **Step3 Scaling**: Scaling of all the columns after encoding was done in order to get efficient end result of the final model.
    
**5. Model Building:**

- Train test split was performed with test_size as 20% and random state of 0.2.
- Further **Five** classification model was fitted for the splitted data set and various accuracy_score/precision score/recall_score/f1_score was fetched.

**Since the data was highly imbalanced classification and confusion matrix was created for best performing model.**

**Name of model used are below:**
- LogisticRegression.
- SVC.
- KNeighborsClassifier.
- DecisionTreeClassifier.
- RandomForestClassifier.
- XGBClassifier.

Hyperparameter Tunning was performed through gridsearchcv of all the above mentioned model to get the best hyperparameter tunned Machine Learning program. 

**6. Conclusion:**

- **LogisticRegression algorithm** outstanded the performance of model with highest **recall score of ~71%**

**Since our ultimate motive is to correctly find the churned customer so False negative comes to utmost importance.
It should be as minimum as possible.So i selected Logistic regression model as the final model for deployment.**




