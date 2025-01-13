import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings

"""

Description of the problem

The Portuguese bank is experiencing a decline in revenue, so they want to be able to identify existing customers who are more likely to take out a long-term deposit. This will allow the bank to focus their marketing efforts on those customers and avoid wasting money and time on customers who are unlikely to sign up.

To address this problem we will create a ranking algorithm to help predict whether or not a customer will sign up for a long-term deposit.


    age. Age of customer (numeric)
    job. Type of job (categorical)
    marital. Marital status (categorical)
    education. Level of education (categorical)
    default. Do you currently have credit (categorical)
    housing. Do you have a housing loan (categorical)
    loan. Do you have a personal loan? (categorical)
    contact. Type of contact communication (categorical)
    month. Last month in which you have been contacted (categorical)
    day_of_week. Last day on which you have been contacted (categorical)
    duration. Duration of previous contact in seconds (numeric)
    campaign. Number of contacts made during this campaign to the customer (numeric)
    pdays. Number of days that elapsed since the last campaign until the customer was contacted (numeric)
    previous. Number of contacts made during the previous campaign to the customer (numeric)
    poutcome. Result of the previous marketing campaign (categorical)
    emp.var.rate. Employment variation rate. Quarterly indicator (numeric)
    cons.price.idx. Consumer price index. Monthly indicator (numeric)
    cons.conf.idx. Consumer confidence index. Monthly indicator (numeric)
    euribor3m. EURIBOR 3-month rate. Daily indicator (numeric)
    nr.employed. Number of employees. Quarterly indicator (numeric)
    y. TARGET. Whether the customer takes out a long-term deposit or not (categorical)

"""

class BankService:

    hyperparams = {
	    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
	    "penalty": ["l1", "l2", "elasticnet", "none"],
	    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [200, 250, 300]
	}


    def __init__(self):
        pass


    def read_dataset(self) -> pd.DataFrame:

        """ Read a dataset """
        dataframe = None
        try:
            dataframe = pd.read_csv("data/raw/bank-marketing-campaign-data.csv", delimiter=";")
        except Exception as e:
            print(e)
            sys.exit(1)
        return dataframe
    
    def display_info_dataset(self, dataframe: pd.DataFrame):
        
        """ Print info from dataset """
        
        print(dataframe.info())
        print(dataframe.head())
        print(dataframe.shape)
        print(dataframe.describe())

    def display_correlation_dataset(self, dataframe: pd.DataFrame):
        
        """ Displaying correlation dataset """

        sns.heatmap(dataframe.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")

        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    def remove_outliers(self, df, column) -> pd.DataFrame:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
        return df
    

    def warn(self, *args, **kwargs):
        pass

    def process_data(self):
        dataframe = self.read_dataset()
        self.display_info_dataset(dataframe=dataframe)

        dataframe.loc[dataframe['default'] == 'unknown', 'default'] = 'no'
        
        dataframe["default"] = dataframe["default"].apply(lambda x: 1 if x == 'yes' else 0)
        dataframe["loan"] = dataframe["loan"].apply(lambda x: 1 if x == 'yes' else 0)
        dataframe["housing"] = dataframe["housing"].apply(lambda x: 1 if x == 'yes' else 0)
        dataframe = dataframe.drop(columns=['job', 'marital', 'education', 'contact'], axis=1)
        dataframe = pd.get_dummies(dataframe, columns=['month', 'day_of_week', 'poutcome'])
        numeric_features = dataframe.select_dtypes(include=["int64", "float64"]).columns
        dataframe["y"] = dataframe["y"].apply(lambda x: 1 if x == 'yes' else 0)
        scaler = StandardScaler()
        dataframe[numeric_features] = scaler.fit_transform(dataframe[numeric_features])

        X = dataframe.drop("y", axis=1)
        y = dataframe["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.80)

        model = LogisticRegression(class_weight='balanced')
        print(y_train.unique())

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

        grid = GridSearchCV(model, self.hyperparams, scoring = "accuracy", cv = 5)
        grid.fit(X_train, y_train)
        print(f"Best hyperparamters: {grid.best_params_}")
