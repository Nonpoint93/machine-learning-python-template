import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import chi2, SelectKBest
import joblib

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
        "max_iter": [200, 250, 300, 500]
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

    def plot_confusion_matrix(self, y_test, y_pred):

        """ display as plot confusion matrix """

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def process_data(self):
        dataframe = self.read_dataset()
        self.display_info_dataset(dataframe=dataframe)

        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        
        dataframe["job_n"] = pd.factorize(dataframe["job"])[0]
        dataframe["marital_n"] = pd.factorize(dataframe["marital"])[0]
        dataframe["education_n"] = pd.factorize(dataframe["education"])[0]
        dataframe["default_n"] = pd.factorize(dataframe["default"])[0]
        dataframe["housing_n"] = pd.factorize(dataframe["housing"])[0]
        dataframe["loan_n"] = pd.factorize(dataframe["loan"])[0]
        dataframe["contact_n"] = pd.factorize(dataframe["contact"])[0]
        dataframe["month_n"] = pd.factorize(dataframe["month"])[0]
        dataframe["day_of_week_n"] = pd.factorize(dataframe["day_of_week"])[0]
        dataframe["poutcome_n"] = pd.factorize(dataframe["poutcome"])[0]
        dataframe["y_n"] = pd.factorize(dataframe['y'])[0]

        numeric_features = dataframe.select_dtypes(include=["int64", "float64"]).columns
        
        scaler = MinMaxScaler()
        scal_features = scaler.fit_transform(dataframe[numeric_features])
        dataframe = pd.DataFrame(scal_features, index = dataframe.index, columns = numeric_features)
        
        self.display_info_dataset(dataframe=dataframe)

        X = dataframe.drop("y_n", axis=1)
        y = dataframe["y_n"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

        model = SelectKBest(chi2, k=7)
        model.fit(X_train, y_train)
        features_selected = model.get_support()
        X_train = pd.DataFrame(model.transform(X_train), columns = X_train.columns.values[features_selected])
        X_test = pd.DataFrame(model.transform(X_test), columns = X_test.columns.values[features_selected])

        self.display_info_dataset(X_train)
        self.display_info_dataset(X_test)

        X_train["y_n"] = list(y_train)
        X_test["y_n"] = list(y_test)
        X_train.to_csv("./data/processed/bank_data_train.csv", index = False)
        X_test.to_csv("./data/processed/bank_data_test.csv", index = False)

        train_data = pd.read_csv("./data/processed/bank_data_train.csv")
        test_data = pd.read_csv("./data/processed/bank_data_test.csv")

        X_train = train_data.drop("y_n", axis=1)
        y_train = train_data["y_n"]
        X_test = test_data.drop("y_n", axis=1)
        y_test = test_data["y_n"]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

        grid = GridSearchCV(model, self.hyperparams, scoring = "recall", cv = 5)
        grid.fit(X_train, y_train)
        print(f"Best hyperparamters: {grid.best_params_}")

        model = LogisticRegression(C = 1000, max_iter = 200, penalty = 'l2', solver = 'lbfgs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

        self.plot_confusion_matrix(y_test, y_pred)

        joblib.dump(model, "./data/processed/bank_model.pkl")