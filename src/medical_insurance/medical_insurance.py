import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


"""

The important insurance company 4Geeks Insurance S.L. wants to calculate, 
based on the physiological data of its customers what will be the premium (cost) to be borne by each of them. 
To do this, it has assembled a whole team of doctors, and based on data from other companies and a particular study, 
it has managed to gather a set of data to train a predictive model.



    age. Age of primary beneficiary (numeric)
    sex. Gender of the primary beneficiary (categorical)
    bmi. Body mass index (numeric)
    children. Number of children/dependents covered by health insurance (numeric)
    smoker. Is the person a smoker? (categorical)
    region. Beneficiary's residential area in the U.S.: northeast, southeast, southwest, northwest (categorical)
    charges. Health insurance premium (numerical)


"""


class MedicalInsurance:
    def __init__(self):
        pass

    def read_csv(self) -> pd.DataFrame:
        try:
            dataframe = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv", delimiter=",")
        except Exception as e:
            print("Error loading the dataset", e)
            return None
        return dataframe

    def process_data(self):
        df = self.read_csv()
        print(df.info())
        print(df.head())

        if df.duplicated().sum() > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            
        print(f"Num duplicated: {df.duplicated().sum()}")

        print(f"NA values:\n{df.isna().sum()}")

        df = df.drop("sex", axis=1)
        df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
        df = pd.get_dummies(df, columns=['region'], drop_first=True)
        sns.heatmap(df.corr(numeric_only=True),  annot = True, fmt = ".2f", cbar = False)
        plt.show()

        train_data, test_data = train_test_split(df, random_state=42, test_size=0.2)

        train_data.to_csv("./data/processed/medical_train_data.csv", index=False)
        test_data.to_csv("./data/processed/medical_test_data.csv", index=False)
        

        _, axis = plt.subplots(3, 1, figsize=(10, 7))
        total_data = pd.concat([train_data, test_data])

        sns.regplot(ax=axis[0], data=total_data, x="bmi", y="charges")
        sns.regplot(ax=axis[1], data=total_data, x="age", y="charges")
        sns.heatmap(ax=axis[2], data=total_data[["charges", "bmi", "age"]].corr(numeric_only=True), annot=True, fmt=".2f", cbar=False)

        plt.tight_layout()
        plt.show()

        X_train = train_data.drop("charges", axis=1)
        y_train = train_data["charges"]
        X_test = test_data.drop("charges", axis=1)
        y_test = test_data["charges"]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        model = LinearRegression()
        model.fit(X_train, y_train)

        print(f"Intercept (a): {model.intercept_}")
        print(f"Coefficients (b1, b2): {model.coef_}")

        y_pred = model.predict(X_test)
        print(y_pred)

        print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
        print(f"Coefficient of determination: {r2_score(y_test, y_pred)}")