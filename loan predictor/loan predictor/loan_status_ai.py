# # import mysql.connector
# # import pandas as pd
# # import numpy as np
# # from sqlalchemy import create_engine
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, classification_report
# # from flask import Flask, request, jsonify

# # # Debugging Statements
# # print("✅ Connecting to MySQL...")

# # # Database Connection using SQLAlchemy
# # engine = create_engine("mysql+pymysql://root:Sunil10%40goyal@localhost/LoanDB")
# # conn = mysql.connector.connect(
# #     host="localhost",
# #     user="root",
# #     password="Sunil10@goyal",
# #     database="LoanDB"
# # )
# # cursor = conn.cursor()

# # print("✅ Connected to MySQL!")

# # # Create Database if it doesn't exist
# # cursor.execute("CREATE DATABASE IF NOT EXISTS LoanDB")
# # cursor.execute("USE LoanDB")

# # # Drop and recreate the table to ensure schema matches CSV
# # print("✅ Creating Table if not exists...")
# # cursor.execute('''
# # DROP TABLE IF EXISTS loan_applications;
# # ''')
# # cursor.execute('''
# # CREATE TABLE loan_applications (
# #     id INT AUTO_INCREMENT PRIMARY KEY,
# #     person_age INT,
# #     person_income FLOAT,
# #     person_home_ownership VARCHAR(50),
# #     person_emp_length FLOAT,
# #     loan_intent VARCHAR(50),
# #     loan_grade VARCHAR(5),
# #     loan_amnt FLOAT,
# #     loan_int_rate FLOAT,
# #     loan_status INT,
# #     loan_percent_income FLOAT,
# #     cb_person_default_on_file VARCHAR(5),
# #     cb_person_cred_hist_length INT
# # );
# # ''')

# # # Close and reopen cursor to sync with MySQL
# # cursor.close()
# # conn.commit()
# # cursor = conn.cursor()  
# # print("✅ Table Created Successfully!")

# # # Load CSV data into MySQL if table is empty
# # csv_file_path = "C:/loan_data.csv"
# # def load_data():
# #     print("✅ Loading Data from MySQL...")
# #     cursor = conn.cursor()
# #     df = pd.read_sql("SELECT * FROM loan_applications", engine)
    
# #     # If not enough data in MySQL, load from CSV
# #     if df.shape[0] < 50:
# #         print(f"⚠️ Insufficient data ({df.shape[0]} rows). Importing from CSV...")
# #         df_csv = pd.read_csv(csv_file_path)
        
# #         # Ensure column names match the MySQL table
# #         df_csv.columns = [
# #             "person_age", "person_income", "person_home_ownership", "person_emp_length", 
# #             "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status", 
# #             "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
# #         ]
        
# #         df_csv.to_sql(name="loan_applications", con=engine, if_exists="append", index=False)
# #         print("✅ CSV Data Imported Successfully!")
        
# #         # Reload from MySQL after insertion
# #         df = pd.read_sql("SELECT * FROM loan_applications", engine)
    
# #     print(f"✅ Final Dataset Shape: {df.shape}")
# #     return df

# # data = load_data()

# # # Data Preprocessing
# # categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

# # # One-Hot Encoding for categorical variables
# # data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# # # Splitting features (X) and target variable (y)
# # X = data.drop(columns=['id', 'loan_status'])
# # y = data['loan_status']

# # # Ensure enough data for test split
# # test_size = 0.2 if data.shape[0] > 10 else 0.3  # Adjust test size dynamically
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if data.shape[0] > 10 else None)

# # # Standardizing numerical features
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)

# # print("✅ Training Model...")
# # # Train Model
# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # model.fit(X_train, y_train)
# # print("✅ Model Training Completed!")

# # # Evaluate Model
# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print(classification_report(y_test, y_pred, zero_division=1))

# # # Flask API for Predictions
# # app = Flask(__name__)

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     data = request.json
# #     features = np.array([[
# #         data['person_age'], data['person_income'], data['person_home_ownership'],
# #         data['person_emp_length'], data['loan_intent'], data['loan_grade'],
# #         data['loan_amnt'], data['loan_int_rate'], data['loan_percent_income'],
# #         data['cb_person_default_on_file'], data['cb_person_cred_hist_length']
# #     ]])
# #     features = scaler.transform(features)
# #     prediction = model.predict(features)[0]
# #     return jsonify({'loan_status': int(prediction)})

# # print("✅ Starting Flask API...")
# # if __name__ == '__main__':
# #     app.run(debug=True)





















# import mysql.connector
# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Database Connection
# engine = create_engine("mysql+pymysql://root:Sunil10%40goyal@localhost/LoanDB")
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="Sunil10@goyal",
#     database="LoanDB"
# )
# cursor = conn.cursor()

# # Load CSV data into MySQL if table is empty
# csv_file_path = "C:/loan_data.csv"
# def load_data():
#     cursor = conn.cursor()
#     df = pd.read_sql("SELECT * FROM loan_applications", engine)
#     if df.shape[0] < 50:
#         df_csv = pd.read_csv(csv_file_path)
#         df_csv.columns = [
#             "person_age", "person_income", "person_home_ownership", "person_emp_length", 
#             "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status", 
#             "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
#         ]
#         df_csv.to_sql(name="loan_applications", con=engine, if_exists="append", index=False)
#         df = pd.read_sql("SELECT * FROM loan_applications", engine)
#     return df

# data = load_data()

# # Data Preprocessing
# categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# X = data.drop(columns=['id', 'loan_status'])
# y = data['loan_status']

# test_size = 0.2 if data.shape[0] > 10 else 0.3
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if data.shape[0] > 10 else None)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train Model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate Model
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, zero_division=1))

# # Function to Predict Loan Status
# def predict_loan_status(person_age, person_income, person_home_ownership, 
#                         person_emp_length, loan_intent, loan_grade, loan_amnt, 
#                         loan_int_rate, loan_percent_income, cb_person_default_on_file, 
#                         cb_person_cred_hist_length):
#     input_data = {
#         "person_age": person_age,
#         "person_income": person_income,
#         "person_home_ownership": person_home_ownership,
#         "person_emp_length": person_emp_length,
#         "loan_intent": loan_intent,
#         "loan_grade": loan_grade,
#         "loan_amnt": loan_amnt,
#         "loan_int_rate": loan_int_rate,
#         "loan_percent_income": loan_percent_income,
#         "cb_person_default_on_file": cb_person_default_on_file,
#         "cb_person_cred_hist_length": cb_person_cred_hist_length
#     }
#     input_df = pd.DataFrame([input_data])
#     input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
#     for col in X.columns:
#         if col not in input_df.columns:
#             input_df[col] = 0
#     input_features = scaler.transform(input_df)
#     prediction = model.predict(input_features)[0]
#     return "Approved" if prediction == 1 else "Denied"

# # Example Prediction
# result = predict_loan_status(
#     person_age=30, 
#     person_income=60000, 
#     person_home_ownership="RENT", 
#     person_emp_length=6, 
#     loan_intent="PERSONAL", 
#     loan_grade="B", 
#     loan_amnt=20000, 
#     loan_int_rate=12.5, 
#     loan_percent_income=0.3, 
#     cb_person_default_on_file="N", 
#     cb_person_cred_hist_length=7
# )
# print("Loan Status Prediction:", result)














# import mysql.connector
# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import accuracy_score, classification_report

# # ------------------- Database Connection -------------------
# engine = create_engine("mysql+pymysql://root:Tabletennis1234@localhost/LoanDB")
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="Tabletennis1234",
#     database="LoanDB"
# )
# cursor = conn.cursor()

# # ------------------- Load CSV Data into MySQL -------------------
# csv_file_path = "C:/loan_data.csv"

# def load_data():
#     df = pd.read_sql("SELECT * FROM loan_applications", engine)
#     if df.shape[0] < 50:
#         df_csv = pd.read_csv(csv_file_path)
#         df_csv.columns = [
#             "person_age", "person_income", "person_home_ownership", "person_emp_length", 
#             "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status", 
#             "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
#         ]
#         df_csv.to_sql(name="loan_applications", con=engine, if_exists="append", index=False)
#         df = pd.read_sql("SELECT * FROM loan_applications", engine)
#     return df

# data = load_data()

# # ------------------- Data Preprocessing -------------------
# categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# # ------------------- Prepare Data for Different Models -------------------
# X = data.drop(columns=['id', 'loan_status', 'loan_int_rate'])
# y_status = data['loan_status']
# y_amount = data['loan_amnt']
# y_interest = data['loan_int_rate'].fillna(data['loan_int_rate'].median())

# # ------------------- Train-Test Split -------------------
# test_size = 0.2 if data.shape[0] > 10 else 0.3

# X_train, X_test, y_train_status, y_test_status = train_test_split(
#     X, y_status, test_size=test_size, random_state=42, stratify=y_status if data.shape[0] > 10 else None
# )
# X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(
#     X, y_amount, test_size=test_size, random_state=42
# )
# X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
#     X, y_interest, test_size=test_size, random_state=42
# )

# # ------------------- Feature Scaling -------------------
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_train_amount = scaler.transform(X_train_amount)
# X_test_amount = scaler.transform(X_test_amount)
# X_train_int = scaler.transform(X_train_int)
# X_test_int = scaler.transform(X_test_int)

# # ------------------- Train Models -------------------
# status_model = RandomForestClassifier(n_estimators=100, random_state=42)
# status_model.fit(X_train, y_train_status)

# amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
# amount_model.fit(X_train_amount, y_train_amount)

# interest_model = RandomForestRegressor(n_estimators=100, random_state=42)
# interest_model.fit(X_train_int, y_train_int)

# # ------------------- Evaluate Loan Approval Model -------------------
# y_pred_status = status_model.predict(X_test)
# print("Loan Approval Model Accuracy:", accuracy_score(y_test_status, y_pred_status))
# print(classification_report(y_test_status, y_pred_status, zero_division=1))

# # ------------------- Prediction Function -------------------
# def predict_loan():
#     person_age = int(input("Enter age: "))
#     person_income = float(input("Enter income: "))
#     person_home_ownership = input("Enter home ownership (RENT/OWN/MORTGAGE): ")
#     person_emp_length = float(input("Enter employment length (in years): "))
#     loan_intent = input("Enter loan intent (PERSONAL, EDUCATION, MEDICAL, etc.): ")
#     loan_grade = input("Enter loan grade (A, B, C, etc.): ")
#     loan_amnt = float(input("Enter requested loan amount: "))
#     loan_percent_income = float(input("Enter loan percent income: "))
#     cb_person_default_on_file = input("Default on file (Y/N): ")
#     cb_person_cred_hist_length = int(input("Enter credit history length: "))
    
#     input_data = {
#         "person_age": person_age,
#         "person_income": person_income,
#         "person_home_ownership": person_home_ownership,
#         "person_emp_length": person_emp_length,
#         "loan_intent": loan_intent,
#         "loan_grade": loan_grade,
#         "loan_amnt": loan_amnt,
#         "loan_percent_income": loan_percent_income,
#         "cb_person_default_on_file": cb_person_default_on_file,
#         "cb_person_cred_hist_length": cb_person_cred_hist_length
#     }
    
#     input_df = pd.DataFrame([input_data])
#     input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    
#     # Ensure all training columns exist
#     for col in X.columns:
#         if col not in input_df.columns:
#             input_df[col] = 0
#     input_df = input_df[X.columns]  # Reorder to match training data
    
#     input_features = scaler.transform(input_df)
    
#     # Predictions
#     status_prediction = status_model.predict(input_features)[0]
#     print("Loan Status Prediction:", "Approved" if status_prediction == 1 else "Denied")
    
#     estimated_amount = amount_model.predict(input_features)[0]
#     print("Estimated Loan Amount Approval:", round(estimated_amount, 2))
    
#     estimated_interest = interest_model.predict(input_features)[0]
#     print("Estimated Interest Rate:", round(estimated_interest, 2), "%")

# # ------------------- Run Prediction -------------------
# predict_loan()









import mysql.connector
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report

# ------------------- Database Connection -------------------
engine = create_engine("mysql+pymysql://root:Tabletennis1234@localhost/LoanDB")
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Tabletennis1234",
    database="LoanDB"
)
cursor = conn.cursor()

# ------------------- Load CSV Data into MySQL -------------------
csv_file_path = "C:/loan_data.csv"

def load_data():
    inspector = inspect(engine)
    
    if "loan_applications" not in inspector.get_table_names():
        print("Table 'loan_applications' not found. Creating it...")

        create_table_sql = """
        CREATE TABLE loan_applications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            person_age INT,
            person_income FLOAT,
            person_home_ownership VARCHAR(50),
            person_emp_length FLOAT,
            loan_intent VARCHAR(50),
            loan_grade VARCHAR(10),
            loan_amnt FLOAT,
            loan_int_rate FLOAT,
            loan_status INT,
            loan_percent_income FLOAT,
            cb_person_default_on_file VARCHAR(10),
            cb_person_cred_hist_length INT
        );
        """
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()

    df = pd.read_sql("SELECT * FROM loan_applications", engine)
    
    if df.shape[0] < 50:
        df_csv = pd.read_csv(csv_file_path)
        df_csv.columns = [
            "person_age", "person_income", "person_home_ownership", "person_emp_length", 
            "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status", 
            "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
        ]
        df_csv.to_sql(name="loan_applications", con=engine, if_exists="append", index=False)
        df = pd.read_sql("SELECT * FROM loan_applications", engine)
    
    return df

data = load_data()

# ------------------- Data Preprocessing -------------------
categorical_columns = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# ------------------- Prepare Data for Different Models -------------------
X = data.drop(columns=['id', 'loan_status', 'loan_int_rate'])
y_status = data['loan_status']
y_amount = data['loan_amnt']
y_interest = data['loan_int_rate'].fillna(data['loan_int_rate'].median())

# ------------------- Train-Test Split -------------------
test_size = 0.2 if data.shape[0] > 10 else 0.3

X_train, X_test, y_train_status, y_test_status = train_test_split(
    X, y_status, test_size=test_size, random_state=42, stratify=y_status if data.shape[0] > 10 else None
)
X_train_amount, X_test_amount, y_train_amount, y_test_amount = train_test_split(
    X, y_amount, test_size=test_size, random_state=42
)
X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
    X, y_interest, test_size=test_size, random_state=42
)

# ------------------- Feature Scaling -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_amount = scaler.transform(X_train_amount)
X_test_amount = scaler.transform(X_test_amount)
X_train_int = scaler.transform(X_train_int)
X_test_int = scaler.transform(X_test_int)

# ------------------- Train Models -------------------
status_model = RandomForestClassifier(n_estimators=100, random_state=42)
status_model.fit(X_train, y_train_status)

amount_model = RandomForestRegressor(n_estimators=100, random_state=42)
amount_model.fit(X_train_amount, y_train_amount)

interest_model = RandomForestRegressor(n_estimators=100, random_state=42)
interest_model.fit(X_train_int, y_train_int)

# ------------------- Evaluate Loan Approval Model -------------------
y_pred_status = status_model.predict(X_test)
print("Loan Approval Model Accuracy:", accuracy_score(y_test_status, y_pred_status))
print(classification_report(y_test_status, y_pred_status, zero_division=1))

# ------------------- Prediction Function -------------------
def predict_loan():
    person_age = int(input("Enter age: "))
    person_income = float(input("Enter income: "))
    person_home_ownership = input("Enter home ownership (RENT/OWN/MORTGAGE): ")
    person_emp_length = float(input("Enter employment length (in years): "))
    loan_intent = input("Enter loan intent (PERSONAL, EDUCATION, MEDICAL, etc.): ")
    loan_grade = input("Enter loan grade (A, B, C, etc.): ")
    loan_amnt = float(input("Enter requested loan amount: "))
    loan_percent_income = float(input("Enter loan percent income: "))
    cb_person_default_on_file = input("Default on file (Y/N): ")
    cb_person_cred_hist_length = int(input("Enter credit history length: "))
    
    input_data = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }
    
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]
    
    input_features = scaler.transform(input_df)
    
    status_prediction = status_model.predict(input_features)[0]
    print("Loan Status Prediction:", "Approved" if status_prediction == 1 else "Denied")
    
    estimated_amount = amount_model.predict(input_features)[0]
    print("Estimated Loan Amount Approval:", round(estimated_amount, 2))
    
    estimated_interest = interest_model.predict(input_features)[0]
    print("Estimated Interest Rate:", round(estimated_interest, 2), "%")

# ------------------- Run Prediction -------------------
predict_loan()

