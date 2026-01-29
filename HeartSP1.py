import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv(r'\Users\User\Downloads\sp1\heart.csv')


df=data
print(df.columns)

"""Variable	Description

age	Age of the patient in years
sex	Gender of the patient (0 = male, 1 = female)
cp	Chest pain type:
0: Typical angina
1: Atypical angina
2: Non-anginal pain
3: Asymptomatic

trestbps	Resting blood pressure in mm Hg

chol	Serum cholesterol in mg/dl

fbs	Fasting blood sugar level, categorized as above 120 mg/dl (1 = true, 0 = false)

restecg	Resting electrocardiographic results:
0: Normal
1: Having ST-T wave abnormality
2: Showing probable or definite left ventricular hypertrophy

thalach	Maximum heart rate achieved during a stress test

exang	Exercise-induced angina (1 = yes, 0 = no)

oldpeak	ST depression induced by exercise relative to rest

slope	Slope of the peak exercise ST segment:
0: Upsloping
1: Flat
2: Downsloping

ca	Number of major vessels (0-4) colored by fluoroscopy

thal	Thalium stress test result:
0: Normal
1: Fixed defect
2: Reversible defect
3: Not described

target	Heart disease status (0 = no disease, 1 = presence of disease)

"""

# Split the data into features and target variable
'''df_X = data.drop('Diabetes_binary', axis=1)
df_y = data['Diabetes_binary']'''
"""
import seaborn as sns

df_continuous = df[['age', 'trestbps', 'chol','thalach','oldpeak']]

# Set up the subplot
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Loop to plot histograms for each continuous feature
for i, col in enumerate(df_continuous.columns):
    x = i // 3
    y = i % 3
    values, bin_edges = np.histogram(df_continuous[col], 
                                     range=(np.floor(df_continuous[col].min()), np.ceil(df_continuous[col].max())))
    
    graph = sns.histplot(data=df_continuous, x=col, bins=bin_edges, kde=True, ax=ax[x, y],
                         edgecolor='none', color='red', alpha=0.6, line_kws={'lw': 3})
    ax[x, y].set_xlabel(col, fontsize=15)
    ax[x, y].set_ylabel('Count', fontsize=12)
    ax[x, y].set_xticks(np.round(bin_edges, 1))
    ax[x, y].set_xticklabels(ax[x, y].get_xticks(), rotation=45)
    ax[x, y].grid(color='lightgrey')
    
    for j, p in enumerate(graph.patches):
        ax[x, y].annotate('{}'.format(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height() + 1),
                          ha='center', fontsize=10, fontweight="bold")
    
    textstr = '\n'.join((
        r'$\mu=%.2f$' % df_continuous[col].mean(),
        r'$\sigma=%.2f$' % df_continuous[col].std()
    ))
    ax[x, y].text(0.75, 0.9, textstr, transform=ax[x, y].transAxes, fontsize=12, verticalalignment='top',
                  color='white', bbox=dict(boxstyle='round', facecolor='#ff826e', edgecolor='white', pad=0.5))

ax[1,2].axis('off')
plt.suptitle('Distribution of Continuous Variables', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()"""

"""print(df.head(5))"""

df_encoded = pd.get_dummies(df, columns=['cp', 'restecg', 'thal'], prefix=['cp', 'restecg','thal'])

# Convert the rest of the categorical variables that don't need one-hot encoding to integer data type
features_to_convert = ['sex', 'fbs', 'exang', 'slope', 'ca', 'target']
for feature in features_to_convert:
    df_encoded[feature] = df_encoded[feature].astype(int)

print(df_encoded.dtypes)

df_final = df_encoded
print(df_final.head(5))

X = df_final.drop('target', axis = 1)
y = df_final['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print(X_train.columns)
DT  = DecisionTreeClassifier (random_state= 0)

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def evaluate(model, X_test, y_test, modelName):
    yPred = model.predict(X_test)
    report  = classification_report(y_test, yPred, output_dict = True, zero_division=1)
    
    metrics = {
        "precision_0": report["0"]["precision"],
        "precision_1": report["1"]["precision"],
        "recall_0": report["0"]["recall"],
        "recall_1": report["1"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "f1_1": report["1"]["f1-score"],
        "macro_avg_precision": report["macro avg"]["precision"],
        "macro_avg_recall": report["macro avg"]["recall"],
        "macro_avg_f1": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(y_test, yPred)
    }
    
    df = pd.DataFrame(metrics, index = [modelName]).round(2)
    
    return df    

def hyperParameter(clf, paramGrid, X_train, y_train, scoring = 'recall', n_splits = 3):
    
    crossVal= StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf_grid = GridSearchCV(clf, paramGrid, cv= crossVal,scoring = scoring, n_jobs = -1 )
    clf_grid.fit(X_train, y_train)
    hyperparameter = clf_grid.best_params_
    return clf_grid.best_estimator_, hyperparameter

paramGrid_DT = { 'criterion' : ['entropy'],
                'max_depth' : [2,3],
                'min_samples_split': [2,3,4],
                'min_samples_leaf': [1,2]}

dt, dt_hyperparams = hyperParameter(DT, paramGrid_DT, X_train, y_train)
print('DT Optimal Hyperparameters: \n', dt_hyperparams)
print()

print(classification_report(y_train, dt.predict(X_train)))

RandomForest = RandomForestClassifier(random_state=0)

paramGrid_RF = {
    'n_estimators':[10,30,50,70,90,110],
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,3,4],
    'min_samples_split': [2,3,4,5],
    'min_samples_leaf':[1,2,3],
}

accur_array=[]

rf,rf_hyperparameters = hyperParameter(RandomForest, paramGrid_RF, X_train, y_train)
print()
print("Optimal hyerparameters are for Random Forest: ", rf_hyperparameters)
print(classification_report(y_train,rf.predict(X_train)))
print(classification_report(y_test,rf.predict(X_test)))
rf_acc = accuracy_score(y_test,rf.predict(X_test))
accur_array.append(rf_acc)

lr = LogisticRegression(solver= 'saga', max_iter=1000)
lr.fit(X_train, y_train)
print()
print("Optimal hyerparameters are for Logistic Regression: solver= saga, max_iter =1000")
print(classification_report(y_train,lr.predict(X_train)))
print(classification_report(y_test,lr.predict(X_test)))
lr_acc = accuracy_score(y_test,lr.predict(X_test))
accur_array.append(lr_acc)


gbModel = GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=2, random_state=500)
gbModel.fit(X_train,y_train)
print("Gradient Boosting: ")
print(classification_report(y_train, gbModel.predict(X_train)))
print(classification_report(y_test, gbModel.predict(X_test)))
GB_acc = accuracy_score(y_test,gbModel.predict(X_test))

print(GB_acc)
accur_array.append(np.round(0.819672131147541,5))



from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

svmPipe = Pipeline([
    ('scalar', StandardScaler()), ('svm', SVC(probability=True))])
paramGrid_SVM = {
    'svm__C':[0.0011, 0.005,0.05,0.1,1,10,20],
    'svm__kernel':['linear'],
    'svm__gamma': ['scale', 'auto'],
    'svm__degree':[2,3,4]
}

svm, svm_hyperparameters = hyperParameter(svmPipe, paramGrid_SVM, X_train, y_train)
print()
print("Optimal SVM parameters: ", svm_hyperparameters)

print()
print(classification_report(y_train, svm.predict(X_train)))
print(classification_report(y_test, svm.predict(X_test)))
SVM_acc = accuracy_score(y_test,svm.predict(X_test))


acc_array=[]
models = [dt,rf,lr,svm,gbModel]
modelName = ['DecisionTree', 'RandomForest', 'LogisticRegrssion', 'SupportVectorMachine', 'GradientBoosting']

modelList = [DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=1, min_samples_split=2), 
             RandomForestClassifier(criterion='gini', max_depth=2, min_samples_leaf=3, min_samples_split=2, n_estimators=30),
             LogisticRegression(solver='liblinear', max_iter=100), 
             SVC(C=0.0011, degree=2, gamma= 'scale', kernel='linear'), 
             GradientBoostingClassifier(learning_rate=0.05, n_estimators=500, max_depth=2, random_state=500)]
train_acc=[]
for i in range (len(models)):
    trainingacc = accuracy_score(y_train,models[i].predict(X_train))
    acc= accuracy_score(y_test,models[i].predict(X_test))
    
    train_acc.append(np.round(trainingacc,5))
    
    acc_array.append(np.round(acc, 5))

z=0
t=0
for i in range(len(acc_array)):
    print("Model: ", modelName[i], " Training Accuracy: " , train_acc[i])
    print("Model: ", modelName[i], " Testing Accuracy:  ", acc_array[i])
    print()
    if acc_array[i]>z:
        z=acc_array[i]
        highestTest= modelList[i]
    if train_acc[i]>t:
        t=train_acc[i]
        highestTrain = modelList[i]    
print("Highest training accuracy Model in testing is: ")
print(highestTrain, " with Accuracy of ", t)
print()
print("Highest testing accuracy Model in testing is: ")
print(highestTest, " with Accuracy of ", z)

#highestTest = RandomForestClassifier(criterion='gini', max_depth=2, min_samples_leaf=3, min_samples_split=2, n_estimators=30)
highestTest.fit(X_train,y_train)
    
from joblib import dump, load
model_filename = "highestTestModel.joblib"
dump(highestTest, model_filename)

def get_user_input():
    input_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {}

    for column in input_columns:
        print()
        if column == 'sex':
            value = int(input(f"Enter value for {column}: 0 for Male | 1 for Female "))
        elif column == 'cp':
            print(f"Enter value for {column}:")
            print("0: Typical angina\n1: Atypical angina\n2: Non-anginal pain\n3: Asymptomatic")
            value = int(input("Your choice (0-3): "))
            while value not in range(4):
                value = int(input("Please enter a value from 0 to 3: "))
            # Map 'cp' value to boolean columns
            user_input['cp_0'] = (value == 0)
            user_input['cp_1'] = (value == 1)
            user_input['cp_2'] = (value == 2)
            user_input['cp_3'] = (value == 3)
        elif column == 'trestbps':
            value = int(input("Enter value for Resting blood pressure in mm Hg: "))
            while not isinstance(value, int):
                value = int(input("Please input a valid Resting Blood Pressure Number: "))
        elif column == 'chol':
            value = int(input("Enter value for Serum Cholesterol in mg/dl: "))
            while not isinstance(value, int):
                value = int(input("Please input a valid Cholesterol number: "))
        elif column == 'fbs':
            value = int(input("Enter 1 if your fasting Blood Sugar level is more than 120, Enter 0 if it is less than 120 mg/dl: "))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'restecg':
            print(f"Enter value for {column}:")
            print("0: Normal\n1: Having ST-T wave abnormality\n2: Showing probable or definite left ventricular hypertrophy")
            value = int(input("Your choice (0-2): "))
            while value not in range(3):
                value = int(input("Please enter a value from 0 to 2: "))
            # Map 'restecg' value to boolean columns
            user_input['restecg_0'] = (value == 0)
            user_input['restecg_1'] = (value == 1)
            user_input['restecg_2'] = (value == 2)
        elif column == 'thalach':
            value = int(input("Enter your Maximum Heart Rate achieved during stress test: "))
            while not isinstance(value, int):
                value = int(input("Please input a valid number: "))
        elif column == 'exang':
            value = int(input("Do you have Exercise-induced Angina? (Heart Pain): 1 = yes, 0 = no: "))
            while value not in [0, 1]:
                value = int(input("Please enter 0 or 1: "))
        elif column == 'oldpeak':
            value = float(input("Enter the ST depression induced by exercise relative to rest: "))
            while not isinstance(value, (int, float)):
                value = float(input("Please input a valid number: "))
        elif column == 'slope':
            print(f"Enter Slope of the peak exercise ST segment:")
            print("0: Upsloping\n1: Flat\n2: Downsloping")
            value = int(input("Your choice (0-2): "))
            while value not in range(3):
                value = int(input("Please enter a value from 0 to 2: "))
        elif column == 'ca':
            value = int(input("Enter the number of major vessels colored by fluoroscopy (0 - 4): "))
            while value not in range(5):
                value = int(input("Please enter a value from 0 to 4: "))
        elif column == 'thal':
            print(f"Enter your {column} Stress Test Result:")
            print("0: Normal\n1: Fixed defect\n2: Reversible defect\n3: Not described")
            value = int(input("Your choice (0-3): "))
            while value not in range(4):
                value = int(input("Please enter a value from 0 to 3: "))
            # Map 'thal' value to boolean columns
            user_input['thal_0'] = (value == 0)
            user_input['thal_1'] = (value == 1)
            user_input['thal_2'] = (value == 2)
            user_input['thal_3'] = (value == 3)
        elif column == 'age':
            value = int(input("Enter your age: "))
            while not isinstance(value, int):
                value = int(input("Please input a valid number: "))
            
        # You might want to add further validation for input values here
        user_input[column] = value

    return user_input

user_Preinput=get_user_input()
selected_columns = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
       'slope', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0', 'restecg_1',
       'restecg_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3']

user_input = {key: value for key, value in user_Preinput.items() if key in selected_columns}
print(user_input)


user_input = {'age': 70, 'sex': 0, 'cp_0': True, 'cp_1': False, 'cp_2': False, 'cp_3': False, 'trestbps': 400, 'chol': 340, 'fbs': 1, 'restecg_0': False, 'restecg_1': False, 'restecg_2': True, 'thalach': 700, 'exang': 1, 'oldpeak': 3.5, 'slope': 1, 'ca': 1, 'thal_0': False, 'thal_1': True, 'thal_2': False, 'thal_3': False}

def make_prediction(user_input,model):
    # Convert user input to a NumPy array for prediction
    input_array = np.array(list(user_input.values())).reshape(1, -1)
    feature_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
       'slope', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0', 'restecg_1',
       'restecg_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3']
    input_df = pd.DataFrame(input_array, columns=feature_names)
    # Make prediction using the trained model
    prediction = model.predict(input_df)
    
    print(prediction)

    return prediction

loadedModel = load(model_filename)

if isinstance(loadedModel, SVC):
    feature_names = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak',
       'slope', 'ca', 'cp_0', 'cp_1', 'cp_2', 'cp_3', 'restecg_0', 'restecg_1',
       'restecg_2', 'thal_0', 'thal_1', 'thal_2', 'thal_3']
    svm_params = {
        'C': 0.0011,
        'degree': 2,
        'gamma': 'scale',
        'kernel': 'linear'
    }

    # Create the pipeline
    svmPipe = Pipeline([
        ('scalar', StandardScaler()), 
        ('svm', SVC(probability=True, **svm_params))
    ])

    svmPipe.fit(X_train[feature_names], y_train)

    # Scale the user input
    user_input_array = np.array([user_input[feature] for feature in feature_names]).reshape(1, -1)
    
    user_input_scaled = svmPipe.named_steps['scalar'].transform(user_input_array)
     
    # Make prediction
    prediction = svmPipe.predict(user_input_array)
    print("Prediction:", prediction)
else:
    print("The loaded model is not an SVC model.")
    prediction = make_prediction(user_input, loadedModel)

print("\nPrediction of Heart Disease existing is: ", prediction)
if prediction == 1: 
    

  print("""Positive. The Model Predicts there might be a heart disease.
         This IS NOT DIRECT MEDICAL ADVICE. PLEASE ADVISE WITH YOUR DOCTOR""")
else:
    print("""Negative. The Model Predicts there might NOT be a heart disease.
          This IS NOT DIRECT MEDICAL ADVICE. PLEASE ADVISE WITH YOUR DOCTOR""")

