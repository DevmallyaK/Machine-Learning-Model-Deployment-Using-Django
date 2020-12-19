import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('C:/Users/Devmallya Karar/Desktop/Model/glass.csv')

X = df.iloc[:, :-1]
y = df.iloc[:,9]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(criterion='entropy', n_estimators=300, random_state=42)
cls.fit(X_train, y_train)

#y_pred = cls.predict(X_test)

print('Accuracy is', cls.score(X_test, y_test)*100,'%')

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(cls, filename)