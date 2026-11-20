import numpy as np 
from sklearn.linear_model import LogisticRegression

#Number of hours studied

X=np.array([
    [1],[2],[3],[4],[5],
    [6],[7],[8],[9],[10]
])

#0=Fail,1=Pass
y=np.array([
    0,0,0,0,1,
    1,1,1,1,1
])

#Create and train the model
model=LogisticRegression()
model.fit(X,y)

#Predict a student who studied 7 hours 
student = np.array([[7]])

prediction=model.predict(student)
probability=model.predict_proba(student)

print("Prediction:",prediction[0])
print("Probability:",probability[0])
print("Intercept:",model.intercept_)
print("Coefficient:",model.coef_)