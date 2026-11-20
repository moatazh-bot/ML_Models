 # th principe of the isolation_forest models , is that we randomly measure the loength of the path to the first anomaly , 
#shorter the path to a such point is , more possible to be an anomaly .
from sklearn.ensemble import IsolationForest
import numpy as np 

#sickit_learn is the llibrary of machine learning where there is a lot of methods and algorithms of ml.
#numpy is the mathematical library , it's important for calculation and creating data structures
#we gonna gnerate data: 20 normal points ~ N(0,1) , 2 outliers =unormals  ~ U(5,6)
np.random.seed(42)
data=np.vstack([np.random.normal(0,1,(20,2)),np.random.uniform(5,6,(2,2))])


#Fit IsolaForest (expect ~10% anomalies , matching 2/22 points) we make the model training 
model = IsolationForest(contamination=0.1,random_state=42, n_estimators=100)
#contamination=0.1 means that may be the 10% of the data is anomaly ( this percentage depends on our fields of searching )
model.fit(data) #we make the model train on our data
predictions=model.predict(data)#after the model is trained on the data we will see the results , what did he predict form this data 

#Print anomalies (-1 indicates anomaly ,1 indicates normal)
anomalies_found = False #by default we begin that we don't have any anomaly
for i, (point,pred)in enumerate(zip(data,predictions)):
    #we will go throught the data , we make the point and its prediction as an enum 
    if pred == -1 :
        print(f"Anomaly at index {i}: {point.round(3)}")
        anomalies_found=True


# Chek if any anomalies wre detected
if not anomalies_found:
    print("No anomalies detected")