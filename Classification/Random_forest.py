import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#loading data 

bank = fetch_ucirepo(id=222)
bank_data = bank.data.original

#we map every no to 0 and yes to 1 and unkown to 0
bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})
bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})

#splitting data as source data and target data 
X = pd.get_dummies(bank_data.drop('y', axis=1), drop_first=True) #source 
y = bank_data['y'] #target

#splitting data for testing and training , 80% for training and 20% for training 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#fitting the model with the data 
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

#applying the prediction
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

for i in range(3):
    plt.figure(figsize=(20, 10))
    plot_tree(
        rf.estimators_[i],
        feature_names=X_train.columns,
        class_names=["no", "yes"],
        filled=True,
        max_depth=2,
        impurity=False,
        proportion=True,
        fontsize=10,
    )
    plt.title(f"Tree {i}")
    plt.savefig(f"tree_{i}.png", dpi=150, bbox_inches="tight")
plt.show()
