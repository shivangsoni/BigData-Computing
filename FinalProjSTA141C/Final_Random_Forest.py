import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

filter = np.genfromtxt('/home/sai/PycharmProjects/sta141c/filter_s3_1_b_new.csv', delimiter=",")
filter = np.delete(filter, (0), axis=0)
np.random.shuffle(filter)
train=filter[0:filter.shape[0]*7//10,:];
test=filter[filter.shape[0]*7//10:filter.shape[0],:];
P = train[:, 0:3]
q = train[:, 3]
R = test[:, 0:3]
s = test[:, 3]
s = (np.asmatrix(s)).T


def rfc(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def main():
    # Create random forest classifier instance
    trained_model = rfc(P, q)
    predictions = trained_model.predict(R)

    # Train and Test Accuracy
    print("Train Accuracy :: ", accuracy_score(q, trained_model.predict(P)) * 100)
    print("Test Accuracy  :: ", accuracy_score(s, predictions) * 100)


if __name__ == "__main__":
    main()

