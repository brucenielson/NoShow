import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    # create_train_test_files()
    X, Y = load_data('train.csv')
    x_train, y_train, x_validate, y_validate = split_train_validation(X,Y)
    clf = train(x_train, y_train)
    y_pred = clf.predict(x_train)
    # returns statistics
    print("")
    print("Training Set:")
    print_statistics(y_pred, y_train)

    # Check validation set
    y_valdiate_pred = clf.predict(x_validate)
    print("")
    print("Validation Set:")
    print_statistics(y_valdiate_pred, y_validate)

    # How well does it make predictions?
    y_valdiate_pred_prob = clf.predict_proba(x_validate)[:,1]
    print("")
    print("Strongest Prediction: %.2f" % np.max(y_valdiate_pred_prob))
    print("Strongest Prediction Index: %d" % np.argmax(y_valdiate_pred_prob))

    # But how often do people no show?
    print("")
    no_shows = float(y_train.sum()) / float(len(y_train))
    print('No Shows in Train Set: %.2f' % no_shows)
    pred_no_shows = float(y_pred.sum()) / float(len(y_pred))
    print('Predicted No Shows: %.2f' % pred_no_shows)

    # List of predictions
    y_predictions_data = pd.DataFrame(y_valdiate_pred_prob, columns=['Probs'])
    y_predictions_data['Predicted'] = y_valdiate_pred
    y_predictions_data['Actual'] = y_validate
    y_predictions_data = y_predictions_data[y_predictions_data['Predicted'] == 1]
    print("")
    print("List of Predictions:")
    print(y_predictions_data)
    print_statistics(y_predictions_data['Predicted'], y_predictions_data['Actual'])




def print_statistics(y_pred, y_actuals):
    print('Misclassified samples: %d out of %d' % ((y_actuals != y_pred).sum(), len(y_actuals)))
    print('Accuracy of set: %.2f' % accuracy_score(y_actuals, y_pred))



def train(x_train, y_train):
    clf = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
    clf.fit(x_train, y_train)
    return clf


def split_train_validation(X, Y):
    # Always split validation set in a consistent way
    size = len(X)
    split_at = int(size * 0.8)
    x_train = X[0:split_at]
    x_validate = X[split_at:]
    y_train = Y[0:split_at]
    y_validate = Y[split_at:]
    return x_train, y_train.to_numpy().flatten(), x_validate, y_validate.to_numpy().flatten()



def load_data(file_name='train.csv'):
    file_name = os.getcwd() + '\\train.csv'
    df = pd.read_csv(file_name)
    return df[df.columns[0:len(df.columns)-1]], df[[df.columns[-1]]]



def label_encode(df, column_name):
    df = df.copy()
    le = preprocessing.LabelEncoder()
    le.fit(df[column_name].unique())
    df[column_name] = le.transform(df[column_name])
    return df


def create_train_test_files():
    # Open original file
    file_name = os.getcwd() + '\\NoShow1.csv'
    df = pd.read_csv(file_name)
    # df = sk.utils.shuffle(df)

    # Take all the relevant columns except unneeded ids and the correct answer
    X = df.loc[:, df.columns[2:13]]
    # encode columns with labeler (i.e. not one hot encoding)
    X = label_encode(X, 'Gender')
    # Calculate difference between schedule and appointment
    X['ScheduledDay'] = pd.to_datetime(X['ScheduledDay']).dt.date
    X['AppointmentDay'] = pd.to_datetime(X['AppointmentDay']).dt.date
    X['DiffDates'] = (X['AppointmentDay'] - X['ScheduledDay']).dt.days
    # Drop unneeded columns
    X = X.drop(columns=['ScheduledDay', 'AppointmentDay'])
    # Do one hot encoding on Neighbourhood
    X = pd.get_dummies(X, columns=['Neighbourhood'])

    # Scale and center
    col_names = ['Age', 'DiffDates']
    features = X[col_names]
    scaler = preprocessing.StandardScaler().fit(features.values)
    scaled = scaler.transform(features.values)
    X[col_names] = scaled

    # Encode result field
    Y = df[['No-show']]
    Y = label_encode(Y, 'No-show')

    # Random train/test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # Combine back to save out to file
    x_train = x_train.copy()
    x_train['No-show'] = y_train
    x_test = x_test.copy()
    x_test['No-show'] = y_test
    # Save as csv files
    x_train.to_csv('train.csv', index=False)
    x_test.to_csv('test.csv', index=False)

    # Read files back to be sure they worked
    file_name = os.getcwd() + '\\train.csv'
    train2 = pd.read_csv(file_name)
    file_name = os.getcwd() + '\\test.csv'
    test2 = pd.read_csv(file_name)
    assert len(x_train) == len(train2)
    assert len(y_train) == len(train2)
    assert len(x_test) == len(test2)
    assert len(y_test) == len(test2)



main()