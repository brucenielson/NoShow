import os
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def main():
   create_train_test_files()


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
    X['ScheduledDay'] = pd.to_datetime(X['ScheduledDay'])
    X['AppointmentDay'] = pd.to_datetime(X['AppointmentDay'])
    X['DiffDates'] = (X['AppointmentDay'] - X['ScheduledDay'])
    # Drop unneeded columns
    X = X.drop(columns=['ScheduledDay', 'AppointmentDay'])
    print(X)
    # Encode result field
    Y = df[['No-show']]
    Y = label_encode(Y, 'No-show')

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    # Combine to save out to file
    x_train = x_train.copy()
    x_train['No-show'] = y_train

    x_test = x_test.copy()
    x_test['No-show'] = y_test

    x_train.to_csv('train.csv', index=False)
    x_test.to_csv('test.csv', index=False)

    print(x_train.head(5))
    print(len(x_train))
    print(x_test.head(5))
    print(len(x_test))

    file_name = os.getcwd() + '\\train.csv'
    train2 = pd.read_csv(file_name)
    file_name = os.getcwd() + '\\test.csv'
    test2 = pd.read_csv(file_name)
    print(train2.head(5))
    print(len(train2))
    print(test2.head(5))
    print(len(test2))

main()