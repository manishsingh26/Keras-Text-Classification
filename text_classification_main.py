import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras import layers

le = LabelEncoder()


class DataAnalysis(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path, delimiter=',', encoding='latin-1')
        df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

        print("Data Description :: ")
        print(df.describe())

        df["v2"] = df["v2"].str.replace("\d+", "")
        df["v2"] = df["v2"].str.replace("\W", " ")
        df["v1"] = le.fit_transform(df["v1"])

        sentences = df["v2"]
        y = df["v1"]

        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25,
                                                                            random_state=1000)
        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)
        return X_train, X_test, y_train, y_test


class NeuralConfiguration(object):

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def model_creation(self):

        input_dim = self.X_train.shape[1]

        print("Creating Sequential Model.")
        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        print("Adding Optimizer in the Model.")
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Fitting Data in th Model.")
        model.fit(self.X_train, self.y_train, epochs=20, verbose=True, batch_size=30)

        print("Working on Model Evaluation.")
        _, accuracy = model.evaluate(self.X_train, self.y_train)
        print('Accuracy: %.2f' % (accuracy * 100))

        print("Predicting Value for the Test Data-set.")
        predicted_data = model.predict(np.array(self.X_test))
        return predicted_data


if __name__ == "__main__":

    file_path_ = r"C:\Users\m4singh\Documents\AnalysisNoteBook\DeepLearning\TextClassification\spam.csv"

    file_obj = DataAnalysis(file_path_)
    X_train_, X_test_, y_train_, y_test_ = file_obj.load_data()

    neural_obj = NeuralConfiguration(X_train_, X_test_, y_train_, y_test_)
    predicted_data_ = neural_obj.model_creation()

    print(predicted_data_[:4])
