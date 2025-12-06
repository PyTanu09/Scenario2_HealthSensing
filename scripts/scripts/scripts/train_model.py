import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 5, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 5, activation="relu"),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def build_convlstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, 5, activation="relu", input_shape=input_shape),
        MaxPooling1D(2),
        LSTM(64),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def run_lopo(data, model_fn):
    participants = data["participant"].unique()
    results = []

    X = np.stack(data["values"].apply(lambda x: np.array(eval(x))).to_numpy())
    X = X.reshape((X.shape[0], X.shape[1], 1))

    y = LabelEncoder().fit_transform(data["label"])
    y_cat = to_categorical(y)

    for test_p in participants:
        train_idx = data["participant"] != test_p
        test_idx = data["participant"] == test_p

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        model = model_fn(X_train.shape[1:], y_cat.shape[1])
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        preds = model.predict(X_test)
        preds = preds.argmax(axis=1)
        true = y[test_idx]

        cm = confusion_matrix(true, preds)
        print(f"\nParticipant {test_p} Confusion Matrix:")
        print(cm)

        results.append(cm)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    print("\nRunning 1D CNN...\n")
    run_lopo(df, build_cnn)

    print("\nRunning Conv-LSTM...\n")
    run_lopo(df, build_convlstm)

if __name__ == "__main__":
    main()
