# test.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import joblib

def main():
    # Load dataset
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target

    # Same split as in train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Load saved model
    model = joblib.load("savedmodel.pth")

    # Evaluate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"✅ Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
