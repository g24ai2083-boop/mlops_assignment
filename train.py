# train.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def main():
    # Load Olivetti faces dataset
    data = fetch_olivetti_faces()
    X = data.data        # shape: (n_samples, 4096)
    y = data.target      # labels: 0–39

    # Train-test split: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Train a DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "savedmodel.pth")
    print("✅ Model trained and saved as savedmodel.pth")

if __name__ == "__main__":
    main()
