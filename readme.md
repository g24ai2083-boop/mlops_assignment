# MLOps Major Assignment – Olivetti Faces

**Name:** Kavuri Guru Naga Venkata Bharadwaj Sharma  
**Roll No:** G24AI2083  

This project implements an end-to-end MLOps pipeline:

- Uses the Olivetti faces dataset from `sklearn.datasets`
- Trains a `DecisionTreeClassifier`
- Saves the model with `joblib`
- Provides `train.py` and `test.py`
- CI pipeline using GitHub Actions (`.github/workflows/ci.yml`)
- Flask app (`app.py`) for image upload and prediction
- Containerization with Docker (`Dockerfile`)
- Kubernetes deployment (`deployment.yml`) with 3 replicas

## Branch Strategy

- `main`: Initial setup (`README.md`, `.gitignore`)
- `dev`: Model training, testing, CI pipeline
- `docker_cicd`: Flask app, Dockerfile, Kubernetes manifests

## Local Usage

```bash
pip install -r requirements.txt
python train.py
python test.py
python app.py
