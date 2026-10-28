# Machine Failure Prediction API

Predicts whether a machine will fail in the next 24 hours, using the Microsoft Azure Predictive Maintenance dataset from Kaggle.

## What it does
- Model: Random Forest trained in Google Colab on rolling-window sensor features (3h and 24h mean/std), error counts and maintenance history. Data was split by time (train, validation, test) because failures are rare and we predict the future.
- The trained pipeline, decision threshold and feature list are saved together in `model.joblib`.
- API: FastAPI with two endpoints, `GET /health` and `POST /predict`. `/predict` takes the last 24 hourly readings of a machine and returns the failure probability and an alert flag.

## Run with Docker
- Build: `docker build -t pdm-api .`
- Run: `docker run -p 8000:8000 pdm-api`
- Then open http://localhost:8000/docs

## CI
GitHub Actions runs the tests, builds the Docker image, starts the container and calls `/health` and `/predict` inside it.
