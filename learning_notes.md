# MLOps Learning Notes

## Step 1: Git + DVC Foundation

### Why should Git track the `.dvc` file but not the CSV?

Git should track the `.dvc` file because it is small metadata. It acts like a pointer to the real dataset.

Git should not track the CSV directly because datasets can be large, change often, and make Git repositories slow and bloated over time.

In this project:

```text
Git tracks: data/raw/global_corporate_ghg_emissions_2022_2023.csv.dvc
DVC tracks: data/raw/global_corporate_ghg_emissions_2022_2023.csv
```

The `.dvc` file says, in effect:

```text
This project version depends on this exact dataset version.
```

### What does the hash inside the `.dvc` file help us guarantee?

The hash guarantees dataset identity.

If the CSV changes by even one character, the hash changes. That lets DVC detect whether we are using the exact same dataset version that produced a model, metric, or experiment.

This matters because ML results depend on data, not just code.

### Mental Model

```text
Git commit = code version + DVC pointer
DVC cache/storage = actual dataset bytes
.dvc hash = fingerprint of the dataset
```

Later, if we check out an old Git commit, the `.dvc` file tells DVC which exact dataset version to restore. That is one of the foundations of reproducible ML experiments.

## Step 2: Data Processing Module

### Why do we need `save_processed_data()`?

`save_processed_data()` exists so writing processed data to disk is isolated in one place.

This matters because later DVC will treat the processed dataset as a pipeline output:

```text
data/processed/modeling_dataset.csv
```

Keeping this logic in one function makes the output path explicit, reusable, and easy to change later.

### Why do we need `run_data_processing()`?

`run_data_processing()` is the orchestration function for the data processing stage.

It connects the smaller steps in the correct order:

```text
load raw data
validate columns
clean data
save processed data
return processed data
```

Without it, the file would only contain separate helper functions. With it, the module has one clear entry point that says:

```text
Run the full data processing stage.
```

Later, DVC can run this stage with:

```bash
python src/data_processing.py
```

That command internally calls:

```python
run_data_processing()
```

## Step 3: Training Module with MLflow

### What pattern does `train.py` follow?

`train.py` follows a simple pipeline orchestration pattern.

Each function has one responsibility:

```text
load_processed_data()  -> read the modeling dataset
build_model_pipeline() -> create preprocessing + model pipeline
evaluate_model()       -> calculate metrics
save_model()           -> write trained model artifact to disk
train_model()          -> orchestrate the full training run
```

This is not a heavy framework like Kubeflow, Airflow, or SageMaker Pipelines. It is a lightweight Python pipeline pattern that prepares the project for those tools later.

The structure is intentional:

```text
small testable functions + one orchestration function + script entry point
```

### Why do we use an sklearn `Pipeline`?

The sklearn `Pipeline` combines preprocessing and model training into one object:

```text
raw input features -> preprocessing -> trained model -> prediction
```

This matters because production inference must use the same transformations as training.

If training uses one preprocessing path and the API uses a different path, the model can receive data in a different shape or meaning than it learned from. This is called training/serving skew.

Saving the full pipeline helps prevent that failure mode.

### Why do we use MLflow?

MLflow tracks experiment metadata.

In this training step, MLflow logs:

```text
parameters -> model settings, such as n_estimators and max_depth
metrics    -> model results, such as mae, rmse, and r2
artifacts  -> saved model outputs
```

This lets us compare runs instead of relying on memory, screenshots, or scattered notebook outputs.

### Method Flow in `train.py`

When we run:

```bash
python src/train.py
```

Python reaches:

```python
if __name__ == "__main__":
    training_metrics = train_model()
```

Then `train_model()` runs the full workflow:

```text
1. Load processed data from data/processed/modeling_dataset.csv
2. Split dataframe into X features and y target
3. Split X/y into train and test sets
4. Build the sklearn Pipeline
5. Start an MLflow experiment run
6. Log model parameters
7. Fit the pipeline on training data
8. Evaluate the pipeline on test data
9. Log metrics to MLflow
10. Save the pipeline locally to models/model.pkl
11. Log the model artifact to MLflow
12. Return metrics for terminal output
```

### Mental Model

```text
DVC answers: Which data version did we use?
Git answers: Which code version did we use?
MLflow answers: What happened during this training run?
```

Together, these create experiment reproducibility.

## Step 4: Prediction Module

### What does `predict.py` do?

`predict.py` loads the trained model pipeline and uses it to make predictions for new input data.

The flow is:

```text
load saved model
convert input dictionary to pandas DataFrame
run model.predict()
return one prediction number
```

Because we saved the full sklearn `Pipeline`, prediction automatically includes:

```text
preprocessing -> model prediction
```

This means `predict.py` does not need to manually one-hot encode categories or scale numeric values. The saved pipeline handles those steps.

### Why do we convert input data into a DataFrame?

API/user input is often convenient as a dictionary:

```python
{
    "country": "Australia",
    "sector": "Energy",
    "revenue_usd_millions": 16816.0,
}
```

But the trained sklearn pipeline expects tabular input:

```text
rows = companies
columns = features
```

Even for one prediction, we pass a one-row DataFrame:

```text
1 row x feature columns
```

This lets the pipeline match column names to the preprocessing rules learned during training.

### Why return `float(prediction[0])`?

sklearn returns predictions in an array-like structure, even for a single input row:

```python
[7.6090]
```

`prediction[0]` gets the first prediction.

`float(prediction[0])` converts it into a normal Python float, which is easier to return from an API as JSON.

### Processed Dataset vs New Input

The processed training dataset contains both:

```text
input features
target/output variable
```

During training:

```text
X = input features
y = target/output variable
```

For this project:

```text
X = company attributes + emissions-related fields
y = scope1_plus_scope2_location_mt
```

For a new prediction request, we only provide `X`. We do not provide the target, because that is what the model is trying to predict.

Mental model:

```text
training data = practice problems with answer key
new input = test question without answer key
model output = predicted answer
```

### Environment Mismatch Lesson

We hit this error:

```text
AttributeError: Can't get attribute '_RemainderColsList'
```

This happened because the model was trained in one Python/sklearn environment but loaded from another.

The model was trained using the project virtual environment:

```text
.venv/
Python 3.9
scikit-learn 1.6.1
```

But prediction was first run with a different environment:

```text
miniconda Python 3.12
different scikit-learn package path
```

Pickled sklearn models are version-sensitive. The loading environment needs compatible package versions with the training environment.

The fix was to run prediction with the project venv:

```bash
.venv/bin/python src/predict.py
```

MLOps lesson:

```text
model artifacts are tied to their dependency environment
```

This is one reason we use:

```text
requirements.txt
Docker
CI/CD checks
```

Those tools help make training and serving environments reproducible.

## Step 5: Docker and Containerization

### Why do we use Docker?

Docker packages the API, model-serving code, model artifact, and Python dependencies into a reproducible runtime environment.

This matters because local Python environments can drift. We already saw that a model trained in one sklearn/Python environment can fail when loaded from another environment.

Docker helps answer:

```text
Can this API run outside my local virtual environment?
```

### What goes in `requirements.txt`?

`requirements.txt` lists Python packages needed by the app inside the container.

Examples:

```text
fastapi
uvicorn
pandas
scikit-learn
mlflow
joblib
```

Docker itself does not go in `requirements.txt` because Docker is not a Python package used by our app. Docker is system tooling installed on the machine that builds and runs containers.

Mental model:

```text
requirements.txt = what Python needs inside the app
Docker Desktop = what the computer needs to build and run containers
```

### Error: `command not found: docker`

This meant Docker Desktop was either not installed or the Docker CLI was not on the shell `PATH`.

We verified Docker existed here:

```bash
/Applications/Docker.app/Contents/Resources/bin/docker --version
```

Then we added Docker's CLI directory to `~/.zshrc`:

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

Lesson:

```text
Docker Desktop can be installed, but the terminal still needs access to the docker CLI.
```

### Error: Docker Desktop running but no Server section

We saw:

```text
failed to connect to the docker API at unix:///Users/arjunsundaram/.docker/run/docker.sock
```

`docker version` showed a `Client` section but no working `Server` section.

That meant:

```text
Docker CLI was installed
Docker Desktop UI was open
Docker Engine/daemon was not actually ready
```

The Docker daemon is the backend service that actually builds and runs containers. The CLI is only the client that talks to it.

Mental model:

```text
docker CLI = client
Docker Engine/daemon = server
docker.sock = communication channel between them
```

For Docker to work, `docker version` should show both:

```text
Client:
Server:
```

### Error: Docker build permission denied under `~/.docker`

During `docker compose build`, Docker needed to write build metadata under:

```text
~/.docker/buildx/
```

The sandbox initially blocked that write.

This was not a project-code issue. Docker normally needs access to its own local metadata directories while building images.

Lesson:

```text
Docker builds use both project files and Docker's own local build state.
```

### Error: invalid container port in `docker-compose.yml`

We saw:

```text
invalid containerPort:  8000
```

The issue was a small formatting mistake:

```yaml
- "8001: 8000"
```

There was a space after the colon.

The correct version is:

```yaml
- "8001:8000"
```

Meaning:

```text
host port 8001 -> container port 8000
```

Lesson:

```text
Docker Compose port mappings are sensitive to formatting.
```

### Error: port 8000 already in use

When running the FastAPI app locally, port `8000` was already occupied by another service.

We used port `8001` for this project.

Mental model:

```text
Only one process can listen on a host port at a time.
```

In Docker Compose, this mapping solved it:

```yaml
ports:
  - "8001:8000"
```

That means:

```text
access the API on localhost:8001
inside the container, the app still listens on port 8000
```

### Successful Container Test

After the image built and the container started, we tested:

```bash
curl http://127.0.0.1:8001/health
```

Response:

```json
{"status":"ok"}
```

We also tested:

```bash
POST http://127.0.0.1:8001/predict
```

Response:

```json
{"predicted_scope_1_plus_scope_2_location_mt":7.608954639355733}
```

This proved:

```text
FastAPI app runs inside Docker
model artifact loads inside Docker
prediction endpoint works from the container
```
