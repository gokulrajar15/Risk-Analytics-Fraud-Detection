from google.cloud import aiplatform
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from datetime import datetime
import joblib

load_dotenv()


GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
CODE_VERSION = os.getenv("CODE_VERSION", "v1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
TEAM_NAME = os.getenv("TEAM_NAME", "data-science")


aiplatform.init(
    project=GCP_PROJECT_ID,
    location=GCP_REGION,
    experiment=EXPERIMENT_NAME,
)

date = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"iris-logistic-regression-{CODE_VERSION}-{date}"

aiplatform.start_run(run=run_name)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy:.4f}")

aiplatform.log_params({"max_iter": 200})
aiplatform.log_metrics({"accuracy": accuracy})

# joblib.dump(model, "logistic_regression_model.pkl")

#################################################### Model Upload to GCS Bucket ####################################################


artifact_uri = f"gs://{os.getenv('GCP_BUCKET_NAME')}/{run_name}/"

print(f"Saving model to: {artifact_uri}")

response = aiplatform.save_model(
    model=model, 
    uri=artifact_uri, 
    display_name="My Model Artifact"
)


#################################################### Model Registry ####################################################

model_registry = os.getenv("MODEL_REGISTRY_NAME")
model_registry_name = f"{model_registry}-{CODE_VERSION}"
model_uri = response.uri
prediction_image = os.getenv("PREDICTION_IMAGE_URI")


existing_models = aiplatform.Model.list(filter=f'display_name="{model_registry_name}"')
if existing_models:
    parent_model = existing_models[0].resource_name
else:
    parent_model = None


model = aiplatform.Model.upload(
    display_name=model_registry_name,
    artifact_uri=model_uri,
    serving_container_image_uri=prediction_image,
    parent_model=parent_model,
    is_default_version=True,
    version_aliases=["dev"],
    version_description="Initial development version",
    labels={
        "team": TEAM_NAME,
        "environment": ENVIRONMENT,
        "version": CODE_VERSION
    }
)

print(f"Model '{model.display_name}' uploaded with version ID: {model.version_id}")

aiplatform.end_run()

#################################################### Model Evaluation ####################################################


#################################################### Model Deployment ####################################################

endpoint_display_name = f"fraud-detection-endpoint-{CODE_VERSION}"

existing_endpoints = aiplatform.Endpoint.list(
    filter=f'display_name="{endpoint_display_name}"',
    order_by="create_time"
)

if existing_endpoints:
    endpoint = existing_endpoints[0]
    print(f"Using existing endpoint: {endpoint.display_name}")
else:
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
    )
    print(f"Created new endpoint: {endpoint.display_name}")


deployed_model_name = f"fraud-detection-model-deployment-{CODE_VERSION}-{run_name}"

# Deploy the model to the endpoint
deployed_model = endpoint.deploy(
    model=model,
    deployed_model_display_name=deployed_model_name,
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    sync=True,
    enable_access_logging=True,
    deploy_request_timeout=1800
)

print(f"Model deployed to endpoint: {endpoint.resource_name}")
print(f"Endpoint ID: {endpoint.name}")
