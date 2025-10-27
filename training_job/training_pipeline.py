from google.cloud import aiplatform
from google.oauth2 import service_account
import os
from dotenv import load_dotenv, dotenv_values
from datetime import datetime

load_dotenv()

env_variables = dict(dotenv_values('.env'))

print(dict(env_variables))

credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
)

aiplatform.init(
    project=os.getenv("GCP_PROJECT_ID"),
    location=os.getenv("GCP_REGION"),
    experiment=os.getenv("EXPERIMENT_NAME"),
    credentials=credentials,
)


training_job = aiplatform.CustomContainerTrainingJob(
    display_name=f"fraud-detection-training-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    container_uri="us-central1-docker.pkg.dev/poetic-velocity-459409-f2/vertex-training-repo/fraud-detection-trainer:latest",
    staging_bucket=os.getenv("GCP_BUCKET_NAME")
)

print(training_job)

print("Starting training job...")

operation = training_job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    environment_variables=env_variables,
)

print("Training job completed.")

