from google.cloud import aiplatform
from google.oauth2 import service_account
import os
from dotenv import load_dotenv, dotenv_values
from datetime import datetime

load_dotenv()

env_variables = dict(dotenv_values('.env'))

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
    container_uri=os.getenv("CONTAINER_IMAGE_URI"),
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

