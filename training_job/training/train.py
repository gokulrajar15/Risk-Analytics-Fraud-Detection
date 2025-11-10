from google.cloud import aiplatform, bigquery
from google.oauth2 import service_account
import os
import sys
import math
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from datetime import datetime
from xgboost import XGBClassifier
from collections import Counter
from logger import setup_logging

load_dotenv()

logger = setup_logging()

# Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
CODE_VERSION = os.getenv("CODE_VERSION", "v1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
TEAM_NAME = os.getenv("TEAM_NAME", "data-science")

logger.info(f"Starting fraud detection training pipeline - Version: {CODE_VERSION}, Environment: {ENVIRONMENT}")

# Initialize clients
try:
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    bigquery_client = bigquery.Client(credentials=credentials)
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, experiment=EXPERIMENT_NAME)
    logger.info("Successfully initialized GCP clients")
except Exception as e:
    logger.error(f"Failed to initialize GCP clients: {e}")
    sys.exit(1)

date = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"fraud-detection-{CODE_VERSION}-{date}"

try:
    aiplatform.start_run(run=run_name)
    logger.info(f"Started Vertex AI experiment run: {run_name}")
except Exception as e:
    logger.error(f"Failed to start Vertex AI run: {e}")
    sys.exit(1)

#################################################### 1. DATA PREPARATION & VALIDATION ####################################################

logger.info("=" * 80)
logger.info("STEP 1: DATA PREPARATION & VALIDATION")
logger.info("=" * 80)

def load_and_validate_data():
    """Load data from BigQuery with validation"""
    try:
        biqquery_path = os.getenv("BIGQUERY_PATH")

        query = f"SELECT * FROM `{biqquery_path}`"
        df = bigquery_client.query(query).to_dataframe()
        
        rows, columns = df.shape
        missing_values = df.isnull().sum().sum()
        fraud_rate = df['fraud'].mean()
        
        logger.info(f"Data successfully loaded: {rows:,} rows, {columns} columns")
        logger.info(f"Data quality check - Missing values: {missing_values}")
        logger.info(f"Data distribution - Fraud rate: {fraud_rate:.3%}")
        
        aiplatform.log_metrics({
            "data_rows": rows,
            "data_columns": columns,
            "missing_values": int(missing_values),
            "fraud_rate": float(fraud_rate)
        })
        
        if missing_values > 0:
            logger.warning(f"Dataset contains {missing_values} missing values")
        
        if fraud_rate < 0.001 or fraud_rate > 0.5:
            logger.warning(f"Unusual fraud rate detected: {fraud_rate:.3%}")
        
        return df
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        aiplatform.log_params({"training_status": "FAILED"})
        aiplatform.end_run()
        sys.exit(1)

def prepare_features(df):
    """Feature preparation and validation"""
    logger.info("Preparing features and target variable")
    
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    feature_count = X.shape[1]
    target_distribution = Counter(y)
    
    logger.info(f"Feature preparation complete - {feature_count} features")
    logger.info(f"Target distribution: {dict(target_distribution)}")
    
    return X, y

# Execute data preparation
df = load_and_validate_data()
X, y = prepare_features(df)

#################################################### 2. DATA SPLITTING & PREPROCESSING ####################################################

logger.info("=" * 80)
logger.info("STEP 2: DATA SPLITTING & PREPROCESSING")
logger.info("=" * 80)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=0.8, 
        stratify=y, 
        random_state=42
    )
    
    # Calculate class weights
    counter = Counter(y_train)
    negative = counter[0]
    positive = counter[1]
    scale_pos_weight = negative / positive
    
    logger.info("Data split completed successfully")
    logger.info(f"Training set: {X_train.shape[0]:,} samples")
    logger.info(f"Test set: {X_test.shape[0]:,} samples")
    logger.info(f"Class balance - Negative: {negative:,}, Positive: {positive:,}")
    logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.3f}")
    
    aiplatform.log_params({
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "scale_pos_weight": scale_pos_weight
    })
    
except Exception as e:
    logger.error(f"Data splitting failed: {e}")
    aiplatform.log_params({"training_status": "FAILED"})
    aiplatform.end_run()
    sys.exit(1)

#################################################### 3. MODEL TRAINING ####################################################

logger.info("=" * 80)
logger.info("STEP 3: MODEL TRAINING")
logger.info("=" * 80)

try:
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )
    
    model_params = model.get_params()
    filtered_model_params = {key: value for key, value in model_params.items() if value is not None and not (isinstance(value, float) and math.isnan(value))}

    print(f"Model parameters: {filtered_model_params}")
    aiplatform.log_params(filtered_model_params)

    logger.info("Model configuration:")
    for param, value in filtered_model_params.items():
        logger.info(f"  {param}: {value}")
    
    logger.info("Starting model training...")
    training_start_time = datetime.now()
    
    model.fit(X_train, y_train)
    
    aiplatform.log_model(
    model=model,
    display_name=f"Fraud Detection Model {CODE_VERSION}"
    )

    training_duration = (datetime.now() - training_start_time).total_seconds()
    logger.info(f"Model training completed successfully in {training_duration:.2f} seconds")
    
    aiplatform.log_metrics({"training_duration_seconds": training_duration})
    
except Exception as e:
    logger.error(f"Model training failed: {e}")
    aiplatform.log_params({"training_status": "FAILED"})
    aiplatform.end_run()
    sys.exit(1)

#################################################### 4. MODEL EVALUATION & VALIDATION ####################################################

logger.info("=" * 80)
logger.info("STEP 4: MODEL EVALUATION & VALIDATION")
logger.info("=" * 80)

def comprehensive_evaluation(model, X_train, y_train, X_test, y_test):
    """Comprehensive model evaluation with production gates"""
    
    logger.info("Running cross-validation on training data...")
    cv_start_time = datetime.now()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    cv_duration = (datetime.now() - cv_start_time).total_seconds()
    logger.info(f"Cross-validation completed in {cv_duration:.2f} seconds")
    
    logger.info("Evaluating model on test set...")
    eval_start_time = datetime.now()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    eval_duration = (datetime.now() - eval_start_time).total_seconds()
    logger.info(f"Test set evaluation completed in {eval_duration:.2f} seconds")
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        # "cv_f1_mean": cv_scores.mean(),
        # "cv_f1_std": cv_scores.std()
    }
    
    logger.info("Model evaluation metrics calculated successfully")
    
    return metrics, y_pred, y_pred_proba

def production_quality_gates(metrics):
    """Production deployment quality gates"""
    
    QUALITY_GATES = {
        "min_f1_score": 0.70,     
        "min_recall": 0.65,       
        "min_precision": 0.55,    
        "min_auc_roc": 0.75,      
        # "max_cv_std": 0.15     
    }
    
    logger.info("Checking production quality gates...")
    logger.info("Quality gate thresholds:")
    for gate, threshold in QUALITY_GATES.items():
        logger.info(f"  {gate}: {threshold}")
    
    failures = []
    
    if metrics["f1_score"] < QUALITY_GATES["min_f1_score"]:
        failures.append(f"F1-score {metrics['f1_score']:.4f} < {QUALITY_GATES['min_f1_score']}")
    
    if metrics["recall"] < QUALITY_GATES["min_recall"]:
        failures.append(f"Recall {metrics['recall']:.4f} < {QUALITY_GATES['min_recall']}")
    
    if metrics["precision"] < QUALITY_GATES["min_precision"]:
        failures.append(f"Precision {metrics['precision']:.4f} < {QUALITY_GATES['min_precision']}")
    
    if metrics["auc_roc"] < QUALITY_GATES["min_auc_roc"]:
        failures.append(f"AUC-ROC {metrics['auc_roc']:.4f} < {QUALITY_GATES['min_auc_roc']}")
    
    # if metrics["cv_f1_std"] > QUALITY_GATES["max_cv_std"]:
    #     failures.append(f"CV F1 std {metrics['cv_f1_std']:.4f} > {QUALITY_GATES['max_cv_std']}")
    
    return len(failures) == 0, failures

try:
    metrics, y_pred, y_pred_proba = comprehensive_evaluation(model, X_train, y_train, X_test, y_test)
    
    aiplatform.log_metrics(metrics)
    
    logger.info("MODEL PERFORMANCE SUMMARY:")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    # logger.info(f"CV F1:     {metrics['cv_f1_mean']:.4f} (±{metrics['cv_f1_std']:.4f})")
    

    cm = confusion_matrix(y_test, y_pred)
    logger.info("CONFUSION MATRIX:")
    logger.info(f"True Negatives: {cm[0,0]:,}, False Positives: {cm[0,1]:,}")
    logger.info(f"False Negatives: {cm[1,0]:,}, True Positives: {cm[1,1]:,}")

except Exception as e:
    logger.error(f"Model evaluation failed: {e}")
    aiplatform.log_params({"training_status": "FAILED"})
    aiplatform.end_run()
    sys.exit(1)

#################################################### 5. QUALITY GATES & DEPLOYMENT DECISION ####################################################

logger.info("=" * 80)
logger.info("STEP 5: QUALITY GATES & DEPLOYMENT DECISION")
logger.info("=" * 80)

try:
    gates_passed, failures = production_quality_gates(metrics)
    
    if not gates_passed:
        logger.error("QUALITY GATES FAILED - Model will not be deployed")
        for failure in failures:
            logger.error(f"  Gate failure: {failure}")
        
        aiplatform.log_params({
            "deployment_approved": False,
            "gate_failures": "; ".join(failures),
            "deployment_status": "REJECTED"
        })
        
        logger.error("MODEL DEPLOYMENT BLOCKED due to quality gate failures")
        aiplatform.end_run()
        sys.exit(1)
    
    logger.info("ALL QUALITY GATES PASSED - Model approved for deployment")
    aiplatform.log_params({
        "deployment_approved": True,
        "deployment_status": "APPROVED"
    })
    
except Exception as e:
    logger.error(f"Quality gates check failed: {e}")
    aiplatform.log_params({"training_status": "FAILED"})
    aiplatform.end_run()
    sys.exit(1)

#################################################### 6. MODEL ARTIFACT STORAGE ####################################################

logger.info("=" * 80)
logger.info("STEP 6: MODEL ARTIFACT STORAGE")
logger.info("=" * 80)

try:
    artifact_uri = f"gs://{os.getenv('GCP_BUCKET_NAME')}/{run_name}/"

    response = aiplatform.save_model(
        model=model, 
        uri=artifact_uri, 
        display_name=f"Fraud Detection Model {CODE_VERSION}"
    )
    
    logger.info("Model artifact saved successfully to Cloud Storage")
    
except Exception as e:
    logger.error(f"Model artifact storage failed: {e}")
    sys.exit(1)

#################################################### 7. MODEL REGISTRY ####################################################

logger.info("=" * 80)
logger.info("STEP 7: MODEL REGISTRY")
logger.info("=" * 80)

try:
    model_registry = os.getenv("MODEL_REGISTRY_NAME")
    model_registry_name = f"{model_registry}-{CODE_VERSION}"
    model_uri = response.uri
    prediction_image = os.getenv("PREDICTION_IMAGE")
    
    # Check for existing models
    existing_models = aiplatform.Model.list(filter=f'display_name="{model_registry_name}"')
    parent_model = existing_models[0].resource_name if existing_models else None
    
    if parent_model:
        logger.info(f"Found existing model parent: {parent_model}")
    else:
        logger.info("Creating new model in registry")
    
    logger.info(f"Registering model: {model_registry_name}")
    
    registered_model = aiplatform.Model.upload(
        display_name=model_registry_name,
        artifact_uri=model_uri,
        serving_container_image_uri=prediction_image,
        parent_model=parent_model,
        is_default_version=False,
        version_aliases=[ENVIRONMENT],
        version_description=run_name,
        labels={
            "experiment_name": EXPERIMENT_NAME,
            "team": TEAM_NAME,
            "environment": ENVIRONMENT,
            "version": CODE_VERSION,
            "model_type": "fraud_detection",
            "algorithm": "xgboost"
        }
    )

    aiplatform.log_params({
    "model_id": registered_model.name,
    "model_version": registered_model.version_id,
    "version_description": registered_model.version_description,
    "artifact_uri": model_uri
    })
        
    logger.info(f"Model successfully registered with version ID: {registered_model.version_id}")
    
except Exception as e:
    logger.error(f"Model registry operation failed: {e}")
    aiplatform.log_params({"training_status": "FAILED"})
    aiplatform.end_run()
    sys.exit(1)

#################################################### 8. DEPLOYMENT PREPARATION ####################################################

logger.info("=" * 80)
logger.info("STEP 8: DEPLOYMENT PREPARATION")
logger.info("=" * 80)

try:
    # Log deployment metadata
    deployment_metadata = {
        "model_version": registered_model.version_id,
        "deployment_timestamp": datetime.now().isoformat(),
        "deployment_strategy": "manual_approval_required"
    }
    
    aiplatform.log_params(deployment_metadata)
    
    logger.info("Model ready for deployment")
    logger.info(f"Model Version: {registered_model.version_id}")
    logger.info(f"Performance Summary: F1={metrics['f1_score']:.4f}, AUC={metrics['auc_roc']:.4f}")
    
    aiplatform.end_run()
    
    logger.info("Training pipeline completed successfully!")
    logger.info("Next step: Deploy model using separate deployment pipeline")
    
except Exception as e:
    logger.error(f"Deployment preparation failed: {e}")
    sys.exit(1)

logger.info("=" * 80)
logger.info("FRAUD DETECTION TRAINING PIPELINE COMPLETED")
logger.info("=" * 80)