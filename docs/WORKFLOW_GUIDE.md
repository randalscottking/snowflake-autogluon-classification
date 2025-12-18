# Snowflake ML Workflow Guide

Comprehensive end-to-end ML pipeline for classification tasks in Snowflake with automated feature selection, AutoGluon training, and model deployment.

## Features

- **Automated Feature Selection**: Uses Pearson correlation to remove redundant features
- **Feature Store Integration**: Registers curated features in Snowflake Feature Store
- **AutoML Training**: Leverages AutoGluon to train multiple models automatically
- **Intelligent Model Selection**: Chooses best model based on F1, Recall, and Precision
- **Dual Model Registry**: Registers models to both Warehouse and Container Services
- **Production Deployment**: Optional deployment to Snowflake Container Services

## Prerequisites

### Snowflake Setup
1. **Databases**: Create necessary databases
   ```sql
   CREATE DATABASE IF NOT EXISTS ML_FEATURES;
   CREATE DATABASE IF NOT EXISTS ML_MODELS;
   CREATE DATABASE IF NOT EXISTS ML_DATA;
   ```

2. **Packages**: Ensure these Python packages are available
   - snowflake-snowpark-python
   - snowflake-ml-python
   - autogluon.tabular
   - pandas
   - numpy
   - scikit-learn

3. **Compute Pool** (for Container Services deployment)
   ```sql
   CREATE COMPUTE POOL IF NOT EXISTS ML_COMPUTE_POOL
   MIN_NODES = 1
   MAX_NODES = 3
   INSTANCE_FAMILY = CPU_X64_M;
   ```

### Data Requirements
Your source table should:
- Be structured with numeric and/or categorical features
- Have a target column for classification
- Include an entity identifier column (e.g., CUSTOMER_ID, USER_ID)

## Quick Start

### Option 1: Python Script Execution

```python
from snowflake.snowpark import Session
from src.snowflake_ml_workflow import SnowflakeMLWorkflow

# Create Snowflake session
session = Session.builder.configs({
    "account": "your_account",
    "user": "your_user",
    "password": "your_password",
    "role": "your_role",
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema"
}).create()

# Initialize workflow
workflow = SnowflakeMLWorkflow(
    session=session,
    source_table="ML_DATA.PUBLIC.CUSTOMER_FEATURES",
    target_column="CHURN_FLAG",
    correlation_threshold=0.95
)

# Execute complete workflow
results = workflow.run_complete_workflow(
    entity_column="CUSTOMER_ID",
    feature_view_name="customer_churn_features",
    model_name="customer_churn_classifier",
    test_size=0.2,
    time_limit=3600
)
```

### Option 2: Snowflake Stored Procedure

1. **Create the stored procedure**:
   ```sql
   CREATE OR REPLACE PROCEDURE run_ml_workflow(
       source_table VARCHAR,
       target_column VARCHAR,
       entity_column VARCHAR,
       model_name VARCHAR
   )
   RETURNS VARIANT
   LANGUAGE PYTHON
   RUNTIME_VERSION = '3.9'
   PACKAGES = ('snowflake-snowpark-python==1.11.1', 
               'snowflake-ml-python==1.1.2',
               'autogluon.tabular==0.8.2', 
               'pandas', 'numpy', 'scikit-learn')
   HANDLER = 'main'
   AS
   $$
   # Paste the entire content of src/snowflake_ml_workflow.py here
   $$;
   ```

2. **Execute the workflow**:
   ```sql
   CALL run_ml_workflow(
       'ML_DATA.PUBLIC.CUSTOMER_FEATURES',
       'CHURN_FLAG',
       'CUSTOMER_ID',
       'customer_churn_classifier'
   );
   ```

## Workflow Steps

### 1. Data Loading
```python
df = workflow.load_data()
```
Loads data from the specified source table.

### 2. Feature Selection
```python
selected_features, report = workflow.select_features_by_correlation(df)
```
- Calculates Pearson correlation between all numeric features
- Removes features with correlation > threshold (default: 0.95)
- Returns selected features and detailed correlation report

### 3. Feature Store Registration
```python
feature_view = workflow.create_feature_view(
    df=df,
    selected_features=selected_features,
    entity_column="CUSTOMER_ID",
    feature_view_name="my_features"
)
```
- Creates a Feature View with selected features
- Registers in Snowflake Feature Store
- Sets up automatic refresh schedule

### 4. Model Training
```python
predictor = workflow.train_autogluon_models(
    train_data=train_df,
    time_limit=3600,
    presets="best_quality"
)
```
- Trains multiple models using AutoGluon
- Includes: LightGBM, CatBoost, XGBoost, Random Forest, Neural Networks
- Applies bagging and stacking for ensemble models

### 5. Model Evaluation
```python
evaluation = workflow.evaluate_and_select_best_model(
    predictor=predictor,
    test_data=test_df
)
```
- Evaluates all models on test data
- Selects best model based on:
  1. F1 Score (primary)
  2. Recall (secondary)
  3. Precision (tertiary)

### 6. Model Registry
```python
model_version = workflow.register_model_to_warehouse(
    predictor=predictor,
    model_name="my_model",
    best_model_name=best_model,
    metrics=evaluation,
    feature_list=selected_features
)
```
- Registers model to Snowflake Model Registry
- Stores metadata: metrics, features, timestamp
- Enables model versioning and lineage tracking

### 7. Container Deployment (Optional)
```python
endpoint = workflow.deploy_to_container_services(
    model_name="my_model",
    model_version=model_version,
    compute_pool="ML_COMPUTE_POOL",
    service_name="my_prediction_service"
)
```
- Deploys model to Snowflake Container Services
- Creates REST API endpoint for predictions
- Auto-scales based on load

## Configuration Options

### Workflow Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_table` | str | Required | Fully qualified table name |
| `target_column` | str | Required | Target variable column name |
| `correlation_threshold` | float | 0.95 | Threshold for dropping correlated features |
| `feature_store_db` | str | "ML_FEATURES" | Database for Feature Store |
| `model_registry_db` | str | "ML_MODELS" | Database for Model Registry |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `time_limit` | int | 3600 | Training time limit in seconds |
| `presets` | str | "best_quality" | AutoGluon preset quality level |
| `num_bag_folds` | int | 5 | Number of bagging folds |
| `num_stack_levels` | int | 1 | Number of stacking levels |
| `test_size` | float | 0.2 | Proportion of data for testing |

### AutoGluon Presets

- **`best_quality`**: Highest quality, longest training time
- **`high_quality`**: Good quality, moderate training time
- **`medium_quality`**: Balanced quality and speed
- **`optimize_for_deployment`**: Fast inference, smaller models

## Output and Results

### Workflow Results
```json
{
  "workflow_status": "SUCCESS",
  "timestamp": "2025-12-18T00:00:00",
  "data_info": {
    "source_table": "ML_DATA.PUBLIC.CUSTOMER_FEATURES",
    "total_rows": 100000,
    "train_rows": 80000,
    "test_rows": 20000
  },
  "feature_selection": {
    "total_numeric_features": 50,
    "selected_features": 35,
    "dropped_features": 15,
    "correlation_threshold": 0.95
  },
  "model_info": {
    "model_name": "customer_churn_classifier",
    "model_version": "v1",
    "best_model": "WeightedEnsemble_L2"
  },
  "performance_metrics": {
    "f1_score": 0.8542,
    "recall": 0.8312,
    "precision": 0.8785,
    "accuracy": 0.8923
  },
  "all_models_evaluated": 12,
  "service_endpoint": "https://churn_prediction_service.snowflakecomputing.app/predict"
}
```

### Feature Selection Report
The workflow generates a detailed report showing:
- Original number of features
- Number of features selected/dropped
- Pairs of highly correlated features
- Correlation values for dropped features

## Best Practices

### 1. Data Preparation
- **Clean your data**: Remove nulls, outliers, and invalid values
- **Encode categoricals**: Use label encoding or one-hot encoding before workflow
- **Scale if needed**: Though AutoGluon handles this internally
- **Balanced classes**: Consider SMOTE or class weights for imbalanced data

### 2. Feature Engineering
- **Domain knowledge**: Start with domain-relevant features
- **Feature interactions**: Create interaction terms if needed
- **Temporal features**: Extract date/time components (day, month, hour, etc.)

### 3. Model Training
- **Time limits**: Start with 1 hour (3600s) for experimentation, increase for production
- **Compute resources**: Use appropriate warehouse size (Large or X-Large recommended)
- **Validation strategy**: Default 5-fold CV is good, adjust based on data size

### 4. Monitoring
- **Track metrics**: Monitor F1, Recall, Precision over time
- **Feature drift**: Check if feature distributions change
- **Model decay**: Retrain periodically when performance drops

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `time_limit` parameter
   - Use smaller `presets` (e.g., "medium_quality")
   - Increase warehouse size

2. **No Features Selected**
   - Lower `correlation_threshold` (e.g., 0.85 or 0.80)
   - Check if features are properly encoded as numeric

3. **Training Too Slow**
   - Reduce `time_limit`
   - Use "medium_quality" or "high_quality" presets
   - Reduce `num_bag_folds` or `num_stack_levels`

4. **Poor Model Performance**
   - Increase training time
   - Add more features or feature engineering
   - Check for data leakage
   - Verify class balance

## Additional Resources

- [Snowflake Feature Store Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml/feature-store/overview)
- [Snowflake Model Registry](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)
- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [Snowflake Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)

## License

MIT License - Feel free to use and modify for your projects.

## Author

Randal Scott King
- Website: www.randalscottking.com
- GitHub: @randalscottking
