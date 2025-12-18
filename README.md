# Snowflake ML Workflow with AutoGluon

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Snowflake](https://img.shields.io/badge/Snowflake-Ready-29B5E8)](https://www.snowflake.com/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-0.8+-orange)](https://auto.gluon.ai/)

End-to-end automated machine learning pipeline for Snowflake with intelligent feature selection, AutoGluon model training, and production deployment to Snowflake Feature Store, Model Registry, and Container Services.

Snowflake doesn't have a built-in AutoML process like Databricks, but I've found that Google's AutoGluon library works just as well and provides slightly better results. 

## Overview

This production-ready template provides a complete MLOps workflow for classification tasks in Snowflake:

- **Automated Feature Engineering**: Pearson correlation-based feature selection removes redundant features
- **Feature Store Integration**: Version-controlled feature management with Snowflake Feature Store
- **AutoML Training**: Leverages AutoGluon to automatically train and optimize multiple models
- **Intelligent Model Selection**: Chooses best model based on F1 Score, Recall, and Precision
- **Dual Model Registry**: Registers models to both Snowflake Warehouse and Container Services
- **Production Deployment**: One-click deployment to Snowflake Container Services with REST API

## Features

### Core Capabilities

**Smart Feature Selection**
- Automatic removal of highly correlated features (configurable threshold)
- Preserves predictive power while reducing dimensionality
- Detailed correlation reports

**AutoML Training**
- Trains multiple model types: LightGBM, XGBoost, CatBoost, Random Forest, Neural Networks
- Automatic hyperparameter optimization
- Ensemble methods (bagging and stacking)

**Model Evaluation**
- Comprehensive metrics: F1 Score, Recall, Precision, Accuracy, ROC-AUC
- Multi-model comparison
- Automated best model selection

**MLOps Integration**
- Snowflake Feature Store for feature versioning
- Model Registry for model lineage and versioning
- Container Services for scalable inference

## Quick Start

### Prerequisites

- Snowflake account with appropriate permissions
- Python 3.9 or higher
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/randalscottking/snowflake-autogluon-classification.git
cd snowflake-autogluon-classification

# Install dependencies
pip install -r requirements.txt
```

### Snowflake Setup

```bash
# Run the setup script in Snowflake
# This creates databases, schemas, warehouses, and sample data
snowsql -f sql/snowflake_setup.sql
```

### Basic Usage

```python
from src.snowflake_ml_workflow import SnowflakeMLWorkflow
from snowflake.snowpark import Session

# Create Snowflake session
session = Session.builder.configs(your_connection_params).create()

# Initialize workflow
workflow = SnowflakeMLWorkflow(
    session=session,
    source_table="ML_DATA.PUBLIC.CUSTOMER_FEATURES",
    target_column="CHURN_FLAG"
)

# Run complete workflow
results = workflow.run_complete_workflow(
    entity_column="CUSTOMER_ID",
    feature_view_name="customer_churn_features",
    model_name="customer_churn_classifier",
    time_limit=3600  # 1 hour training
)

print(f"Best Model: {results['model_info']['best_model']}")
print(f"F1 Score: {results['performance_metrics']['f1_score']:.4f}")
```

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- [Complete Documentation](docs/WORKFLOW_GUIDE.md) - Full workflow guide
- [Configuration Reference](docs/CONFIGURATION.md) - All configuration options
- [API Reference](docs/API.md) - Detailed API documentation
- [Examples](examples/) - Usage examples and tutorials

## Project Structure

```
snowflake-autogluon-classification/
├── src/                              # Source code
│   └── snowflake_ml_workflow.py     # Main workflow class
├── examples/                         # Usage examples
│   └── example_usage.py             # Example implementation
├── config/                           # Configuration files
│   └── workflow_config.yaml         # Default configuration
├── sql/                              # SQL scripts
│   └── snowflake_setup.sql          # Environment setup
├── docs/                             # Documentation
│   └── WORKFLOW_GUIDE.md            # Complete guide
├── tests/                            # Unit tests
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## Configuration

Customize the workflow using `config/workflow_config.yaml`:

```yaml
# Data Configuration
data:
  source_table: "ML_DATA.PUBLIC.CUSTOMER_FEATURES"
  target_column: "CHURN_FLAG"
  entity_column: "CUSTOMER_ID"

# Feature Engineering
features:
  correlation_threshold: 0.95
  
# Training Configuration
training:
  test_size: 0.2
  time_limit: 3600
  presets: "best_quality"
```

See [Configuration Reference](docs/CONFIGURATION.md) for all options.

## Workflow Steps

1. **Data Loading** - Load data from Snowflake table
2. **Feature Selection** - Calculate correlations and remove redundant features
3. **Feature Store** - Register features to Snowflake Feature Store
4. **Model Training** - Train multiple models with AutoGluon
5. **Model Evaluation** - Evaluate and select best model
6. **Model Registry** - Register model with metadata
7. **Deployment** - (Optional) Deploy to Container Services

## Model Selection

Models are automatically selected based on:

1. **F1 Score** (Primary) - Balance of precision and recall
2. **Recall** (Secondary) - Minimize false negatives
3. **Precision** (Tertiary) - Minimize false positives

## Example Output

```json
{
  "workflow_status": "SUCCESS",
  "model_info": {
    "model_name": "customer_churn_classifier",
    "best_model": "WeightedEnsemble_L2"
  },
  "performance_metrics": {
    "f1_score": 0.8542,
    "recall": 0.8312,
    "precision": 0.8785,
    "accuracy": 0.8923
  },
  "feature_selection": {
    "selected_features": 35,
    "dropped_features": 15
  }
}
```

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Randal Scott King**
- Website: [randalscottking.com](https://www.randalscottking.com)
- GitHub: [@randalscottking](https://github.com/randalscottking)
- Location: Atlanta, GA

## Acknowledgments

- [Snowflake](https://www.snowflake.com/) for ML platform capabilities
- [AutoGluon](https://auto.gluon.ai/) for AutoML framework
- Snowflake ML community for inspiration and feedback

## Additional Resources

- [Snowflake Feature Store Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml/feature-store/overview)
- [Snowflake Model Registry](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)
- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [Snowflake Container Services](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/overview)

## Issues & Support

For bugs, questions, or feature requests, please [open an issue](https://github.com/randalscottking/snowflake-autogluon-classification/issues).

---

If you find this project useful, please consider giving it a star!
