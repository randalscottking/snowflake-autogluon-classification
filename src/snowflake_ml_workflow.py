"""
Snowflake ML Workflow Template
===============================
End-to-end ML pipeline for classification tasks with:
- Feature selection using Pearson correlation
- Feature Store integration
- AutoGluon AutoML training
- Model evaluation and selection
- Model Registry (Warehouse + Container Services)

Author: Randal Scott King
"""

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from snowflake.ml.feature_store import (
    FeatureStore,
    FeatureView,
    Entity,
    CreationMode
)
from snowflake.ml.registry import Registry
from snowflake.ml.modeling.preprocessing import StandardScaler
from snowflake.snowpark import Session
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime


class SnowflakeMLWorkflow:
    """
    Comprehensive ML workflow for Snowflake with feature engineering,
    AutoGluon training, and model registry integration.
    """
    
    def __init__(
        self,
        session: Session,
        source_table: str,
        target_column: str,
        feature_store_db: str = "ML_FEATURES",
        model_registry_db: str = "ML_MODELS",
        correlation_threshold: float = 0.95
    ):
        """
        Initialize the ML workflow.
        
        Args:
            session: Snowflake Snowpark session
            source_table: Fully qualified table name (DB.SCHEMA.TABLE)
            target_column: Name of the target variable column
            feature_store_db: Database for feature store
            model_registry_db: Database for model registry
            correlation_threshold: Threshold for dropping highly correlated features
        """
        self.session = session
        self.source_table = source_table
        self.target_column = target_column
        self.feature_store_db = feature_store_db
        self.model_registry_db = model_registry_db
        self.correlation_threshold = correlation_threshold
        
        # Initialize Feature Store and Registry
        self.feature_store = FeatureStore(
            session=session,
            database=feature_store_db,
            name="ML_FEATURE_STORE",
            default_warehouse=session.get_current_warehouse(),
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST
        )
        
        self.registry = Registry(
            session=session,
            database_name=model_registry_db,
            schema_name="MODEL_REGISTRY"
        )
        
    def load_data(self) -> snowpark.DataFrame:
        """Load data from source table."""
        print(f"Loading data from {self.source_table}...")
        df = self.session.table(self.source_table)
        print(f"Loaded {df.count()} rows")
        return df
    
    def select_features_by_correlation(
        self,
        df: snowpark.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[List[str], Dict]:
        """
        Select features by removing highly correlated columns.
        
        Uses Pearson correlation to identify and remove redundant features.
        Keeps the first feature in each highly correlated pair.
        
        Args:
            df: Snowpark DataFrame
            exclude_columns: Columns to exclude from analysis (e.g., IDs, target)
            
        Returns:
            Tuple of (selected_features, correlation_report)
        """
        print("Calculating feature correlations...")
        
        # Get numeric columns only
        pdf = df.to_pandas()
        numeric_cols = pdf.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude specified columns
        if exclude_columns is None:
            exclude_columns = [self.target_column]
        else:
            exclude_columns = list(set(exclude_columns + [self.target_column]))
        
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        # Calculate correlation matrix
        corr_matrix = pdf[feature_cols].corr().abs()
        
        # Find features to drop
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)
        
        to_drop = []
        dropped_pairs = []
        
        for column in upper_corr.columns:
            correlated = upper_corr[column][upper_corr[column] > self.correlation_threshold]
            if not correlated.empty:
                for idx, corr_value in correlated.items():
                    if column not in to_drop:
                        to_drop.append(column)
                        dropped_pairs.append({
                            'kept_feature': idx,
                            'dropped_feature': column,
                            'correlation': float(corr_value)
                        })
        
        selected_features = [col for col in feature_cols if col not in to_drop]
        
        report = {
            'total_numeric_features': len(feature_cols),
            'selected_features': len(selected_features),
            'dropped_features': len(to_drop),
            'correlation_threshold': self.correlation_threshold,
            'dropped_pairs': dropped_pairs,
            'selected_feature_list': selected_features
        }
        
        print(f"Feature selection complete:")
        print(f"  - Original features: {len(feature_cols)}")
        print(f"  - Selected features: {len(selected_features)}")
        print(f"  - Dropped features: {len(to_drop)}")
        
        return selected_features, report
    
    def create_feature_view(
        self,
        df: snowpark.DataFrame,
        selected_features: List[str],
        entity_column: str,
        feature_view_name: str
    ) -> FeatureView:
        """
        Create and register a Feature View in the Feature Store.
        
        Args:
            df: Snowpark DataFrame with features
            selected_features: List of feature column names
            entity_column: Column to use as entity (e.g., customer_id)
            feature_view_name: Name for the feature view
            
        Returns:
            FeatureView object
        """
        print(f"Creating feature view '{feature_view_name}'...")
        
        # Define entity
        entity = Entity(name=entity_column, join_keys=[entity_column])
        
        # Select features and entity
        feature_cols = selected_features + [entity_column, self.target_column]
        feature_df = df.select(feature_cols)
        
        # Create feature view
        feature_view = FeatureView(
            name=feature_view_name,
            entities=[entity],
            feature_df=feature_df,
            refresh_freq="1 day",
            desc=f"Curated features for ML model training - Created {datetime.now()}"
        )
        
        # Register in feature store
        self.feature_store.register_feature_view(
            feature_view=feature_view,
            version="1.0",
            block=True
        )
        
        print(f"Feature view '{feature_view_name}' registered successfully")
        return feature_view
    
    def load_features_for_training(
        self,
        feature_view_name: str,
        version: str = "1.0"
    ) -> pd.DataFrame:
        """
        Load features from Feature Store for model training.
        
        Args:
            feature_view_name: Name of the feature view
            version: Version of the feature view
            
        Returns:
            Pandas DataFrame with features
        """
        print(f"Loading features from Feature Store: {feature_view_name} v{version}")
        
        # Retrieve feature view
        feature_view = self.feature_store.get_feature_view(
            name=feature_view_name,
            version=version
        )
        
        # Load as pandas DataFrame
        spine_df = self.session.sql(f"""
            SELECT * FROM {self.feature_store_db}.{feature_view_name}
        """)
        
        pdf = spine_df.to_pandas()
        print(f"Loaded {len(pdf)} rows with {len(pdf.columns)} columns")
        
        return pdf
    
    def train_autogluon_models(
        self,
        train_data: pd.DataFrame,
        time_limit: int = 3600,
        presets: str = "best_quality",
        num_bag_folds: int = 5,
        num_stack_levels: int = 1
    ) -> TabularPredictor:
        """
        Train multiple classification models using AutoGluon.
        
        Args:
            train_data: Training data with target column
            time_limit: Time limit in seconds for training
            presets: AutoGluon preset ('best_quality', 'high_quality', 'medium_quality')
            num_bag_folds: Number of folds for bagging
            num_stack_levels: Number of stacking levels
            
        Returns:
            Trained TabularPredictor
        """
        print("Starting AutoGluon training...")
        print(f"  - Time limit: {time_limit}s")
        print(f"  - Presets: {presets}")
        print(f"  - Bag folds: {num_bag_folds}")
        print(f"  - Stack levels: {num_stack_levels}")
        
        # Configure AutoGluon
        predictor = TabularPredictor(
            label=self.target_column,
            problem_type='binary',  # Change to 'multiclass' if needed
            eval_metric='f1',
            path='./autogluon_models'
        )
        
        # Train models
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            presets=presets,
            num_bag_folds=num_bag_folds,
            num_stack_levels=num_stack_levels,
            verbosity=2
        )
        
        print("AutoGluon training complete!")
        return predictor
    
    def evaluate_and_select_best_model(
        self,
        predictor: TabularPredictor,
        test_data: pd.DataFrame
    ) -> Dict:
        """
        Evaluate all trained models and select the best one.
        
        Selection criteria (in priority order):
        1. F1 Score
        2. Recall
        3. Precision
        
        Args:
            predictor: Trained AutoGluon predictor
            test_data: Test data for evaluation
            
        Returns:
            Dictionary with best model info and all model metrics
        """
        print("Evaluating models...")
        
        # Get leaderboard with detailed metrics
        leaderboard = predictor.leaderboard(data=test_data, silent=True)
        
        # Get detailed metrics for each model
        model_metrics = []
        
        for model_name in leaderboard['model']:
            try:
                # Get predictions
                y_true = test_data[self.target_column]
                y_pred = predictor.predict(test_data, model=model_name)
                
                # Calculate metrics
                from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
                
                f1 = f1_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                precision = precision_score(y_true, y_pred, average='weighted')
                accuracy = accuracy_score(y_true, y_pred)
                
                model_metrics.append({
                    'model_name': model_name,
                    'f1_score': f1,
                    'recall': recall,
                    'precision': precision,
                    'accuracy': accuracy,
                    'score_val': leaderboard[leaderboard['model'] == model_name]['score_val'].values[0]
                })
            except Exception as e:
                print(f"Warning: Could not evaluate {model_name}: {e}")
        
        # Convert to DataFrame for easier analysis
        metrics_df = pd.DataFrame(model_metrics)
        
        # Select best model based on F1, then Recall, then Precision
        metrics_df = metrics_df.sort_values(
            by=['f1_score', 'recall', 'precision'],
            ascending=False
        )
        
        best_model = metrics_df.iloc[0].to_dict()
        
        print(f"\nBest Model Selected: {best_model['model_name']}")
        print(f"  - F1 Score: {best_model['f1_score']:.4f}")
        print(f"  - Recall: {best_model['recall']:.4f}")
        print(f"  - Precision: {best_model['precision']:.4f}")
        print(f"  - Accuracy: {best_model['accuracy']:.4f}")
        
        return {
            'best_model': best_model,
            'all_models': metrics_df.to_dict('records'),
            'leaderboard': leaderboard
        }
    
    def register_model_to_warehouse(
        self,
        predictor: TabularPredictor,
        model_name: str,
        best_model_name: str,
        metrics: Dict,
        feature_list: List[str]
    ) -> str:
        """
        Register the best model to Snowflake Model Registry (Warehouse).
        
        Args:
            predictor: Trained AutoGluon predictor
            model_name: Name for the registered model
            best_model_name: Name of the best performing model
            metrics: Model evaluation metrics
            feature_list: List of features used
            
        Returns:
            Model version string
        """
        print(f"Registering model '{model_name}' to Snowflake Model Registry...")
        
        # Prepare model metadata
        metadata = {
            'model_type': 'AutoGluon TabularPredictor',
            'best_model': best_model_name,
            'f1_score': metrics['best_model']['f1_score'],
            'recall': metrics['best_model']['recall'],
            'precision': metrics['best_model']['precision'],
            'accuracy': metrics['best_model']['accuracy'],
            'features': feature_list,
            'target_column': self.target_column,
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Register model
        model_version = self.registry.log_model(
            model_name=model_name,
            model_version='v1',
            model=predictor,
            conda_dependencies=['autogluon.tabular'],
            metadata=metadata,
            comment=f"Best model: {best_model_name} with F1={metrics['best_model']['f1_score']:.4f}"
        )
        
        print(f"Model registered successfully: {model_version}")
        return model_version
    
    def deploy_to_container_services(
        self,
        model_name: str,
        model_version: str,
        compute_pool: str,
        service_name: str
    ) -> str:
        """
        Deploy model to Snowflake Container Services.
        
        Args:
            model_name: Registered model name
            model_version: Model version
            compute_pool: Snowflake compute pool for the service
            service_name: Name for the deployed service
            
        Returns:
            Service endpoint URL
        """
        print(f"Deploying model to Container Services: {service_name}")
        
        # Create service specification
        service_spec = f"""
        CREATE SERVICE IF NOT EXISTS {service_name}
        IN COMPUTE POOL {compute_pool}
        FROM @ML_MODELS.MODEL_REGISTRY.{model_name}/{model_version}
        SPECIFICATION = $$
        spec:
          containers:
          - name: model-inference
            image: /ml_models/model_registry/autogluon:latest
            env:
              MODEL_NAME: {model_name}
              MODEL_VERSION: {model_version}
          endpoints:
          - name: predict
            port: 8080
            protocol: http
        $$
        """
        
        # Execute service creation
        self.session.sql(service_spec).collect()
        
        # Get service endpoint
        service_info = self.session.sql(f"""
            SHOW SERVICES LIKE '{service_name}'
        """).collect()
        
        endpoint = f"https://{service_name}.snowflakecomputing.app/predict"
        
        print(f"Service deployed successfully!")
        print(f"Endpoint: {endpoint}")
        
        return endpoint
    
    def run_complete_workflow(
        self,
        entity_column: str,
        feature_view_name: str,
        model_name: str,
        test_size: float = 0.2,
        time_limit: int = 3600,
        deploy_to_container: bool = False,
        compute_pool: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> Dict:
        """
        Execute the complete ML workflow end-to-end.
        
        Args:
            entity_column: Column to use as entity identifier
            feature_view_name: Name for the feature view
            model_name: Name for the registered model
            test_size: Proportion of data for testing
            time_limit: Time limit for AutoGluon training
            deploy_to_container: Whether to deploy to Container Services
            compute_pool: Compute pool for container deployment
            service_name: Service name for container deployment
            
        Returns:
            Dictionary with workflow results
        """
        print("="*80)
        print("STARTING COMPLETE ML WORKFLOW")
        print("="*80)
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Feature selection
        selected_features, correlation_report = self.select_features_by_correlation(df)
        
        # Step 3: Create feature view
        feature_view = self.create_feature_view(
            df=df,
            selected_features=selected_features,
            entity_column=entity_column,
            feature_view_name=feature_view_name
        )
        
        # Step 4: Load features for training
        training_data = self.load_features_for_training(feature_view_name)
        
        # Step 5: Split data
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            training_data,
            test_size=test_size,
            random_state=42,
            stratify=training_data[self.target_column]
        )
        
        print(f"Data split: {len(train_df)} train, {len(test_df)} test")
        
        # Step 6: Train AutoGluon models
        predictor = self.train_autogluon_models(
            train_data=train_df,
            time_limit=time_limit
        )
        
        # Step 7: Evaluate and select best model
        evaluation_results = self.evaluate_and_select_best_model(
            predictor=predictor,
            test_data=test_df
        )
        
        # Step 8: Register to Model Registry
        model_version = self.register_model_to_warehouse(
            predictor=predictor,
            model_name=model_name,
            best_model_name=evaluation_results['best_model']['model_name'],
            metrics=evaluation_results,
            feature_list=selected_features
        )
        
        # Step 9: Optional - Deploy to Container Services
        service_endpoint = None
        if deploy_to_container and compute_pool and service_name:
            service_endpoint = self.deploy_to_container_services(
                model_name=model_name,
                model_version=model_version,
                compute_pool=compute_pool,
                service_name=service_name
            )
        
        # Compile results
        results = {
            'workflow_status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'source_table': self.source_table,
                'total_rows': len(training_data),
                'train_rows': len(train_df),
                'test_rows': len(test_df)
            },
            'feature_selection': correlation_report,
            'feature_view': feature_view_name,
            'model_info': {
                'model_name': model_name,
                'model_version': model_version,
                'best_model': evaluation_results['best_model']['model_name']
            },
            'performance_metrics': evaluation_results['best_model'],
            'all_models_evaluated': len(evaluation_results['all_models']),
            'service_endpoint': service_endpoint
        }
        
        print("="*80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(json.dumps(results, indent=2, default=str))
        
        return results
