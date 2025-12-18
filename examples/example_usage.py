"""
Example Usage Script for Snowflake ML Workflow
===============================================
This script demonstrates how to use the ML workflow template
with a configuration file.

Author: Randal Scott King
"""

import yaml
from snowflake.snowpark import Session
from src.snowflake_ml_workflow import SnowflakeMLWorkflow


def load_config(config_path: str = 'config/workflow_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_snowflake_session(config: dict) -> Session:
    """Create Snowflake session from config."""
    connection_params = {
        "account": config['snowflake']['account'],
        "user": config['snowflake']['user'],
        "role": config['snowflake']['role'],
        "warehouse": config['snowflake']['warehouse'],
        "database": config['snowflake']['database'],
        "schema": config['snowflake']['schema'],
    }
    
    # Add password or authenticator
    # Option 1: Password (not recommended for production)
    # connection_params['password'] = "your_password"
    
    # Option 2: SSO
    # connection_params['authenticator'] = 'externalbrowser'
    
    # Option 3: Key-pair authentication (recommended)
    # connection_params['private_key'] = your_private_key
    
    return Session.builder.configs(connection_params).create()


def main():
    """Execute the ML workflow with configuration."""
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Create Snowflake session
    print("Creating Snowflake session...")
    session = create_snowflake_session(config)
    
    try:
        # Initialize workflow
        print("\nInitializing ML workflow...")
        workflow = SnowflakeMLWorkflow(
            session=session,
            source_table=config['data']['source_table'],
            target_column=config['data']['target_column'],
            feature_store_db=config['features']['feature_store']['database'],
            model_registry_db=config['model_registry']['database'],
            correlation_threshold=config['features']['correlation_threshold']
        )
        
        # Execute workflow
        print("\nExecuting complete ML workflow...")
        results = workflow.run_complete_workflow(
            entity_column=config['data']['entity_column'],
            feature_view_name=config['features']['feature_store']['feature_view_name'],
            model_name=config['model_registry']['model_name'],
            test_size=config['training']['test_size'],
            time_limit=config['training']['autogluon']['time_limit'],
            deploy_to_container=config['container_services']['enabled'],
            compute_pool=config['container_services'].get('compute_pool'),
            service_name=config['container_services'].get('service_name')
        )
        
        # Print results
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nModel Name: {results['model_info']['model_name']}")
        print(f"Best Model: {results['model_info']['best_model']}")
        print(f"\nPerformance Metrics:")
        print(f"  F1 Score:  {results['performance_metrics']['f1_score']:.4f}")
        print(f"  Recall:    {results['performance_metrics']['recall']:.4f}")
        print(f"  Precision: {results['performance_metrics']['precision']:.4f}")
        print(f"  Accuracy:  {results['performance_metrics']['accuracy']:.4f}")
        print(f"\nFeatures Selected: {results['feature_selection']['selected_features']}")
        print(f"Features Dropped: {results['feature_selection']['dropped_features']}")
        print(f"Total Models Evaluated: {results['all_models_evaluated']}")
        
        if results['service_endpoint']:
            print(f"\nService Endpoint: {results['service_endpoint']}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: Workflow failed with error: {e}")
        raise
        
    finally:
        # Close session
        session.close()
        print("\nSnowflake session closed.")


def run_with_custom_parameters():
    """
    Example of running workflow with custom parameters
    (overriding config file settings).
    """
    config = load_config()
    session = create_snowflake_session(config)
    
    try:
        # Initialize with custom settings
        workflow = SnowflakeMLWorkflow(
            session=session,
            source_table="ML_DATA.PUBLIC.MY_CUSTOM_TABLE",
            target_column="MY_TARGET",
            correlation_threshold=0.90  # Lower threshold
        )
        
        # Run with custom training parameters
        results = workflow.run_complete_workflow(
            entity_column="USER_ID",
            feature_view_name="my_custom_features",
            model_name="my_custom_model",
            time_limit=7200,  # 2 hours instead of 1
            test_size=0.25    # 25% test split instead of 20%
        )
        
        return results
        
    finally:
        session.close()


def run_individual_steps():
    """
    Example of running individual workflow steps
    instead of the complete workflow.
    """
    config = load_config()
    session = create_snowflake_session(config)
    
    try:
        workflow = SnowflakeMLWorkflow(
            session=session,
            source_table=config['data']['source_table'],
            target_column=config['data']['target_column']
        )
        
        # Step 1: Load data
        df = workflow.load_data()
        print(f"Loaded {df.count()} rows")
        
        # Step 2: Feature selection
        selected_features, report = workflow.select_features_by_correlation(df)
        print(f"Selected {len(selected_features)} features")
        print(f"Dropped {report['dropped_features']} features")
        
        # Step 3: Create feature view
        feature_view = workflow.create_feature_view(
            df=df,
            selected_features=selected_features,
            entity_column=config['data']['entity_column'],
            feature_view_name="my_features_manual"
        )
        
        # Step 4: Load features and train
        training_data = workflow.load_features_for_training("my_features_manual")
        
        # Split data manually
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            training_data,
            test_size=0.2,
            random_state=42,
            stratify=training_data[config['data']['target_column']]
        )
        
        # Step 5: Train models
        predictor = workflow.train_autogluon_models(
            train_data=train_df,
            time_limit=1800,  # 30 minutes for quick test
            presets="medium_quality"  # Faster training
        )
        
        # Step 6: Evaluate
        evaluation = workflow.evaluate_and_select_best_model(
            predictor=predictor,
            test_data=test_df
        )
        
        # Step 7: Register model
        model_version = workflow.register_model_to_warehouse(
            predictor=predictor,
            model_name="manual_workflow_model",
            best_model_name=evaluation['best_model']['model_name'],
            metrics=evaluation,
            feature_list=selected_features
        )
        
        print(f"\nModel registered: {model_version}")
        
        return {
            'features': selected_features,
            'evaluation': evaluation,
            'model_version': model_version
        }
        
    finally:
        session.close()


def quick_test():
    """
    Quick test with minimal training time for development.
    """
    config = load_config()
    session = create_snowflake_session(config)
    
    try:
        workflow = SnowflakeMLWorkflow(
            session=session,
            source_table=config['data']['source_table'],
            target_column=config['data']['target_column']
        )
        
        # Quick test with minimal settings
        results = workflow.run_complete_workflow(
            entity_column=config['data']['entity_column'],
            feature_view_name="test_features",
            model_name="test_model",
            time_limit=300,  # Only 5 minutes
            test_size=0.3    # Larger test set for quick feedback
        )
        
        print("\nQuick test completed!")
        print(f"Best model: {results['model_info']['best_model']}")
        print(f"F1 Score: {results['performance_metrics']['f1_score']:.4f}")
        
        return results
        
    finally:
        session.close()


if __name__ == "__main__":
    # Choose which example to run:
    
    # Option 1: Run complete workflow with config file (recommended)
    results = main()
    
    # Option 2: Run with custom parameters
    # results = run_with_custom_parameters()
    
    # Option 3: Run individual steps
    # results = run_individual_steps()
    
    # Option 4: Quick test for development
    # results = quick_test()
