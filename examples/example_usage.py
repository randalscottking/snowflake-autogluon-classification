from datetime import datetime
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
    """Execute the ML workflow with configuration and experiment tracking."""
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Create Snowflake session
    print("Creating Snowflake session...")
    session = create_snowflake_session(config)
    
    try:
        # Initialize workflow WITH experiment tracking
        print("\nInitializing ML workflow with experiment tracking...")
        workflow = SnowflakeMLWorkflow(
            session=session,
            source_table=config['data']['source_table'],
            target_column=config['data']['target_column'],
            feature_store_db=config['features']['feature_store']['database'],
            model_registry_db=config['model_registry']['database'],
            correlation_threshold=config['features']['correlation_threshold'],
            experiment_name="customer_churn_experiments"  # NEW: Experiment name
        )
        
        # Execute workflow WITH experiment run tracking
        print("\nExecuting complete ML workflow with experiment logging...")
        results = workflow.run_complete_workflow(
            entity_column=config['data']['entity_column'],
            feature_view_name=config['features']['feature_store']['feature_view_name'],
            model_name=config['model_registry']['model_name'],
            test_size=config['training']['test_size'],
            time_limit=config['training']['autogluon']['time_limit'],
            deploy_to_container=config['container_services']['enabled'],
            compute_pool=config['container_services'].get('compute_pool'),
            service_name=config['container_services'].get('service_name'),
            experiment_run_name="run_" + datetime.now().strftime("%Y%m%d_%H%M%S")  # NEW: Unique run name
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
        
        if results['experiment_run']:
            print(f"\n📊 Experiment Run: {results['experiment_run']}")
            print(f"View in Snowsight: AI & ML » Experiments » customer_churn_experiments")
        
        if results['service_endpoint']:
            print(f"\n🚀 Service Endpoint: {results['service_endpoint']}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: Workflow failed with error: {e}")
        raise
        
    finally:
        # Close session
        session.close()
        print("\nSnowflake session closed.")


if __name__ == "__main__":
    results = main()
