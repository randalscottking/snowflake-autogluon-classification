-- ============================================================================
-- Snowflake ML Workflow - Environment Setup
-- ============================================================================
-- This script sets up all necessary Snowflake objects for the ML workflow
-- Author: Randal Scott King
-- ============================================================================

-- ============================================================================
-- SECTION 1: Database and Schema Setup
-- ============================================================================

-- Create databases
CREATE DATABASE IF NOT EXISTS ML_DATA
    COMMENT = 'Database for source data and raw features';

CREATE DATABASE IF NOT EXISTS ML_FEATURES
    COMMENT = 'Database for Feature Store and curated features';

CREATE DATABASE IF NOT EXISTS ML_MODELS
    COMMENT = 'Database for Model Registry and model artifacts';

-- Create schemas
USE DATABASE ML_DATA;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS STAGING;

USE DATABASE ML_FEATURES;
CREATE SCHEMA IF NOT EXISTS FEATURE_STORE;

USE DATABASE ML_MODELS;
CREATE SCHEMA IF NOT EXISTS MODEL_REGISTRY;
CREATE SCHEMA IF NOT EXISTS MODEL_ARTIFACTS;

-- ============================================================================
-- SECTION 2: Warehouse Setup
-- ============================================================================

-- Create ML warehouse
CREATE WAREHOUSE IF NOT EXISTS ML_WAREHOUSE
    WAREHOUSE_SIZE = 'LARGE'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for ML training and inference';

-- Create inference warehouse (smaller, for production predictions)
CREATE WAREHOUSE IF NOT EXISTS ML_INFERENCE_WAREHOUSE
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for model inference';

-- ============================================================================
-- SECTION 3: Compute Pool for Container Services
-- ============================================================================

-- Create compute pool for Container Services
CREATE COMPUTE POOL IF NOT EXISTS ML_COMPUTE_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = CPU_X64_M
    AUTO_RESUME = TRUE
    AUTO_SUSPEND_SECS = 300
    COMMENT = 'Compute pool for ML model deployment';

-- ============================================================================
-- SECTION 4: Role and Privilege Setup
-- ============================================================================

-- Create ML role
CREATE ROLE IF NOT EXISTS ML_ENGINEER;
CREATE ROLE IF NOT EXISTS ML_ANALYST;

-- Grant database privileges to ML_ENGINEER
GRANT USAGE ON DATABASE ML_DATA TO ROLE ML_ENGINEER;
GRANT USAGE ON DATABASE ML_FEATURES TO ROLE ML_ENGINEER;
GRANT USAGE ON DATABASE ML_MODELS TO ROLE ML_ENGINEER;

GRANT ALL ON SCHEMA ML_DATA.PUBLIC TO ROLE ML_ENGINEER;
GRANT ALL ON SCHEMA ML_FEATURES.FEATURE_STORE TO ROLE ML_ENGINEER;
GRANT ALL ON SCHEMA ML_MODELS.MODEL_REGISTRY TO ROLE ML_ENGINEER;

-- Grant warehouse privileges
GRANT USAGE ON WAREHOUSE ML_WAREHOUSE TO ROLE ML_ENGINEER;
GRANT USAGE ON WAREHOUSE ML_INFERENCE_WAREHOUSE TO ROLE ML_ENGINEER;

-- Grant compute pool privileges
GRANT USAGE ON COMPUTE POOL ML_COMPUTE_POOL TO ROLE ML_ENGINEER;
GRANT MONITOR ON COMPUTE POOL ML_COMPUTE_POOL TO ROLE ML_ENGINEER;

-- Grant to ML_ANALYST (read-only)
GRANT USAGE ON DATABASE ML_DATA TO ROLE ML_ANALYST;
GRANT USAGE ON DATABASE ML_FEATURES TO ROLE ML_ANALYST;
GRANT USAGE ON DATABASE ML_MODELS TO ROLE ML_ANALYST;
GRANT SELECT ON ALL TABLES IN DATABASE ML_DATA TO ROLE ML_ANALYST;
GRANT SELECT ON ALL TABLES IN DATABASE ML_FEATURES TO ROLE ML_ANALYST;
GRANT SELECT ON ALL TABLES IN DATABASE ML_MODELS TO ROLE ML_ANALYST;

-- ============================================================================
-- SECTION 5: Example Source Data Table
-- ============================================================================

USE DATABASE ML_DATA;
USE SCHEMA PUBLIC;

-- Create example customer features table
CREATE OR REPLACE TABLE CUSTOMER_FEATURES (
    CUSTOMER_ID NUMBER(38,0) PRIMARY KEY,
    
    -- Demographic features
    AGE NUMBER(3,0),
    GENDER VARCHAR(10),
    LOCATION VARCHAR(50),
    INCOME_BRACKET VARCHAR(20),
    EDUCATION_LEVEL VARCHAR(20),
    
    -- Behavioral features
    ACCOUNT_AGE_DAYS NUMBER(10,0),
    TOTAL_PURCHASES NUMBER(10,0),
    AVG_PURCHASE_VALUE NUMBER(10,2),
    LAST_PURCHASE_DAYS_AGO NUMBER(10,0),
    PURCHASE_FREQUENCY NUMBER(10,2),
    
    -- Engagement features
    LOGIN_COUNT_30D NUMBER(10,0),
    SUPPORT_TICKETS_30D NUMBER(10,0),
    EMAIL_OPEN_RATE NUMBER(5,4),
    CLICK_THROUGH_RATE NUMBER(5,4),
    TIME_ON_PLATFORM_HOURS NUMBER(10,2),
    
    -- Financial features
    CURRENT_BALANCE NUMBER(12,2),
    CREDIT_SCORE NUMBER(4,0),
    PAYMENT_HISTORY_SCORE NUMBER(5,2),
    OUTSTANDING_BALANCE NUMBER(12,2),
    
    -- Calculated features
    CUSTOMER_LIFETIME_VALUE NUMBER(12,2),
    ENGAGEMENT_SCORE NUMBER(5,2),
    RISK_SCORE NUMBER(5,2),
    
    -- Target variable
    CHURN_FLAG NUMBER(1,0) COMMENT '1 = Churned, 0 = Active',
    
    -- Metadata
    CREATED_DATE TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_DATE TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
COMMENT = 'Customer features for churn prediction model';

-- Create index on customer ID
CREATE INDEX IF NOT EXISTS IDX_CUSTOMER_ID 
    ON CUSTOMER_FEATURES(CUSTOMER_ID);

-- ============================================================================
-- SECTION 6: Sample Data Generation (Optional)
-- ============================================================================

-- Generate sample data for testing
INSERT INTO CUSTOMER_FEATURES
SELECT
    SEQ8() AS CUSTOMER_ID,
    UNIFORM(18, 80, RANDOM()) AS AGE,
    CASE WHEN UNIFORM(0, 1, RANDOM()) < 0.5 THEN 'Male' ELSE 'Female' END AS GENDER,
    ARRAY_SLICE(ARRAY_CONSTRUCT('New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'), 
                UNIFORM(0, 4, RANDOM()), 1)[0]::VARCHAR AS LOCATION,
    ARRAY_SLICE(ARRAY_CONSTRUCT('<25K', '25-50K', '50-75K', '75-100K', '>100K'), 
                UNIFORM(0, 4, RANDOM()), 1)[0]::VARCHAR AS INCOME_BRACKET,
    ARRAY_SLICE(ARRAY_CONSTRUCT('High School', 'Some College', 'Bachelor', 'Master', 'PhD'), 
                UNIFORM(0, 4, RANDOM()), 1)[0]::VARCHAR AS EDUCATION_LEVEL,
    UNIFORM(30, 1800, RANDOM()) AS ACCOUNT_AGE_DAYS,
    UNIFORM(0, 200, RANDOM()) AS TOTAL_PURCHASES,
    UNIFORM(10, 500, RANDOM()) AS AVG_PURCHASE_VALUE,
    UNIFORM(0, 180, RANDOM()) AS LAST_PURCHASE_DAYS_AGO,
    UNIFORM(0, 20, RANDOM()) * 0.1 AS PURCHASE_FREQUENCY,
    UNIFORM(0, 100, RANDOM()) AS LOGIN_COUNT_30D,
    UNIFORM(0, 10, RANDOM()) AS SUPPORT_TICKETS_30D,
    UNIFORM(0, 100, RANDOM()) * 0.01 AS EMAIL_OPEN_RATE,
    UNIFORM(0, 100, RANDOM()) * 0.01 AS CLICK_THROUGH_RATE,
    UNIFORM(0, 500, RANDOM()) * 0.1 AS TIME_ON_PLATFORM_HOURS,
    UNIFORM(0, 10000, RANDOM()) AS CURRENT_BALANCE,
    UNIFORM(300, 850, RANDOM()) AS CREDIT_SCORE,
    UNIFORM(50, 100, RANDOM()) AS PAYMENT_HISTORY_SCORE,
    UNIFORM(0, 5000, RANDOM()) AS OUTSTANDING_BALANCE,
    UNIFORM(100, 10000, RANDOM()) AS CUSTOMER_LIFETIME_VALUE,
    UNIFORM(0, 100, RANDOM()) AS ENGAGEMENT_SCORE,
    UNIFORM(0, 100, RANDOM()) AS RISK_SCORE,
    CASE WHEN UNIFORM(0, 1, RANDOM()) < 0.2 THEN 1 ELSE 0 END AS CHURN_FLAG,
    CURRENT_TIMESTAMP() AS CREATED_DATE,
    CURRENT_TIMESTAMP() AS UPDATED_DATE
FROM TABLE(GENERATOR(ROWCOUNT => 10000));

-- Verify data
SELECT COUNT(*) AS TOTAL_ROWS,
       SUM(CHURN_FLAG) AS CHURNED_CUSTOMERS,
       AVG(CHURN_FLAG) * 100 AS CHURN_RATE_PCT
FROM CUSTOMER_FEATURES;

-- ============================================================================
-- SECTION 7: Create Stored Procedure
-- ============================================================================

-- Note: The complete stored procedure code should be pasted here
-- from the snowflake_ml_workflow.py file

CREATE OR REPLACE PROCEDURE ML_MODELS.MODEL_REGISTRY.RUN_ML_WORKFLOW(
    SOURCE_TABLE VARCHAR,
    TARGET_COLUMN VARCHAR,
    ENTITY_COLUMN VARCHAR,
    FEATURE_VIEW_NAME VARCHAR,
    MODEL_NAME VARCHAR,
    TEST_SIZE FLOAT DEFAULT 0.2,
    TIME_LIMIT INTEGER DEFAULT 3600,
    CORRELATION_THRESHOLD FLOAT DEFAULT 0.95
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
PACKAGES = (
    'snowflake-snowpark-python==1.11.1',
    'snowflake-ml-python==1.1.2',
    'autogluon.tabular==0.8.2',
    'pandas',
    'numpy',
    'scikit-learn'
)
HANDLER = 'main'
COMMENT = 'End-to-end ML workflow with AutoGluon and Feature Store'
AS
$$
# Paste the complete content of src/snowflake_ml_workflow.py here
# (The SnowflakeMLWorkflow class and main function)
$$;

-- ============================================================================
-- SECTION 8: Create Monitoring Views
-- ============================================================================

USE DATABASE ML_MODELS;
USE SCHEMA MODEL_REGISTRY;

-- View for model performance tracking
CREATE OR REPLACE VIEW MODEL_PERFORMANCE_HISTORY AS
SELECT
    MODEL_NAME,
    MODEL_VERSION,
    CREATED_TIMESTAMP,
    METADATA:performance_metrics:f1_score::FLOAT AS F1_SCORE,
    METADATA:performance_metrics:recall::FLOAT AS RECALL,
    METADATA:performance_metrics:precision::FLOAT AS PRECISION,
    METADATA:performance_metrics:accuracy::FLOAT AS ACCURACY,
    METADATA:best_model::STRING AS BEST_MODEL,
    METADATA:features::ARRAY AS FEATURES,
    METADATA:training_timestamp::TIMESTAMP_NTZ AS TRAINING_TIMESTAMP
FROM SYSTEM$REGISTRY.MODEL_VERSIONS
WHERE MODEL_NAME LIKE '%classifier%'
ORDER BY CREATED_TIMESTAMP DESC;

-- View for feature store lineage
USE DATABASE ML_FEATURES;
USE SCHEMA FEATURE_STORE;

CREATE OR REPLACE VIEW FEATURE_LINEAGE AS
SELECT
    FEATURE_VIEW_NAME,
    VERSION,
    ENTITY_NAME,
    FEATURE_COUNT,
    SOURCE_TABLE,
    CREATED_TIMESTAMP,
    REFRESH_FREQUENCY,
    LAST_REFRESH_TIME
FROM SYSTEM$FEATURE_STORE.FEATURE_VIEWS
ORDER BY CREATED_TIMESTAMP DESC;

-- ============================================================================
-- SECTION 9: Verification Queries
-- ============================================================================

-- Check database setup
SHOW DATABASES LIKE 'ML_%';

-- Check warehouse setup
SHOW WAREHOUSES LIKE 'ML_%';

-- Check compute pools
SHOW COMPUTE POOLS;

-- Check roles
SHOW GRANTS TO ROLE ML_ENGINEER;

-- Check sample data
USE DATABASE ML_DATA;
USE SCHEMA PUBLIC;

SELECT
    COUNT(*) AS TOTAL_ROWS,
    COUNT(DISTINCT CUSTOMER_ID) AS UNIQUE_CUSTOMERS,
    SUM(CHURN_FLAG) AS CHURNED_CUSTOMERS,
    ROUND(AVG(CHURN_FLAG) * 100, 2) AS CHURN_RATE_PCT,
    ROUND(AVG(AGE), 1) AS AVG_AGE,
    ROUND(AVG(CUSTOMER_LIFETIME_VALUE), 2) AS AVG_LIFETIME_VALUE
FROM CUSTOMER_FEATURES;

-- ============================================================================
-- SECTION 10: Example Workflow Execution
-- ============================================================================

-- Set context
USE ROLE ML_ENGINEER;
USE WAREHOUSE ML_WAREHOUSE;
USE DATABASE ML_MODELS;
USE SCHEMA MODEL_REGISTRY;

-- Execute ML workflow
CALL RUN_ML_WORKFLOW(
    'ML_DATA.PUBLIC.CUSTOMER_FEATURES',  -- source_table
    'CHURN_FLAG',                         -- target_column
    'CUSTOMER_ID',                        -- entity_column
    'customer_churn_features',            -- feature_view_name
    'customer_churn_classifier',          -- model_name
    0.2,                                  -- test_size
    3600,                                 -- time_limit (1 hour)
    0.95                                  -- correlation_threshold
);

-- Query results
SELECT * FROM ML_MODELS.MODEL_REGISTRY.MODEL_PERFORMANCE_HISTORY
WHERE MODEL_NAME = 'customer_churn_classifier'
ORDER BY CREATED_TIMESTAMP DESC
LIMIT 1;

-- ============================================================================
-- SECTION 11: Cleanup (Use with caution!)
-- ============================================================================

/*
-- Uncomment to drop all objects (WARNING: This will delete all data!)

USE ROLE ACCOUNTADMIN;

-- Drop compute pool
DROP COMPUTE POOL IF EXISTS ML_COMPUTE_POOL;

-- Drop warehouses
DROP WAREHOUSE IF EXISTS ML_WAREHOUSE;
DROP WAREHOUSE IF EXISTS ML_INFERENCE_WAREHOUSE;

-- Drop databases
DROP DATABASE IF EXISTS ML_DATA;
DROP DATABASE IF EXISTS ML_FEATURES;
DROP DATABASE IF EXISTS ML_MODELS;

-- Drop roles
DROP ROLE IF EXISTS ML_ENGINEER;
DROP ROLE IF EXISTS ML_ANALYST;
*/

-- ============================================================================
-- END OF SETUP SCRIPT
-- ============================================================================
