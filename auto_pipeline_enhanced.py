# auto_pipeline_enhanced.py (Windows-compatible version without emojis)
# Enhanced automated ML retraining pipeline with personalized trigger features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import json
from datetime import datetime, timedelta
import os
import logging

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class EnhancedRAModelPipeline:
    """Enhanced automated ML pipeline for RA flare prediction with personalized triggers"""
    
    def __init__(self, data_file='ra_data_enhanced.csv', config_file='pipeline_config.json'):
        self.data_file = data_file
        self.config_file = config_file
        self.config = self.load_config()
        self.metadata_file = 'training_metadata.json'
        
        # Enhanced feature list (16 features)
        self.feature_columns = [
            'age',
            'sex',
            'disease_duration',
            'bmi',
            'sleep_hours',
            'smoking_status',
            'air_quality_index',
            'min_temperature',
            'max_temperature',
            'humidity',
            'barometric_pressure',
            'precipitation',
            'wind_speed',
            'current_pain_score',
            'tender_joint_count',
            'swollen_joint_count'
        ]
        
        logging.info("=" * 60)
        logging.info("[RA] Enhanced RA Flare Prediction Pipeline")
        logging.info("=" * 60)
        logging.info("[DATA] Features: {}".format(len(self.feature_columns)))
        logging.info("[FILE] Data file: {}".format(self.data_file))
    
    def load_config(self):
        """Load pipeline configuration"""
        default_config = {
            "min_samples_for_training": 100,
            "min_new_samples": 50,
            "min_accuracy": 0.70,
            "min_auc": 0.75,
            "min_improvement": 0.00,  # Set to 0 for testing
            "days_between_retraining": 7,
            "test_size": 0.2,
            "random_state": 42
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logging.info("[OK] Loaded config from {}".format(self.config_file))
                return {**default_config, **config}
            except Exception as e:
                logging.warning("[WARN] Error loading config: {}. Using defaults.".format(e))
                return default_config
        else:
            logging.info("[INFO] Using default configuration")
            return default_config
    
    def load_data(self):
        """Load enhanced training data"""
        try:
            if not os.path.exists(self.data_file):
                logging.error("[ERROR] Data file not found: {}".format(self.data_file))
                return None
            
            df = pd.read_csv(self.data_file)
            logging.info("[OK] Loaded {} samples from {}".format(len(df), self.data_file))
            
            # Validate required columns
            required_cols = self.feature_columns + ['inflammation']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                logging.error("[ERROR] Missing columns: {}".format(missing_cols))
                return None
            
            logging.info("[DATA] Flare rate: {:.1f}%".format(df['inflammation'].mean()*100))
            return df
            
        except Exception as e:
            logging.error("[ERROR] Error loading data: {}".format(e))
            return None
    
    def load_previous_metadata(self):
        """Load metadata from previous training"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                logging.info("[INFO] Previous model: accuracy={:.4f}".format(
                    metadata.get('accuracy', 0)
                ))
                return metadata
            else:
                logging.info("[INFO] No previous training metadata found")
                return None
        except Exception as e:
            logging.warning("[WARN] Error loading metadata: {}".format(e))
            return None
    
    def check_retraining_triggers(self, df, prev_metadata):
        """
        Check if retraining should be triggered
        
        Triggers:
        1. New data volume (>= min_new_samples)
        2. Time-based (>= days_between_retraining)
        3. Model performance degradation
        4. Force flag
        """
        triggers = []
        should_retrain = False
        
        # Check force flag
        force = os.getenv('FORCE_RETRAIN', 'false').lower() == 'true'
        if force:
            triggers.append("Force retrain flag set")
            should_retrain = True
        
        # Check data volume
        current_samples = len(df)
        if prev_metadata:
            prev_samples = prev_metadata.get('total_samples', 0)
            new_samples = current_samples - prev_samples
            
            if new_samples >= self.config['min_new_samples']:
                triggers.append("New data: {} samples (>= {})".format(
                    new_samples, self.config['min_new_samples']
                ))
                should_retrain = True
        else:
            triggers.append("Initial training (no previous model)")
            should_retrain = True
        
        # Check time-based trigger
        if prev_metadata and 'timestamp' in prev_metadata:
            last_train = datetime.fromisoformat(prev_metadata['timestamp'])
            days_since = (datetime.now() - last_train).days
            
            if days_since >= self.config['days_between_retraining']:
                triggers.append("Time-based: {} days since last training".format(days_since))
                should_retrain = True
        
        # Check performance degradation
        if prev_metadata:
            prev_accuracy = prev_metadata.get('accuracy', 0)
            if prev_accuracy < self.config['min_accuracy']:
                triggers.append("Performance degradation: accuracy {:.4f} < {}".format(
                    prev_accuracy, self.config['min_accuracy']
                ))
                should_retrain = True
        
        logging.info("\n[CHECK] Retraining Check:")
        logging.info("  Should retrain: {}".format(should_retrain))
        if triggers:
            logging.info("  Triggers:")
            for trigger in triggers:
                logging.info("    - {}".format(trigger))
        
        return should_retrain, triggers
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train enhanced Gradient Boosting model"""
        logging.info("\n[TRAIN] Training Enhanced Model...")
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config['random_state']
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        logging.info("\n[DATA] Model Performance:")
        logging.info("  Accuracy: {:.4f}".format(accuracy))
        logging.info("  F1 Score: {:.4f}".format(f1))
        logging.info("  AUC:      {:.4f}".format(auc))
        
        # Feature importance
        feature_importance = sorted(
            zip(self.feature_columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        
        logging.info("\n[TOP] Top 5 Feature Importances:")
        for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            logging.info("  {}. {:25s} {:.4f}".format(i, feature, importance))
        
        return model, accuracy, f1, auc
    
    def validate_model(self, accuracy, auc, prev_metadata):
        """
        Validate if model meets quality standards
        
        Checks:
        1. Minimum accuracy threshold
        2. Minimum AUC threshold
        3. Improvement over previous model
        """
        reasons = []
        
        # Check minimum thresholds
        if accuracy < self.config['min_accuracy']:
            reasons.append("Accuracy {:.4f} < minimum {}".format(
                accuracy, self.config['min_accuracy']
            ))
        
        if auc < self.config['min_auc']:
            reasons.append("AUC {:.4f} < minimum {}".format(
                auc, self.config['min_auc']
            ))
        
        # Check improvement
        if prev_metadata:
            prev_accuracy = prev_metadata.get('accuracy', 0)
            improvement = accuracy - prev_accuracy
            
            if improvement < self.config['min_improvement']:
                reasons.append("Improvement {:.4f} < minimum {}".format(
                    improvement, self.config['min_improvement']
                ))
        
        is_valid = len(reasons) == 0
        
        logging.info("\n[OK] Model Validation:")
        logging.info("  Valid: {}".format(is_valid))
        if not is_valid:
            logging.warning("  Reasons:")
            for reason in reasons:
                logging.warning("    - {}".format(reason))
        
        return is_valid
    
    def save_model(self, model, scaler, metadata):
        """Save model artifacts"""
        try:
            # Save model
            joblib.dump(model, 'ra_model_gradient_boosting.pkl')
            logging.info("[SAVE] Saved model: ra_model_gradient_boosting.pkl")
            
            # Save scaler
            joblib.dump(scaler, 'ra_scaler_gradient_boosting.pkl')
            logging.info("[SAVE] Saved scaler: ra_scaler_gradient_boosting.pkl")
            
            # Save features
            joblib.dump(self.feature_columns, 'ra_features_gradient_boosting.pkl')
            logging.info("[SAVE] Saved features: ra_features_gradient_boosting.pkl")
            
            # Save model type
            joblib.dump('gradient_boosting', 'ra_model_type.pkl')
            logging.info("[SAVE] Saved model type: ra_model_type.pkl")
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.info("[SAVE] Saved metadata: {}".format(self.metadata_file))
            
            return True
            
        except Exception as e:
            logging.error("[ERROR] Error saving model: {}".format(e))
            return False
    
    def run(self):
        """Execute the enhanced pipeline"""
        logging.info("\n" + "=" * 60)
        logging.info("[START] Pipeline Started: {}".format(datetime.now().isoformat()))
        logging.info("=" * 60)
        
        # Load data
        df = self.load_data()
        if df is None:
            logging.error("[ERROR] Pipeline failed: Cannot load data")
            return False
        
        # Load previous metadata
        prev_metadata = self.load_previous_metadata()
        
        # Check retraining triggers
        should_retrain, triggers = self.check_retraining_triggers(df, prev_metadata)
        
        if not should_retrain:
            logging.info("\n[OK] No retraining needed")
            return True
        
        # Prepare data
        X = df[self.feature_columns]
        y = df['inflammation']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        logging.info("\n[DATA] Data Split:")
        logging.info("  Training samples: {}".format(len(X_train)))
        logging.info("  Test samples:     {}".format(len(X_test)))
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model, accuracy, f1, auc = self.train_model(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Validate model
        is_valid = self.validate_model(accuracy, auc, prev_metadata)
        
        if not is_valid:
            logging.warning("\n[WARN] Model did not pass validation - skipping deployment")
            return False
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'gradient_boosting',
            'accuracy': float(accuracy),
            'f1': float(f1),
            'auc': float(auc),
            'training_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'total_samples': int(len(df)),
            'features': self.feature_columns,
            'n_features': len(self.feature_columns),
            'triggers': triggers,
            'branch': 'main'
        }
        
        # Save model
        success = self.save_model(model, scaler, metadata)
        
        if success:
            logging.info("\n" + "=" * 60)
            logging.info("[OK] PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("=" * 60)
            logging.info("  Model Type: Enhanced Gradient Boosting")
            logging.info("  Features: {}".format(len(self.feature_columns)))
            logging.info("  Accuracy: {:.4f}".format(accuracy))
            logging.info("  F1 Score: {:.4f}".format(f1))
            logging.info("  AUC: {:.4f}".format(auc))
            logging.info("=" * 60)
            return True
        else:
            logging.error("\n[ERROR] Pipeline failed: Error saving model")
            return False

if __name__ == "__main__":
    pipeline = EnhancedRAModelPipeline()
    success = pipeline.run()
    exit(0 if success else 1)
