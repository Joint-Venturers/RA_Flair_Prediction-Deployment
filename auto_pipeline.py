# Automated ML Pipeline
# Monitors triggers, trains model, validates, and deploys

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import logging
from typing import Dict, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_pipeline.log'),
        logging.StreamHandler()
    ]
)

class AutoMLPipeline:
    """Automated ML Pipeline with configurable thresholds"""
    
    def __init__(self, config_file='pipeline_config.json'):
        self.config = self.load_config(config_file)
        self.features = [
            'age', 'gender', 'disease_duration', 'bmi',
            'min_temperature', 'max_temperature', 'humidity',
            'barometric_pressure', 'precipitation', 'wind_speed',
            'tender_joint_count', 'swollen_joint_count'
        ]
        self.target = 'inflammation'
        logging.info("AutoML Pipeline initialized")
    
    def load_config(self, config_file: str) -> Dict:
        """Load or create configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "triggers": {
                    "min_new_samples": 100,
                    "min_days_since_last_train": 7,
                    "accuracy_threshold": 0.80,
                    "min_improvement": 0.01
                },
                "validation": {
                    "test_size": 0.2,
                    "min_accuracy": 0.80,
                    "min_auc": 0.85
                },
                "github": {
                    "branch": "dev"
                }
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logging.info(f"Created default config: {config_file}")
            return default_config
    
    def check_triggers(self) -> Tuple[bool, str]:
        """Check if retraining should be triggered"""
        triggers = self.config['triggers']
        
        # Check 1: New data threshold
        new_samples = self.get_new_sample_count()
        if new_samples >= triggers['min_new_samples']:
            return True, f"New data: {new_samples} samples (>= {triggers['min_new_samples']})"
        
        # Check 2: Time-based
        days_since = self.days_since_last_training()
        if days_since >= triggers['min_days_since_last_train']:
            return True, f"Time-based: {days_since} days (>= {triggers['min_days_since_last_train']})"
        
        # Check 3: Performance degradation
        current_perf = self.get_current_performance()
        if current_perf and current_perf < triggers['accuracy_threshold']:
            return True, f"Performance: {current_perf:.4f} (< {triggers['accuracy_threshold']})"
        
        return False, "No triggers activated"
    
    def get_new_sample_count(self) -> int:
        """Count new samples since last training"""
        try:
            if not os.path.exists('training_metadata.json'):
                return 999  # Force train if never trained
            
            with open('training_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            last_count = metadata.get('total_samples', 0)
            df = pd.read_csv('ra_data_simple.csv')
            current_count = len(df)
            
            return max(0, current_count - last_count)
        except Exception as e:
            logging.warning(f"Could not count new samples: {e}")
            return 0
    
    def days_since_last_training(self) -> int:
        """Calculate days since last training"""
        try:
            if not os.path.exists('training_metadata.json'):
                return 999
            
            with open('training_metadata.json', 'r') as f:
                metadata = json.load(f)
            
            last_train = datetime.fromisoformat(metadata['timestamp'])
            days = (datetime.now() - last_train).days
            return days
        except:
            return 999
    
    def get_current_performance(self) -> Optional[float]:
        """Get current model accuracy"""
        try:
            with open('training_metadata.json', 'r') as f:
                metadata = json.load(f)
            return metadata.get('accuracy')
        except:
            return None
    
    def train_model(self, df: pd.DataFrame) -> Tuple:
        """Train Gradient Boosting model"""
        logging.info("Training model...")
        
        X = df[self.features]
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['validation']['test_size'],
            random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            min_samples_split=5, min_samples_leaf=2,
            subsample=0.8, max_features='sqrt', random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'auc': float(roc_auc_score(y_test, y_prob)),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logging.info(f"Training complete: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        return model, scaler, metrics
    
    def validate_model(self, metrics: Dict) -> Tuple[bool, str]:
        """Validate model meets requirements"""
        validation = self.config['validation']
        
        if metrics['accuracy'] < validation['min_accuracy']:
            return False, f"Accuracy too low: {metrics['accuracy']:.4f}"
        
        if metrics['auc'] < validation['min_auc']:
            return False, f"AUC too low: {metrics['auc']:.4f}"
        
        return True, "Validation passed"
    
    def should_deploy(self, new_metrics: Dict, old_metrics: Optional[Dict]) -> Tuple[bool, str]:
        """Check if new model should be deployed"""
        if not old_metrics:
            return True, "First deployment"
        
        improvement = new_metrics['accuracy'] - old_metrics['accuracy']
        min_improvement = self.config['triggers']['min_improvement']
        
        if improvement >= min_improvement:
            return True, f"Improved by {improvement:.4f}"
        elif improvement > 0:
            return True, f"Minor improvement ({improvement:.4f})"
        else:
            return False, f"No improvement ({improvement:.4f})"
    
    def save_model(self, model, scaler, metrics: Dict):
        """Save model artifacts"""
        logging.info("Saving model...")
        
        joblib.dump(model, 'ra_model_gradient_boosting.pkl')
        joblib.dump(scaler, 'ra_scaler_gradient_boosting.pkl')
        joblib.dump(self.features, 'ra_features_gradient_boosting.pkl')
        joblib.dump('gradient_boosting', 'ra_model_type.pkl')
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'training_samples': metrics['training_samples'],
            'test_samples': metrics['test_samples'],
            'total_samples': metrics['training_samples'] + metrics['test_samples'],
            'features': self.features,
            'branch': self.config['github']['branch']
        }
        
        with open('training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info("Model saved")
    
    def run(self, force: bool = False):
        """Main pipeline execution"""
        logging.info("=" * 70)
        logging.info("AUTOMATED ML PIPELINE - STARTING")
        logging.info("=" * 70)
        
        # Check triggers
        if not force:
            should_run, reason = self.check_triggers()
            if not should_run:
                logging.info(f"No retraining needed: {reason}")
                return
            logging.info(f"Trigger activated: {reason}")
        else:
            logging.info("Force retraining enabled")
        
        # Load data
        try:
            df = pd.read_csv('ra_data_simple.csv')
            logging.info(f"Loaded {len(df)} samples")
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            return
        
        # Train model
        try:
            model, scaler, new_metrics = self.train_model(df)
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return
        
        # Validate
        is_valid, msg = self.validate_model(new_metrics)
        if not is_valid:
            logging.error(f"Validation failed: {msg}")
            return
        
        logging.info(f"Validation passed: {msg}")
        
        # Check if should deploy
        old_metrics = None
        if os.path.exists('training_metadata.json'):
            with open('training_metadata.json', 'r') as f:
                old_data = json.load(f)
                old_metrics = {
                    'accuracy': old_data.get('accuracy'),
                    'f1': old_data.get('f1'),
                    'auc': old_data.get('auc')
                }
        
        should_deploy, deploy_reason = self.should_deploy(new_metrics, old_metrics)
        if not should_deploy:
            logging.info(f"Not deploying: {deploy_reason}")
            return
        
        logging.info(f"Deploying: {deploy_reason}")
        
        # Save model
        self.save_model(model, scaler, new_metrics)
        
        logging.info("✅ Pipeline complete - Model ready for deployment")
        logging.info("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated ML Pipeline')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--config', default='pipeline_config.json', help='Config file')
    
    args = parser.parse_args()
    
    pipeline = AutoMLPipeline(config_file=args.config)
    pipeline.run(force=args.force)


if __name__ == "__main__":
    main()
