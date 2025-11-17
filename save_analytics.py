# save_analytics.py
# Save ML training metrics to Supabase

import os
import json
from datetime import datetime
from supabase import create_client, Client
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)

class AnalyticsSaver:
    """Save ML training analytics to Supabase"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            logging.warning("Supabase credentials not found. Analytics will not be saved.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                logging.info("Supabase client initialized")
            except Exception as e:
                logging.error(f"Failed to initialize Supabase: {e}")
                self.client = None
    
    def save_training_metrics(self, metadata: Dict) -> bool:
        """
        Save training metrics to Supabase
        
        Args:
            metadata: Dictionary containing training metadata
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logging.warning("Supabase client not initialized")
            return False
        
        try:
            data = {
                'model_type': metadata.get('model_type', 'gradient_boosting'),
                'accuracy': float(metadata.get('accuracy', 0)),
                'f1_score': float(metadata.get('f1', 0)),
                'auc': float(metadata.get('auc', 0)),
                'training_samples': int(metadata.get('training_samples', 0)),
                'test_samples': int(metadata.get('test_samples', 0)),
                'total_samples': int(metadata.get('total_samples', 0)),
                'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                'features': metadata.get('features', []),
                'status': 'success'
            }
            
            result = self.client.table('model_training_history').insert(data).execute()
            logging.info(f"Training metrics saved to Supabase: {result.data}")
            
            # Also update daily summary
            self._update_daily_summary(data)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save training metrics: {e}")
            return False
    
    def _update_daily_summary(self, training_data: Dict):
        """Update or insert daily analytics summary"""
        try:
            today = datetime.now().date().isoformat()
            
            # Check if entry exists
            result = self.client.table('analytics_summary')\
                .select('*')\
                .eq('date', today)\
                .execute()
            
            if result.data:
                # Update existing
                self.client.table('analytics_summary')\
                    .update({
                        'model_accuracy': training_data['accuracy'],
                        'updated_at': datetime.now().isoformat()
                    })\
                    .eq('date', today)\
                    .execute()
            else:
                # Insert new
                self.client.table('analytics_summary')\
                    .insert({
                        'date': today,
                        'model_accuracy': training_data['accuracy'],
                        'total_predictions': 0,
                        'high_risk_count': 0,
                        'moderate_risk_count': 0,
                        'low_risk_count': 0
                    })\
                    .execute()
                
            logging.info(f"Daily summary updated for {today}")
            
        except Exception as e:
            logging.error(f"Failed to update daily summary: {e}")


def save_training_analytics(metadata_file: str = 'training_metadata.json'):
    """
    Read training metadata and save to Supabase
    
    Args:
        metadata_file: Path to training metadata JSON file
    """
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        saver = AnalyticsSaver()
        success = saver.save_training_metrics(metadata)
        
        if success:
            logging.info("✅ Analytics saved successfully")
        else:
            logging.warning("⚠️ Analytics not saved")
            
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {metadata_file}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in metadata file: {metadata_file}")
    except Exception as e:
        logging.error(f"Error saving analytics: {e}")


if __name__ == "__main__":
    save_training_analytics()
