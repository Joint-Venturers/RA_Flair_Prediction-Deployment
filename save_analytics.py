# save_analytics_rest.py
# Save analytics using REST API 

import os
import json
import requests
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def save_to_supabase(url: str, key: str, metadata: dict):
    """Save training metrics using Supabase REST API"""
    
    # Prepare data
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
    
    # Headers
    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    try:
        # Insert training history
        response = requests.post(
            f'{url}/rest/v1/model_training_history',
            headers=headers,
            json=data
        )
        
        if response.status_code in [200, 201]:
            logging.info("✅ Training metrics saved to Supabase")
            logging.info(f"   Model: {data['model_type']}, Accuracy: {data['accuracy']:.4f}")
            
            # Update daily summary
            update_daily_summary(url, key, data)
            return True
        else:
            logging.error(f"❌ Failed to save metrics: {response.status_code}")
            logging.error(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def update_daily_summary(url: str, key: str, training_data: dict):
    """Update daily analytics summary"""
    headers = {
        'apikey': key,
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'Prefer': 'return=representation'
    }
    
    today = datetime.now().date().isoformat()
    
    try:
        # Check if exists
        response = requests.get(
            f'{url}/rest/v1/analytics_summary?date=eq.{today}',
            headers=headers
        )
        
        summary_data = {
            'model_accuracy': training_data['accuracy'],
            'updated_at': datetime.now().isoformat()
        }
        
        if response.status_code == 200 and response.json():
            # Update existing
            requests.patch(
                f'{url}/rest/v1/analytics_summary?date=eq.{today}',
                headers=headers,
                json=summary_data
            )
            logging.info(f"✅ Updated daily summary for {today}")
        else:
            # Insert new
            summary_data.update({
                'date': today,
                'total_predictions': 0,
                'high_risk_count': 0,
                'moderate_risk_count': 0,
                'low_risk_count': 0
            })
            requests.post(
                f'{url}/rest/v1/analytics_summary',
                headers=headers,
                json=summary_data
            )
            logging.info(f"✅ Created daily summary for {today}")
            
    except Exception as e:
        logging.error(f"❌ Failed to update summary: {e}")

if __name__ == "__main__":
    # Load environment variables
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY')
    
    if not url or not key:
        logging.error("❌ SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
        exit(1)
    
    # Load metadata
    try:
        with open('training_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        logging.info(f"📊 Loading metadata from training_metadata.json")
        logging.info(f"   Accuracy: {metadata.get('accuracy', 'N/A')}")
        logging.info(f"   Samples: {metadata.get('total_samples', 'N/A')}")
        
        success = save_to_supabase(url, key, metadata)
        
        if success:
            logging.info("✅ Analytics saved successfully")
        else:
            logging.warning("⚠️ Analytics not saved")
            
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        import traceback
        logging.error(traceback.format_exc())
