# 🚀 Complete FREE Deployment Guide
# Local Training + Render.com Deployment

## Overview

This guide shows you how to:
1. Train Gradient Boosting model locally (FREE)
2. Test API locally
3. Deploy to Render.com (FREE - 750 hrs/month)

**Total Cost: $0/month** ✅

---

## 📁 Required Files

Make sure you have these files in your project folder:

```
RA_flare_prediction_DEPLOYMENT/
├── ra_data_simple.csv                      # Training data
├── train_gradient_boosting_local.py        # Training script
├── app.py                                  # FastAPI server
├── requirements.txt                        # Dependencies
├── test_local_api.py                       # Local testing
└── README.md                               # This file
```

---

## Part 1: Local Training (FREE)

### **Step 1: Generate Training Data**

First, make sure you have the training data:

```bash
# If you don't have ra_data_simple.csv, generate it:
python step1_generate_research_dataset.py
python step2_simple_features.py
```

Expected output:
```
✅ Loaded 2000 samples
✅ Saved: ra_data_simple.csv
```

---

### **Step 2: Train Gradient Boosting Model**

```bash
python train_gradient_boosting_local.py
```

**Expected Output:**
```
======================================================================
LOCAL TRAINING - GRADIENT BOOSTING MODEL
======================================================================

📂 Loading training data...
✅ Loaded 2000 samples

📊 Features (12 raw features):
    1. age
    2. gender
    3. disease_duration
    4. bmi
    5. min_temperature
    6. max_temperature
    7. humidity
    8. barometric_pressure
    9. precipitation
   10. wind_speed
   11. tender_joint_count
   12. swollen_joint_count
   Target: inflammation (0=Remission, 1=Inflammation)

✅ Data split:
   Training: 1600 samples (54.0% inflammation)
   Testing: 400 samples (54.0% inflammation)

⚙️ Scaling features...
✅ Features scaled

======================================================================
🚀 TRAINING GRADIENT BOOSTING MODEL
======================================================================

⚙️ Hyperparameters:
   n_estimators: 100
   learning_rate: 0.1
   max_depth: 5
   min_samples_split: 5
   min_samples_leaf: 2
   subsample: 0.8
   max_features: sqrt
   random_state: 42

🔨 Training model...
✅ Training complete!

======================================================================
📊 MODEL EVALUATION
======================================================================

📈 Performance Metrics:
   Accuracy:  0.8525 (85.25%)
   Precision: 0.8456
   Recall:    0.8434
   F1 Score:  0.8445
   ROC AUC:   0.9178

📋 Classification Report:
              precision    recall  f1-score   support

   Remission       0.86      0.87      0.87       184
Inflammation       0.85      0.84      0.84       216

    accuracy                           0.85       400
   macro avg       0.85      0.85      0.85       400
weighted avg       0.85      0.85      0.85       400

Confusion Matrix:
                Predicted
              Remission  Inflammation
Actual Remission       160        24
       Inflammation     35       181

🔍 Top 5 Most Important Features:
   tender_joint_count       : 0.2145
   swollen_joint_count      : 0.1823
   disease_duration         : 0.1534
   age                      : 0.1234
   humidity                 : 0.0987

======================================================================
💾 SAVING MODEL
======================================================================

💾 Saving model artifacts...
✅ Saved: ra_model_gradient_boosting.pkl
✅ Saved: ra_scaler_gradient_boosting.pkl
✅ Saved: ra_features_gradient_boosting.pkl
✅ Saved: ra_model_type.pkl

======================================================================
✅ TRAINING COMPLETE
======================================================================

📋 Summary:
   Algorithm: Gradient Boosting
   Accuracy: 85.25%
   AUC: 0.9178
   Features: 12

📦 Files created:
   - ra_model_gradient_boosting.pkl
   - ra_scaler_gradient_boosting.pkl
   - ra_features_gradient_boosting.pkl
   - ra_model_type.pkl

💰 Cost: $0 (completely FREE!)
```

**Training Time:** ~2-5 minutes (depending on your PC)

**Files Created:**
- ✅ `ra_model_gradient_boosting.pkl` (trained model)
- ✅ `ra_scaler_gradient_boosting.pkl` (feature scaler)
- ✅ `ra_features_gradient_boosting.pkl` (feature names)
- ✅ `ra_model_type.pkl` (model type identifier)

---

## Part 2: Test Locally (FREE)

### **Step 3: Start Local Server**

```bash
python -m uvicorn app:app --reload
```

**Expected Output:**
```
🔧 Loading model...
✅ Model loaded: gradient_boosting
📊 Features: 12
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Server is now running at:** `http://localhost:8000`

---

### **Step 4: Test API (In New Terminal)**

```bash
python test_local_api.py
```

**Expected Output:**
```
======================================================================
TESTING LOCAL API
======================================================================

🧪 Test 1: Root Endpoint
GET http://localhost:8000/
✅ Status: 200
{
  "message": "RA Inflammation Predictor API",
  "version": "1.0.0",
  "model": "gradient_boosting",
  "status": "healthy",
  "endpoints": {
    "predict": "POST /predict",
    "health": "GET /health",
    "info": "GET /info"
  },
  "docs": "/docs"
}

🧪 Test 2: Health Check
GET http://localhost:8000/health
✅ Status: 200
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "gradient_boosting",
  "timestamp": "2025-11-13T12:45:00"
}

🧪 Test 3: Model Info
GET http://localhost:8000/info
✅ Status: 200
   Model: GradientBoostingClassifier
   Algorithm: gradient_boosting
   Features: 12

🧪 Test 4: Low Risk Patient
POST http://localhost:8000/predict
✅ Status: 200
   Probability: 0.2341 (23.41%)
   Prediction: Remission
   Risk Level: LOW
   Confidence: 0.7659

🧪 Test 5: High Risk Patient
POST http://localhost:8000/predict
✅ Status: 200
   Probability: 0.8567 (85.67%)
   Prediction: Inflammation
   Risk Level: HIGH
   Confidence: 0.8567

======================================================================
✅ LOCAL API TESTING COMPLETE
======================================================================

📋 All tests passed!
🚀 Ready to deploy to Render.com
```

### **Interactive Docs:**

Visit: `http://localhost:8000/docs`

You'll see interactive API documentation where you can test endpoints directly!

---

## Part 3: Deploy to Render.com (FREE)

### **Step 5: Create GitHub Repository**

```bash
# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit - RA Inflammation Predictor"

# Create repo on GitHub and push
git remote add origin https://github.com/your-username/ra-inflammation-api.git
git branch -M main
git push -u origin main
```

**Important:** Make sure these files are in your repo:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `ra_model_gradient_boosting.pkl`
- ✅ `ra_scaler_gradient_boosting.pkl`
- ✅ `ra_features_gradient_boosting.pkl`
- ✅ `ra_model_type.pkl`

---

### **Step 6: Deploy on Render.com**

#### **A. Sign Up (FREE)**
1. Go to [render.com](https://render.com)
2. Sign up with GitHub (FREE account)

#### **B. Create New Web Service**
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Select your repo: `ra-inflammation-api`

#### **C. Configure Service**
```
Name: ra-inflammation-api
Region: Oregon (US West)
Branch: main
Root Directory: (leave empty)
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
Instance Type: Free
```

#### **D. Deploy**
Click **"Create Web Service"**

**Deployment time:** ~3-5 minutes

---

### **Step 7: Test Deployed API**

Once deployed, you'll get a URL like:
```
https://ra-inflammation-api.onrender.com
```

**Test it:**

```bash
# Health check
curl https://ra-inflammation-api.onrender.com/health

# Make prediction
curl -X POST https://ra-inflammation-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "gender": 1,
    "disease_duration": 10,
    "bmi": 26.5,
    "min_temperature": 15,
    "max_temperature": 20,
    "humidity": 65,
    "barometric_pressure": 1015,
    "precipitation": 0.5,
    "wind_speed": 5,
    "tender_joint_count": 2,
    "swollen_joint_count": 1.5
  }'
```

**Expected Response:**
```json
{
  "inflammation_probability": 0.4523,
  "inflammation_prediction": 0,
  "risk_level": "MODERATE",
  "confidence": 0.5477,
  "prediction_date": "2025-11-13T17:45:00",
  "model_type": "gradient_boosting"
}
```

---

## Part 4: Use in Your Dashboard

### **JavaScript/React Example:**

```javascript
const API_URL = "https://ra-inflammation-api.onrender.com";

async function getPrediction(patientData) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(patientData)
  });
  
  const result = await response.json();
  return result;
}

// Usage
const patient = {
  age: 55,
  gender: 1,
  disease_duration: 10,
  bmi: 26.5,
  min_temperature: 15,
  max_temperature: 20,
  humidity: 65,
  barometric_pressure: 1015,
  precipitation: 0.5,
  wind_speed: 5,
  tender_joint_count: 2,
  swollen_joint_count: 1.5
};

const prediction = await getPrediction(patient);
console.log(`Risk: ${prediction.risk_level} (${(prediction.inflammation_probability * 100).toFixed(1)}%)`);
```

### **Python Example:**

```python
import requests

API_URL = "https://ra-inflammation-api.onrender.com"

patient_data = {
    "age": 55,
    "gender": 1,
    "disease_duration": 10,
    "bmi": 26.5,
    "min_temperature": 15,
    "max_temperature": 20,
    "humidity": 65,
    "barometric_pressure": 1015,
    "precipitation": 0.5,
    "wind_speed": 5,
    "tender_joint_count": 2,
    "swollen_joint_count": 1.5
}

response = requests.post(f"{API_URL}/predict", json=patient_data)
result = response.json()

print(f"Risk: {result['risk_level']} ({result['inflammation_probability']*100:.1f}%)")
```

---

## 💰 Cost Breakdown

### **FREE Components:**

| Component | Provider | Cost |
|-----------|----------|------|
| Model Training | Your PC | $0 |
| API Hosting | Render.com | $0 (750 hrs/month) |
| Database | Supabase | $0 (500 MB) |
| Frontend | Vercel | $0 (unlimited) |
| GitHub | GitHub | $0 |
| **TOTAL** | | **$0/month** ✅ |

### **Render.com Free Tier:**
- ✅ 750 hours/month (enough for 24/7 with 1 service)
- ✅ Automatic HTTPS
- ✅ Auto-scaling
- ✅ Spins down after 15 min inactivity
- ✅ Spins up on request (~30 seconds)

---

## 🔧 Troubleshooting

### **Issue: Model not loading**
**Solution:** Make sure all `.pkl` files are committed to GitHub

### **Issue: Build failed on Render**
**Solution:** Check build logs. Usually missing `requirements.txt`

### **Issue: Service spinning down**
**Solution:** Normal on free tier. First request takes ~30s (cold start)

### **Issue: Out of memory**
**Solution:** Reduce model size or upgrade to paid tier ($7/month)

---

## 📊 Performance

### **Model:**
- Accuracy: ~85%
- AUC: ~0.92
- Inference time: <100ms
- Cold start: ~30s (first request)
- Warm requests: <100ms

### **Render Free Tier:**
- CPU: 0.1 vCPU
- RAM: 512 MB
- Bandwidth: 100 GB/month
- Uptime: 750 hours/month

---

## ✅ Summary

### **What You Have:**

1. ✅ **Local Training** - Gradient Boosting model trained on your PC
2. ✅ **FastAPI Server** - Production-ready API
3. ✅ **FREE Deployment** - Hosted on Render.com
4. ✅ **Interactive Docs** - Automatic Swagger UI
5. ✅ **HTTPS Enabled** - Secure by default
6. ✅ **Auto-scaling** - Handles traffic automatically

### **Total Cost:** $0/month 🎉

### **Next Steps:**
1. Deploy frontend on Vercel
2. Connect to Supabase database
3. Add monitoring (Sentry - free tier)
4. Set up CI/CD (GitHub Actions - free)

**Your RA inflammation predictor is now live and FREE!** 🚀