# client_test.py - Enhanced API Testing Client

import requests
import json
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

class RAAPIClient:
    """Enhanced RA API Testing Client"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            print(f"✅ API Status: {health_data['status']}")
            print(f"✅ Models Loaded: {health_data['models_loaded']}")
            print(f"✅ Version: {health_data['version']}")
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_basic_prediction(self):
        """Test basic prediction endpoint"""
        print("\n🔮 Testing basic prediction...")
        
        # Test data
        test_request = {
            "weather_data": {
                "temperature": 5,
                "humidity": 85,
                "pressure": 995,
                "weather_condition": "rainy",
                "temp_change_24h": -12,
                "pressure_change_24h": -20,
                "humidity_change_24h": 25
            },
            "pain_history": {
                "day_1_avg": 7.5,
                "day_3_avg": 6.8,
                "day_7_avg": 5.9
            },
            "patient_data": {
                "age": 55,
                "disease_duration": 8,
                "medication_adherence": 0.7,
                "sleep_quality": 3,
                "stress_level": 8
            },
            "include_analytics": True,
            "user_id": "test_user_123"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=test_request,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ Flare Probability: {result['flare_probability']:.1%}")
            print(f"✅ Risk Level: {result['risk_level']}")
            print(f"✅ Confidence: {result['confidence_score']:.1%}")
            print(f"✅ Risk Factors: {len(result['risk_factors'])}")
            print(f"✅ Recommendations: {len(result['recommendations'])}")
            print(f"✅ Models Used: {len(result['model_predictions'])}")
            
            # Display top risk factors
            if result['risk_factors']:
                print("\n🎯 Top Risk Factors:")
                for i, factor in enumerate(result['risk_factors'][:3], 1):
                    print(f"   {i}. {factor}")
            
            # Display top recommendations
            if result['recommendations']:
                print("\n💡 Top Recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"   {i}. {rec}")
            
            return True
            
        except Exception as e:
            print(f"❌ Basic prediction failed: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
            return False
    
    def test_multiple_scenarios(self):
        """Test multiple prediction scenarios"""
        print("\n🧪 Testing multiple scenarios...")
        
        scenarios = [
            {
                "name": "High Risk Weather",
                "data": {
                    "weather_data": {"temperature": 2, "humidity": 90, "pressure": 985, "weather_condition": "stormy", "temp_change_24h": -15, "pressure_change_24h": -25, "humidity_change_24h": 30},
                    "pain_history": {"day_1_avg": 8.0, "day_3_avg": 7.5, "day_7_avg": 6.5},
                    "patient_data": {"age": 65, "disease_duration": 15, "medication_adherence": 0.5, "sleep_quality": 2, "stress_level": 9}
                }
            },
            {
                "name": "Low Risk Conditions",
                "data": {
                    "weather_data": {"temperature": 22, "humidity": 45, "pressure": 1020, "weather_condition": "sunny", "temp_change_24h": 2, "pressure_change_24h": 5, "humidity_change_24h": -5},
                    "pain_history": {"day_1_avg": 3.0, "day_3_avg": 2.8, "day_7_avg": 3.2},
                    "patient_data": {"age": 45, "disease_duration": 5, "medication_adherence": 0.95, "sleep_quality": 8, "stress_level": 3}
                }
            },
            {
                "name": "Moderate Risk Mixed",
                "data": {
                    "weather_data": {"temperature": 12, "humidity": 65, "pressure": 1005, "weather_condition": "cloudy", "temp_change_24h": -5, "pressure_change_24h": -8, "humidity_change_24h": 12},
                    "pain_history": {"day_1_avg": 5.5, "day_3_avg": 5.0, "day_7_avg": 4.8},
                    "patient_data": {"age": 55, "disease_duration": 10, "medication_adherence": 0.8, "sleep_quality": 5, "stress_level": 6}
                }
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json={"include_analytics": False, "user_id": "test", **scenario["data"]},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                results.append({
                    "scenario": scenario["name"],
                    "probability": result["flare_probability"],
                    "risk_level": result["risk_level"],
                    "confidence": result["confidence_score"]
                })
                
                print(f"✅ {scenario['name']:20s}: {result['risk_level']:8s} ({result['flare_probability']:.1%})")
                
            except Exception as e:
                print(f"❌ {scenario['name']} failed: {e}")
        
        return results
    
    def test_analytics_endpoint(self):
        """Test analytics endpoint with historical data"""
        print("\n📊 Testing analytics endpoint...")
        
        # Generate sample historical data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        
        historical_data = []
        for i, date in enumerate(dates):
            # Simulate realistic patterns
            base_pain = 4 + np.sin(i / 5) * 2 + np.random.normal(0, 1)
            pain_level = max(1, min(10, base_pain))
            
            temp = 15 + 10 * np.sin(i / 10) + np.random.normal(0, 3)
            humidity = max(20, min(90, 60 + np.random.normal(0, 15)))
            pressure = 1013 + np.random.normal(0, 10)
            
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "weather_data": {
                    "temperature": temp,
                    "humidity": humidity,
                    "pressure": pressure,
                    "weather_condition": ["sunny", "cloudy", "rainy"][i % 3],
                    "temp_change_24h": np.random.normal(0, 5),
                    "pressure_change_24h": np.random.normal(0, 8),
                    "humidity_change_24h": np.random.normal(0, 10)
                },
                "pain_history": {
                    "day_1_avg": pain_level,
                    "day_3_avg": max(1, min(10, pain_level + np.random.normal(0, 0.5))),
                    "day_7_avg": max(1, min(10, pain_level + np.random.normal(0, 1)))
                },
                "patient_data": {
                    "age": 55,
                    "disease_duration": 8,
                    "medication_adherence": max(0.1, min(1.0, 0.8 + np.random.normal(0, 0.1))),
                    "sleep_quality": max(1, min(10, 6 + np.random.normal(0, 1.5))),
                    "stress_level": max(1, min(10, 5 + np.random.normal(0, 2)))
                },
                "flare_occurred": pain_level > 7  # Simple flare definition
            })
        
        analytics_request = {
            "user_id": "test_analytics_user",
            "historical_data": historical_data,
            "analytics_type": "comprehensive"
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analytics",
                json=analytics_request,
                headers={"Content-Type": "application/json"},
                timeout=30  # Analytics can take longer
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ Analytics generated for {result['data_period']['total_records']} records")
            print(f"✅ Period: {result['data_period']['start_date']} to {result['data_period']['end_date']}")
            
            # Display key insights
            analytics = result.get('analytics', {})
            if 'risk_analytics' in analytics:
                risk_data = analytics['risk_analytics']
                print(f"✅ Current Risk Score: {risk_data.get('current_risk_score', 'N/A')}")
                print(f"✅ Risk Level: {risk_data.get('risk_level', 'N/A')}")
                print(f"✅ Trend Direction: {risk_data.get('trend_direction', 'N/A')}")
            
            if 'summary_insights' in analytics:
                insights = analytics['summary_insights']
                print(f"✅ Generated {len(insights)} insights")
                for i, insight in enumerate(insights[:2], 1):
                    print(f"   {i}. {insight}")
            
            return True
            
        except Exception as e:
            print(f"❌ Analytics test failed: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
            return False
    
    def test_correlation_analysis(self):
        """Test correlation analysis endpoint"""
        print("\n🔗 Testing correlation analysis...")
        
        # Generate correlated sample data
        n_points = 60
        base_temp = 15 + 10 * np.sin(np.arange(n_points) / 10)
        base_humidity = 60 + 20 * np.sin(np.arange(n_points) / 8)
        
        historical_data = []
        for i in range(n_points):
            # Create realistic correlations
            temp = base_temp[i] + np.random.normal(0, 3)
            humidity = base_humidity[i] + np.random.normal(0, 10)
            
            # Pain correlated with cold weather and high humidity
            weather_pain_factor = (temp < 10) * 2 + (humidity > 70) * 1.5
            pain = max(1, min(10, 4 + weather_pain_factor + np.random.normal(0, 1)))
            
            historical_data.append({
                "date": (datetime.now() - timedelta(days=n_points-i)).strftime("%Y-%m-%d"),
                "weather_data": {
                    "temperature": temp,
                    "humidity": humidity,
                    "pressure": 1013 + np.random.normal(0, 15),
                    "weather_condition": "cloudy",
                    "temp_change_24h": 0,
                    "pressure_change_24h": 0,
                    "humidity_change_24h": 0
                },
                "pain_history": {
                    "day_1_avg": pain,
                    "day_3_avg": pain,
                    "day_7_avg": pain
                },
                "patient_data": {
                    "age": 55,
                    "disease_duration": 8,
                    "medication_adherence": 0.8,
                    "sleep_quality": max(1, min(10, 7 - pain * 0.3)),  # Poor sleep with high pain
                    "stress_level": max(1, min(10, 4 + pain * 0.4))    # Higher stress with high pain
                }
            })
        
        correlation_request = {
            "user_id": "correlation_test_user",
            "historical_data": historical_data
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/analytics/correlations",
                json=correlation_request,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ Correlation analysis completed")
            print(f"✅ Data period: {result['data_period']['date_range']}")
            
            correlation_analysis = result.get('correlation_analysis', {})
            
            # Display strong correlations
            strong_correlations = correlation_analysis.get('strong_correlations', [])
            if strong_correlations:
                print(f"✅ Found {len(strong_correlations)} strong correlations")
                for i, corr in enumerate(strong_correlations[:3], 1):
                    print(f"   {i}. {corr['symptom']} ↔ {corr['trigger']}: {corr['correlation_strength']:.3f}")
            
            # Display insights
            insights = correlation_analysis.get('personalized_insights', [])
            if insights:
                print(f"✅ Generated {len(insights)} correlation insights")
                for i, insight in enumerate(insights[:2], 1):
                    print(f"   {i}. {insight}")
            
            return True
            
        except Exception as e:
            print(f"❌ Correlation analysis failed: {e}")
            if hasattr(e, 'response'):
                print(f"Response: {e.response.text}")
            return False
    
    def test_model_info(self):
        """Test model information endpoint"""
        print("\n🤖 Testing model info endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/models/info")
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ Total models loaded: {result['total_models']}")
            print(f"✅ Scalers available: {result['scalers_loaded']}")
            print(f"✅ Feature engineer: {'✅' if result['feature_engineer_loaded'] else '❌'}")
            
            if result['model_details']:
                print("✅ Model details:")
                for name, details in result['model_details'].items():
                    print(f"   • {name}: {details['model_type']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model info test failed: {e}")
            return False
    
    def run_full_test_suite(self):
        """Run complete test suite"""
        print("🚀 Starting Enhanced RA API Test Suite")
        print("=" * 60)
        
        test_results = {
            'health_check': self.test_health_check(),
            'basic_prediction': self.test_basic_prediction(),
            'multiple_scenarios': len(self.test_multiple_scenarios()) > 0,
            'analytics': self.test_analytics_endpoint(),
            'correlations': self.test_correlation_analysis(),
            'model_info': self.test_model_info()
        }
        
        print("\n" + "=" * 60)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():20s}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
        
        if passed == total:
            print("🎉 All tests passed! Enhanced RA API is working correctly.")
        else:
            print(f"⚠️ Some tests failed. Please check the API configuration and models.")
        
        return test_results

def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RA API Test Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["all", "health", "predict", "analytics", "correlations"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    client = RAAPIClient(args.url)
    
    if args.test == "all":
        client.run_full_test_suite()
    elif args.test == "health":
        client.test_health_check()
    elif args.test == "predict":
        client.test_basic_prediction()
        client.test_multiple_scenarios()
    elif args.test == "analytics":
        client.test_analytics_endpoint()
    elif args.test == "correlations":
        client.test_correlation_analysis()

if __name__ == "__main__":
    main()