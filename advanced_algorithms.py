# advanced_algorithms.py - Enhanced ML Models for RA Flare Prediction

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Optional: LSTM for time series (requires tensorflow)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be skipped.")

class AdvancedMLEnsemble:
    """Advanced ML ensemble with multiple algorithms for RA flare prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_scores = {}
        self.is_trained = False
        
    def create_base_models(self):
        """Initialize advanced base models"""
        models = {
            # Enhanced Random Forest
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            # Enhanced Gradient Boosting
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            
            # XGBoost - High performance gradient boosting
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                random_state=42
            ),
            
            # LightGBM - Fast gradient boosting
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ),
            
            # Neural Network
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=1000,
                learning_rate_init=0.001,
                random_state=42
            )
        }
        
        return models
    
    def create_stacking_ensemble(self, base_models):
        """Create stacking ensemble with meta-learner"""
        
        # Base estimators for stacking
        base_estimators = [
            ('rf', base_models['random_forest']),
            ('xgb', base_models['xgboost']),
            ('lgb', base_models['lightgbm']),
            ('nn', base_models['neural_network'])
        ]
        
        # Meta-learner (final estimator)
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,  # 5-fold cross-validation for meta-features
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        return stacking_model
    
    def create_lstm_model(self, sequence_length=7, n_features=20):
        """Create LSTM model for sequential pattern learning"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_lstm_sequences(self, X, y, sequence_length=7):
        """Prepare sequences for LSTM training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        print("Training advanced ML ensemble...")
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Flare rate: {y_train.mean():.2%}")
        
        # Initialize scalers
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = StandardScaler()  # Could use RobustScaler if needed
        
        # Scale features
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Create base models
        base_models = self.create_base_models()
        
        # Train individual models
        for name, model in base_models.items():
            print(f"\nTraining {name}...")
            
            try:
                if name in ['neural_network']:
                    # Use scaled features for neural network
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    # Use original features for tree-based models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation score
                if name in ['neural_network']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                
                # Store results
                self.models[name] = model
                self.performance_scores[name] = {
                    'accuracy': accuracy,
                    'auc_score': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                print(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc_score:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Train stacking ensemble
        print(f"\nTraining stacking ensemble...")
        try:
            # Use models that trained successfully
            successful_models = {name: model for name, model in base_models.items() 
                               if name in self.models}
            
            if len(successful_models) >= 2:
                stacking_model = self.create_stacking_ensemble(successful_models)
                stacking_model.fit(X_train, y_train)
                
                y_pred_stack = stacking_model.predict(X_test)
                y_pred_proba_stack = stacking_model.predict_proba(X_test)[:, 1]
                
                accuracy_stack = accuracy_score(y_test, y_pred_stack)
                auc_stack = roc_auc_score(y_test, y_pred_proba_stack)
                
                self.models['stacking_ensemble'] = stacking_model
                self.performance_scores['stacking_ensemble'] = {
                    'accuracy': accuracy_stack,
                    'auc_score': auc_stack,
                    'cv_mean': 0.0,  # Would need separate CV for stacking
                    'cv_std': 0.0
                }
                
                print(f"Stacking Ensemble - Accuracy: {accuracy_stack:.3f}, AUC: {auc_stack:.3f}")
        
        except Exception as e:
            print(f"Error training stacking ensemble: {e}")
        
        # Train LSTM model (if available and enough data)
        if TENSORFLOW_AVAILABLE and len(X_train) > 50:
            print(f"\nTraining LSTM model...")
            try:
                # Prepare sequences
                X_train_seq, y_train_seq = self.prepare_lstm_sequences(X_train_scaled, y_train)
                X_test_seq, y_test_seq = self.prepare_lstm_sequences(X_test_scaled, y_test)
                
                if len(X_train_seq) > 20:  # Minimum sequences needed
                    lstm_model = self.create_lstm_model(n_features=X_train.shape[1])
                    
                    # Train with early stopping
                    history = lstm_model.fit(
                        X_train_seq, y_train_seq,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    # Evaluate
                    y_pred_lstm = (lstm_model.predict(X_test_seq) > 0.5).astype(int).flatten()
                    y_pred_proba_lstm = lstm_model.predict(X_test_seq).flatten()
                    
                    accuracy_lstm = accuracy_score(y_test_seq, y_pred_lstm)
                    auc_lstm = roc_auc_score(y_test_seq, y_pred_proba_lstm)
                    
                    self.models['lstm'] = lstm_model
                    self.performance_scores['lstm'] = {
                        'accuracy': accuracy_lstm,
                        'auc_score': auc_lstm,
                        'cv_mean': 0.0,
                        'cv_std': 0.0
                    }
                    
                    print(f"LSTM - Accuracy: {accuracy_lstm:.3f}, AUC: {auc_lstm:.3f}")
                    
            except Exception as e:
                print(f"Error training LSTM: {e}")
        
        self.is_trained = True
        print(f"\nTraining completed! {len(self.models)} models trained successfully.")
        return self.performance_scores
    
    def predict_ensemble(self, features):
        """Make predictions using ensemble of trained models"""
        if not self.is_trained:
            raise ValueError("Models must be trained first")
        
        predictions = {}
        probabilities = {}
        
        # Scale features
        features_scaled = self.scalers['standard'].transform(features)
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if name == 'lstm':
                    # Handle LSTM prediction (needs sequences)
                    if len(features) >= 7:
                        seq_features = features_scaled[-7:].reshape(1, 7, -1)
                        prob = model.predict(seq_features)[0][0]
                    else:
                        prob = 0.5  # Default if not enough history
                elif name in ['neural_network']:
                    prob = model.predict_proba(features_scaled)[0][1]
                elif name == 'stacking_ensemble':
                    prob = model.predict_proba(features)[0][1]
                else:
                    prob = model.predict_proba(features)[0][1]
                
                probabilities[name] = float(prob)
                predictions[name] = int(prob > 0.5)
                
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                continue
        
        # Weighted ensemble (based on performance)
        if probabilities:
            weights = self._calculate_model_weights()
            ensemble_prob = sum(probabilities[name] * weights.get(name, 0) 
                              for name in probabilities) / sum(weights.values())
        else:
            ensemble_prob = 0.5
        
        return {
            'ensemble_probability': float(ensemble_prob),
            'individual_predictions': probabilities,
            'binary_predictions': predictions
        }
    
    def _calculate_model_weights(self):
        """Calculate weights based on model performance"""
        weights = {}
        
        for name, scores in self.performance_scores.items():
            # Weight based on AUC score (higher is better)
            auc = scores.get('auc_score', 0.5)
            weights[name] = max(0, auc - 0.5) ** 2  # Square to emphasize better models
        
        # Ensure some weight even for poor models
        for name in weights:
            weights[name] = max(weights[name], 0.1)
        
        return weights
    
    def get_feature_importance_analysis(self, feature_names):
        """Analyze feature importance across models"""
        importance_analysis = {}
        
        # Get importance from models that support it
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                try:
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    importance_analysis[name] = importance_df
                except Exception as e:
                    print(f"Warning: Could not get feature importance for {name}: {e}")
                    continue
        
        # Calculate average importance across models (FIXED)
        if importance_analysis:
            # Create base dataframe with all features
            avg_importance = pd.DataFrame({'feature': feature_names, 'avg_importance': 0.0})
            model_count = 0
            
            for model_name, model_importance in importance_analysis.items():
                # Merge importances, using 0 for missing features
                temp_df = model_importance[['feature', 'importance']].rename(
                    columns={'importance': f'imp_{model_name}'}
                )
                avg_importance = avg_importance.merge(temp_df, on='feature', how='left')
                avg_importance[f'imp_{model_name}'] = avg_importance[f'imp_{model_name}'].fillna(0)
                avg_importance['avg_importance'] += avg_importance[f'imp_{model_name}']
                model_count += 1
            
            # Calculate actual average
            if model_count > 0:
                avg_importance['avg_importance'] = avg_importance['avg_importance'] / model_count
            
            # Sort by average importance
            avg_importance = avg_importance.sort_values('avg_importance', ascending=False)
            importance_analysis['average'] = avg_importance[['feature', 'avg_importance']]
        
        return importance_analysis
    
    def save_models(self, filepath_prefix):
        """Save all trained models"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'performance_scores': self.performance_scores,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        # Save main model data
        joblib.dump(model_data, f"{filepath_prefix}_ensemble.joblib")
        
        # Save LSTM separately if it exists (joblib can't handle TensorFlow models)
        if 'lstm' in self.models and TENSORFLOW_AVAILABLE:
            self.models['lstm'].save(f"{filepath_prefix}_lstm.h5")
            # Remove LSTM from main save to avoid errors
            temp_models = model_data['models'].copy()
            del temp_models['lstm']
            model_data['models'] = temp_models
            joblib.dump(model_data, f"{filepath_prefix}_ensemble.joblib")
        
        print(f"Models saved to {filepath_prefix}_ensemble.joblib")
        return f"{filepath_prefix}_ensemble.joblib"


# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    # Load enhanced training data (from feature engineering step)
    df = pd.read_csv('enhanced_training_data.csv')
    
    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'flare_occurred']
    X = df[feature_columns]
    y = df['flare_occurred']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train advanced ensemble
    ensemble = AdvancedMLEnsemble()
    performance = ensemble.train_models(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    for model_name, scores in performance.items():
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {scores['accuracy']:.3f}")
        print(f"  AUC Score: {scores['auc_score']:.3f}")
        print(f"  CV Score: {scores['cv_mean']:.3f} ± {scores['cv_std']:.3f}")
        print()
    
    # Feature importance analysis
    importance_analysis = ensemble.get_feature_importance_analysis(feature_columns)
    if 'average' in importance_analysis:
        print("TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        top_features = importance_analysis['average'].head(10)
        for _, row in top_features.iterrows():
            print(f"{row['feature']:30s}: {row['avg_importance']:.4f}")
    
    # Save models
    ensemble.save_models('models/advanced_ra_model')