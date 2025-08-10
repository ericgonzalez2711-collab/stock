"""
Machine Learning module for predicting next-day stock price movements.
Implements Decision Tree and Logistic Regression models using technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.technical_indicators import TechnicalIndicators
from ..config import Config


class TechnicalFeatureEngineer:
    """Feature engineering for technical indicators."""
    
    @staticmethod
    def create_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from OHLCV data and technical indicators.
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with engineered features
        """
        features_df = data.copy()
        
        # Ensure we have technical indicators
        if 'rsi' not in features_df.columns:
            features_df = TechnicalIndicators.calculate_all_indicators(features_df)
        
        # Price-based features
        features_df['price_momentum_5'] = features_df['close'].pct_change(5)
        features_df['price_momentum_10'] = features_df['close'].pct_change(10)
        features_df['price_momentum_20'] = features_df['close'].pct_change(20)
        
        # RSI momentum
        features_df['rsi_momentum'] = features_df['rsi'].diff()
        features_df['rsi_sma_5'] = features_df['rsi'].rolling(5).mean()
        
        # MACD features
        if 'macd' in features_df.columns:
            features_df['macd_momentum'] = features_df['macd'].diff()
            features_df['macd_signal_diff'] = features_df['macd'] - features_df['signal']
        
        # Moving average relationships
        if 'sma_20' in features_df.columns and 'sma_50' in features_df.columns:
            features_df['ma_ratio'] = features_df['sma_20'] / (features_df['sma_50'] + 1e-10)
            features_df['ma_ratio'] = np.clip(features_df['ma_ratio'], 0.5, 2.0)  # Clip to reasonable range
            features_df['price_above_ma20'] = (features_df['close'] > features_df['sma_20']).astype(int)
            features_df['price_above_ma50'] = (features_df['close'] > features_df['sma_50']).astype(int)
        
        # Bollinger Bands features
        if 'bb_upper' in features_df.columns:
            bb_range = features_df['bb_upper'] - features_df['bb_lower']
            features_df['bb_position'] = ((features_df['close'] - features_df['bb_lower']) / 
                                        (bb_range + 1e-10))
            features_df['bb_position'] = np.clip(features_df['bb_position'], -1, 2)  # Clip extreme values
            features_df['bb_squeeze'] = (bb_range / (features_df['bb_middle'] + 1e-10))
            features_df['bb_squeeze'] = np.clip(features_df['bb_squeeze'], 0, 1)  # Clip extreme values
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_momentum'] = features_df['volume'].pct_change()
            features_df['volume_momentum'] = np.clip(features_df['volume_momentum'], -10, 10)  # Clip extreme values
            
            volume_sma = features_df['volume'].rolling(20).mean()
            features_df['volume_sma_ratio'] = features_df['volume'] / (volume_sma + 1e-10)
            features_df['volume_sma_ratio'] = np.clip(features_df['volume_sma_ratio'], 0, 10)  # Clip extreme values
            
            # Price-volume relationship
            price_change = features_df['close'].pct_change()
            price_change = np.clip(price_change, -1, 1)  # Clip extreme price changes
            features_df['pv_trend'] = price_change * features_df['volume_momentum']
        
        # Volatility features
        features_df['volatility_momentum'] = features_df['volatility'].diff()
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        
        # Support/Resistance features
        if 'support' in features_df.columns and 'resistance' in features_df.columns:
            features_df['support_distance'] = (features_df['close'] - features_df['support']) / (features_df['close'] + 1e-10)
            features_df['support_distance'] = np.clip(features_df['support_distance'], -1, 1)
            features_df['resistance_distance'] = (features_df['resistance'] - features_df['close']) / (features_df['close'] + 1e-10)
            features_df['resistance_distance'] = np.clip(features_df['resistance_distance'], -1, 1)
        
        # Stochastic features
        if 'k_percent' in features_df.columns:
            features_df['stoch_momentum'] = features_df['k_percent'].diff()
            features_df['stoch_oversold'] = (features_df['k_percent'] < 20).astype(int)
            features_df['stoch_overbought'] = (features_df['k_percent'] > 80).astype(int)
        
        # Market regime features
        if 'sma_20' in features_df.columns and 'sma_50' in features_df.columns:
            features_df['trend_strength'] = abs(features_df['sma_20'] - features_df['sma_50']) / (features_df['close'] + 1e-10)
            features_df['trend_strength'] = np.clip(features_df['trend_strength'], 0, 1)
        else:
            features_df['trend_strength'] = 0
            
        if 'volatility' in features_df.columns:
            vol_mean = features_df['volatility'].rolling(50).mean()
            features_df['volatility_regime'] = (features_df['volatility'] > vol_mean).astype(int)
        else:
            features_df['volatility_regime'] = 0
        
        return features_df
    
    @staticmethod
    def create_target(data: pd.DataFrame, target_type: str = 'direction') -> pd.Series:
        """
        Create target variable for prediction.
        
        Args:
            data: DataFrame with price data
            target_type: 'direction' (up/down) or 'return' (continuous)
            
        Returns:
            Series with target values
        """
        if target_type == 'direction':
            # Next day price direction (1 = up, 0 = down/flat)
            next_day_return = data['close'].shift(-1) / data['close'] - 1
            target = (next_day_return > 0.005).astype(int)  # 0.5% threshold for "up"
        elif target_type == 'return':
            # Next day return
            target = data['close'].shift(-1) / data['close'] - 1
        else:
            raise ValueError("target_type must be 'direction' or 'return'")
        
        return target
    
    @staticmethod
    def select_features(data: pd.DataFrame) -> List[str]:
        """
        Select relevant features for ML model.
        
        Args:
            data: DataFrame with all features
            
        Returns:
            List of selected feature names
        """
        # Core technical indicators
        core_features = [
            'rsi', 'macd', 'histogram', 'signal',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'volatility', 'atr', 'volume_ratio'
        ]
        
        # Momentum features
        momentum_features = [
            'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'rsi_momentum', 'macd_momentum', 'volume_momentum',
            'volatility_momentum'
        ]
        
        # Relative position features
        position_features = [
            'ma_ratio', 'price_above_ma20', 'price_above_ma50',
            'bb_position', 'support_distance', 'resistance_distance'
        ]
        
        # Market regime features
        regime_features = [
            'trend_strength', 'volatility_regime',
            'stoch_oversold', 'stoch_overbought'
        ]
        
        # Combine all feature groups
        all_features = core_features + momentum_features + position_features + regime_features
        
        # Return only features that exist in the data
        available_features = [f for f in all_features if f in data.columns]
        
        logger.info(f"Selected {len(available_features)} features for ML model")
        return available_features


class StockPredictiveModel:
    """Machine Learning model for predicting stock price movements."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the predictive model.
        
        Args:
            model_type: Type of model ('decision_tree', 'logistic_regression', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.feature_importance = None
        self.model_metrics = {}
        
        # Initialize model based on type
        if model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Initialized {model_type} model")
    
    def prepare_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training by combining multiple stocks and engineering features.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            
        Returns:
            Tuple of (features_df, target_series)
        """
        all_features = []
        all_targets = []
        
        for symbol, data in data_dict.items():
            if data.empty:
                continue
            
            logger.info(f"Preparing data for {symbol}...")
            
            # Engineer features
            features_data = TechnicalFeatureEngineer.create_features(data)
            
            # Create target
            target_data = TechnicalFeatureEngineer.create_target(features_data, 'direction')
            
            # Select features
            feature_cols = TechnicalFeatureEngineer.select_features(features_data)
            
            # Filter data
            features_subset = features_data[feature_cols].copy()
            features_subset['symbol'] = symbol
            
            # Remove rows with NaN values
            valid_indices = features_subset.dropna().index.intersection(target_data.dropna().index)
            
            if len(valid_indices) > 100:  # Minimum data requirement
                all_features.append(features_subset.loc[valid_indices])
                all_targets.append(target_data.loc[valid_indices])
                logger.success(f"Added {len(valid_indices)} valid samples from {symbol}")
            else:
                logger.warning(f"Insufficient valid data for {symbol}: {len(valid_indices)} samples")
        
        if not all_features:
            raise ValueError("No valid data available for training")
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_target = pd.concat(all_targets, ignore_index=True)
        
        # Remove symbol column for training (keeping for reference)
        symbol_column = combined_features['symbol']
        combined_features = combined_features.drop('symbol', axis=1)
        
        # Clean data - handle infinities and extreme values
        combined_features = self._clean_data(combined_features)
        
        logger.success(f"Prepared dataset with {len(combined_features)} samples and {len(combined_features.columns)} features")
        
        return combined_features, combined_target
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling infinities, NaNs, and extreme values.
        
        Args:
            data: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Replace infinities with NaN
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with column median
        for col in cleaned_data.columns:
            if cleaned_data[col].dtype in ['float64', 'int64']:
                median_val = cleaned_data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                cleaned_data[col] = cleaned_data[col].fillna(median_val)
                
                # Clip extreme values (beyond 99.9% percentile)
                if not cleaned_data[col].empty:
                    lower_bound = cleaned_data[col].quantile(0.001)
                    upper_bound = cleaned_data[col].quantile(0.999)
                    cleaned_data[col] = np.clip(cleaned_data[col], lower_bound, upper_bound)
        
        # Remove any remaining NaN rows
        cleaned_data = cleaned_data.dropna()
        
        logger.info(f"Data cleaning completed. Removed {len(data) - len(cleaned_data)} problematic rows")
        
        return cleaned_data
    
    def train(self, data_dict: Dict[str, pd.DataFrame], test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the ML model.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        # Prepare data
        X, y = self.prepare_data(data_dict)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features for logistic regression
        if self.model_type == 'logistic_regression':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        logger.info(f"Training {self.model_type} on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Detailed classification report
        class_report = classification_report(y_test, y_pred_test, output_dict=True)
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Store metrics
        self.model_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'feature_importance': self.feature_importance
        }
        
        self.is_trained = True
        
        logger.success(f"Model training completed. Test accuracy: {test_accuracy:.4f}")
        
        return self.model_metrics
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Engineer features
        features_data = TechnicalFeatureEngineer.create_features(data)
        
        # Select features
        feature_cols = TechnicalFeatureEngineer.select_features(features_data)
        X = features_data[feature_cols].copy()
        
        # Clean the data
        X = self._clean_data(X)
        
        if X.empty:
            logger.warning("No valid data for prediction")
            return {'prediction': None, 'probability': None}
        
        # Scale if needed
        if self.model_type == 'logistic_regression':
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get latest prediction
        latest_prediction = prediction[-1]
        latest_probability = probabilities[-1]
        
        return {
            'prediction': int(latest_prediction),
            'probability': {
                'down': float(latest_probability[0]),
                'up': float(latest_probability[1])
            },
            'confidence': float(max(latest_probability)),
            'timestamp': X.index[-1] if not X.empty else None
        }
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get top N most important features."""
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model type")
            return pd.DataFrame()
        
        return self.feature_importance.head(top_n)
    
    def plot_model_analysis(self, save_path: str = None):
        """Plot model analysis including feature importance and confusion matrix."""
        if not self.is_trained:
            logger.error("Model must be trained before plotting analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_type.title()} Model Analysis', fontsize=16)
        
        # Feature importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(15)
            axes[0, 0].barh(range(len(top_features)), top_features['importance'])
            axes[0, 0].set_yticks(range(len(top_features)))
            axes[0, 0].set_yticklabels(top_features['feature'])
            axes[0, 0].set_title('Top 15 Feature Importance')
            axes[0, 0].set_xlabel('Importance')
        
        # Confusion matrix
        cm = self.model_metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                   ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_ylabel('Actual')
        axes[0, 1].set_xlabel('Predicted')
        
        # Model metrics comparison
        metrics_data = {
            'Train Accuracy': self.model_metrics['train_accuracy'],
            'Test Accuracy': self.model_metrics['test_accuracy'],
            'CV Mean': self.model_metrics['cv_mean']
        }
        
        axes[1, 0].bar(metrics_data.keys(), metrics_data.values())
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (key, value) in enumerate(metrics_data.items()):
            axes[1, 0].text(i, value + 0.01, f'{value:.3f}', ha='center')
        
        # Precision-Recall by class
        class_report = self.model_metrics['classification_report']
        precision_up = class_report['1']['precision']
        recall_up = class_report['1']['recall']
        f1_up = class_report['1']['f1-score']
        
        precision_down = class_report['0']['precision']
        recall_down = class_report['0']['recall']
        f1_down = class_report['0']['f1-score']
        
        x = ['Precision', 'Recall', 'F1-Score']
        up_scores = [precision_up, recall_up, f1_up]
        down_scores = [precision_down, recall_down, f1_down]
        
        x_pos = np.arange(len(x))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, down_scores, width, label='Down', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, up_scores, width, label='Up', alpha=0.8)
        axes[1, 1].set_title('Precision, Recall, and F1-Score by Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(x)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model analysis plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_metrics': self.model_metrics,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.model_metrics = model_data['model_metrics']
        self.feature_importance = model_data.get('feature_importance')
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the trained model."""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        summary = {
            'model_type': self.model_type,
            'test_accuracy': self.model_metrics['test_accuracy'],
            'cv_accuracy': f"{self.model_metrics['cv_mean']:.4f} ± {self.model_metrics['cv_std']:.4f}",
            'num_features': len(self.feature_names),
            'top_features': self.get_feature_importance(5).to_dict('records') if self.feature_importance is not None else []
        }
        
        return summary