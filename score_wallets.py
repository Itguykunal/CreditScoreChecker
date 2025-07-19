#!/usr/bin/env python3
"""
Fixed DeFi Credit Scoring Script
Automatically detects field names and handles different JSON structures
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

class DeFiCreditScorer:
    """Credit scoring model for DeFi wallets with auto field detection"""
    
    def __init__(self):
        self.feature_weights = {
            'transaction_volume': 0.25,
            'repayment_behavior': 0.30,
            'risk_indicators': 0.20,
            'activity_consistency': 0.15,
            'portfolio_diversity': 0.10
        }
        self.user_field = None
        self.action_field = None
        self.timestamp_field = None
        self.reserve_field = None
        
    def detect_field_names(self, df):
        """Auto-detect field names in the dataset"""
        print("🔍 Auto-detecting field names...")
        
        columns = df.columns.tolist()
        print(f"Available columns: {columns}")
        
        # Detect user/wallet field
        user_candidates = [col for col in columns if any(term in col.lower() 
                          for term in ['user', 'wallet', 'address', 'from', 'account'])]
        
        if user_candidates:
            self.user_field = user_candidates[0]
        else:
            # Look for fields with wallet address patterns
            for col in columns:
                sample_values = df[col].dropna().head(10)
                if sample_values.dtype == 'object':
                    # Check if values look like Ethereum addresses
                    wallet_like = sample_values.astype(str).str.match(r'^0x[a-fA-F0-9]{40}$')
                    if wallet_like.any():
                        self.user_field = col
                        break
        
        # Detect action field
        action_candidates = [col for col in columns if any(term in col.lower() 
                            for term in ['action', 'type', 'event', 'function'])]
        if action_candidates:
            self.action_field = action_candidates[0]
        
        # Detect timestamp field
        timestamp_candidates = [col for col in columns if any(term in col.lower() 
                               for term in ['timestamp', 'time', 'date', 'block'])]
        if timestamp_candidates:
            self.timestamp_field = timestamp_candidates[0]
        
        # Detect reserve/asset field
        reserve_candidates = [col for col in columns if any(term in col.lower() 
                             for term in ['reserve', 'asset', 'token', 'underlying'])]
        if reserve_candidates:
            self.reserve_field = reserve_candidates[0]
        
        print(f"✅ Detected fields:")
        print(f"   User/Wallet: {self.user_field}")
        print(f"   Action: {self.action_field}")
        print(f"   Timestamp: {self.timestamp_field}")
        print(f"   Reserve/Asset: {self.reserve_field}")
        
        if not self.user_field:
            print("❌ Could not detect user/wallet field!")
            print("Available columns:", columns)
            raise ValueError("Cannot proceed without wallet identifier field")
    
    def load_transactions(self, file_path):
        """Load transaction data from JSON file with structure detection"""
        print(f"Loading transactions from {file_path}...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Try common keys for transaction lists
            possible_keys = ['transactions', 'data', 'records', 'events']
            df = None
            
            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    print(f"Found transaction data in '{key}' field")
                    break
            
            if df is None:
                # Try the first list found in the dict
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        df = pd.DataFrame(value)
                        print(f"Using data from '{key}' field")
                        break
            
            if df is None:
                raise ValueError("Could not find transaction data in JSON structure")
        else:
            raise ValueError("Unexpected JSON structure")
        
        # Auto-detect field names
        self.detect_field_names(df)
        
        print(f"Loaded {len(df)} transactions for {df[self.user_field].nunique()} unique wallets")
        return df
    
    def preprocess_data(self, df):
        """Clean and preprocess transaction data"""
        print("Preprocessing transaction data...")
        
        # Convert timestamps if available
        if self.timestamp_field and self.timestamp_field in df.columns:
            # Try different timestamp formats
            try:
                df[self.timestamp_field] = pd.to_datetime(df[self.timestamp_field], unit='s', errors='coerce')
            except:
                try:
                    df[self.timestamp_field] = pd.to_datetime(df[self.timestamp_field], errors='coerce')
                except:
                    print(f"⚠️ Could not parse timestamps in {self.timestamp_field}")
        
        # Convert amounts to numeric
        amount_cols = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower()]
        for col in amount_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=[self.user_field])
        
        return df
    
    def engineer_features(self, df):
        """Engineer features for credit scoring"""
        print("Engineering features...")
        
        features = {}
        
        for wallet in df[self.user_field].unique():
            wallet_data = df[df[self.user_field] == wallet].copy()
            wallet_features = self._calculate_wallet_features(wallet_data)
            features[wallet] = wallet_features
        
        feature_df = pd.DataFrame.from_dict(features, orient='index')
        return feature_df
    
    def _calculate_wallet_features(self, wallet_data):
        """Calculate features for a single wallet"""
        features = {}
        
        # Basic transaction metrics
        features['total_transactions'] = len(wallet_data)
        
        # Action-based features
        if self.action_field and self.action_field in wallet_data.columns:
            features['unique_actions'] = wallet_data[self.action_field].nunique()
            
            # Transaction type distribution
            action_counts = wallet_data[self.action_field].value_counts()
            features['deposits'] = action_counts.get('deposit', 0)
            features['borrows'] = action_counts.get('borrow', 0)
            features['repays'] = action_counts.get('repay', 0)
            features['liquidations'] = action_counts.get('liquidationcall', 0)
            features['redeems'] = action_counts.get('redeemunderlying', 0)
            
            # Also check for variations in naming
            for action_type in action_counts.index:
                if 'liquidat' in str(action_type).lower():
                    features['liquidations'] += action_counts[action_type]
                elif 'repay' in str(action_type).lower():
                    features['repays'] += action_counts[action_type]
                elif 'borrow' in str(action_type).lower():
                    features['borrows'] += action_counts[action_type]
                elif 'deposit' in str(action_type).lower():
                    features['deposits'] += action_counts[action_type]
        else:
            # Default values if no action field
            features['unique_actions'] = 1
            features['deposits'] = features['total_transactions']
            features['borrows'] = 0
            features['repays'] = 0
            features['liquidations'] = 0
            features['redeems'] = 0
        
        # Risk indicators
        features['liquidation_ratio'] = features['liquidations'] / max(features['total_transactions'], 1)
        features['borrow_ratio'] = features['borrows'] / max(features['total_transactions'], 1)
        
        # Repayment behavior
        if features['borrows'] > 0:
            features['repayment_rate'] = features['repays'] / features['borrows']
        else:
            features['repayment_rate'] = 1.0  # No borrows = perfect repayment
        
        # Activity consistency
        if self.timestamp_field and self.timestamp_field in wallet_data.columns:
            timestamps = wallet_data[self.timestamp_field].dropna()
            if len(timestamps) > 0:
                dates = timestamps.dt.date
                features['active_days'] = dates.nunique()
                features['account_age_days'] = (dates.max() - dates.min()).days + 1
                features['avg_transactions_per_day'] = features['total_transactions'] / max(features['account_age_days'], 1)
            else:
                features['active_days'] = 1
                features['account_age_days'] = 1
                features['avg_transactions_per_day'] = features['total_transactions']
        else:
            features['active_days'] = 1
            features['account_age_days'] = 1
            features['avg_transactions_per_day'] = features['total_transactions']
        
        # Portfolio diversity
        if self.reserve_field and self.reserve_field in wallet_data.columns:
            features['unique_assets'] = wallet_data[self.reserve_field].nunique()
        else:
            features['unique_assets'] = 1
        
        return features
    
    def calculate_scores(self, feature_df):
        """Calculate credit scores for all wallets"""
        print("Calculating credit scores...")
        
        scores = {}
        
        for wallet in feature_df.index:
            features = feature_df.loc[wallet]
            score = self._calculate_wallet_score(features)
            scores[wallet] = score
        
        return scores
    
    def _calculate_wallet_score(self, features):
        """Calculate credit score for a single wallet"""
        score = 0
        
        # 1. Transaction Volume Score (0-250 points)
        volume_score = min(250, features['total_transactions'] * 5)
        score += volume_score
        
        # 2. Repayment Behavior Score (0-300 points)
        repayment_score = features['repayment_rate'] * 300
        score += repayment_score
        
        # 3. Risk Indicators Score (0-200 points)
        risk_penalty = features['liquidation_ratio'] * 200
        risk_score = max(0, 200 - risk_penalty)
        score += risk_score
        
        # 4. Activity Consistency Score (0-150 points)
        if features['account_age_days'] > 0:
            consistency_score = min(150, (features['active_days'] / features['account_age_days']) * 150)
        else:
            consistency_score = 0
        score += consistency_score
        
        # 5. Portfolio Diversity Score (0-100 points)
        diversity_score = min(100, features['unique_assets'] * 20)
        score += diversity_score
        
        # Bonus for responsible behavior
        if features['repayment_rate'] >= 1.0 and features['liquidation_ratio'] == 0:
            score += 50  # Responsibility bonus
        
        # Penalty for bot-like behavior
        if features['avg_transactions_per_day'] > 100:
            score *= 0.5  # Heavy penalty for excessive activity
        
        return min(1000, max(0, int(score)))
    
    def generate_score_report(self, scores, feature_df):
        """Generate analysis report of scores"""
        print("Generating score analysis...")
        
        score_df = pd.DataFrame.from_dict(scores, orient='index', columns=['credit_score'])
        score_df = score_df.join(feature_df)
        
        # Score distribution
        bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        score_df['score_range'] = pd.cut(score_df['credit_score'], bins=bins, include_lowest=True)
        
        distribution = score_df['score_range'].value_counts().sort_index()
        
        analysis = {
            'total_wallets': len(scores),
            'average_score': score_df['credit_score'].mean(),
            'median_score': score_df['credit_score'].median(),
            'score_distribution': {str(k): int(v) for k, v in distribution.to_dict().items()},
            'high_risk_wallets': len(score_df[score_df['credit_score'] < 300]),
            'excellent_wallets': len(score_df[score_df['credit_score'] > 800])
        }
        
        return analysis, score_df
    
    def save_results(self, scores, analysis, output_file='wallet_scores.json'):
        """Save scoring results to file"""
        # Convert numpy types to native Python types for JSON serialization
        clean_scores = {str(k): int(v) for k, v in scores.items()}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0',
            'total_wallets_scored': len(scores),
            'field_mappings': {
                'user_field': self.user_field,
                'action_field': self.action_field,
                'timestamp_field': self.timestamp_field,
                'reserve_field': self.reserve_field
            },
            'analysis': analysis,
            'wallet_scores': clean_scores
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")

def main():
    """Main scoring function"""
    parser = argparse.ArgumentParser(description='DeFi Credit Scoring Script (Auto-detecting field names)')
    parser.add_argument('input_file', help='Path to transaction JSON file')
    parser.add_argument('-o', '--output', default='wallet_scores.json', 
                       help='Output file for scores (default: wallet_scores.json)')
    
    args = parser.parse_args()
    
    # Initialize scorer
    scorer = DeFiCreditScorer()
    
    try:
        # Load and process data
        df = scorer.load_transactions(args.input_file)
        df = scorer.preprocess_data(df)
        
        # Engineer features
        feature_df = scorer.engineer_features(df)
        
        # Calculate scores
        scores = scorer.calculate_scores(feature_df)
        
        # Generate analysis
        analysis, score_df = scorer.generate_score_report(scores, feature_df)
        
        # Print summary
        print(f"\n=== SCORING COMPLETE ===")
        print(f"Total wallets scored: {analysis['total_wallets']}")
        print(f"Average score: {analysis['average_score']:.2f}")
        print(f"Median score: {analysis['median_score']:.2f}")
        print(f"High-risk wallets (< 300): {analysis['high_risk_wallets']}")
        print(f"Excellent wallets (> 800): {analysis['excellent_wallets']}")
        
        # Save results
        scorer.save_results(scores, analysis, args.output)
        
        return scores, analysis, score_df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    scores, analysis, score_df = main()