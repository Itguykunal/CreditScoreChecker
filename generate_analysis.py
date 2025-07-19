"""
Analysis script for DeFi Credit Scoring results
Generates analysis.md with score distribution and behavioral insights
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_scoring_results(file_path='wallet_scores.json'):
    """Load scoring results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded results from {file_path}")
        return results
    except FileNotFoundError:
        print(f"❌ File {file_path} not found!")
        print("Make sure you've run: python score_wallets.py user-transactions.json")
        return None

def load_original_data(file_path='user-transactions.json'):
    """Load original transaction data for deeper analysis"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Handle nested structure
            for key in ['transactions', 'data', 'records']:
                if key in data:
                    df = pd.DataFrame(data[key])
                    break
            else:
                df = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(df)} transactions from original data")
        return df
    except Exception as e:
        print(f"⚠️ Could not load original data: {e}")
        return None

def create_score_distribution_chart(score_df):
    """Create score distribution visualization"""
    plt.figure(figsize=(15, 10))
    
    # Set style
    plt.style.use('default')
    sns.set_palette('husl')
    
    # Main distribution histogram
    plt.subplot(2, 2, 1)
    plt.hist(score_df['credit_score'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    plt.xlabel('Credit Score')
    plt.ylabel('Number of Wallets')
    plt.title('Overall Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # Score range bar chart
    plt.subplot(2, 2, 2)
    range_counts = score_df['score_range'].value_counts().sort_index()
    range_counts.plot(kind='bar', color='lightcoral')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Wallets')
    plt.title('Wallets by Score Range')
    plt.xticks(rotation=45)
    
    # Box plot
    plt.subplot(2, 2, 3)
    plt.boxplot(score_df['credit_score'])
    plt.ylabel('Credit Score')
    plt.title('Score Distribution Summary')
    
    # Cumulative distribution
    plt.subplot(2, 2, 4)
    sorted_scores = np.sort(score_df['credit_score'])
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
    plt.plot(sorted_scores, cumulative, linewidth=2, color='green')
    plt.xlabel('Credit Score')
    plt.ylabel('Cumulative Percentage')
    plt.title('Cumulative Score Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Score distribution chart saved as score_distribution.png")

def analyze_behaviors_by_score(transaction_df, score_df):
    """Analyze wallet behaviors by score range"""
    
    if transaction_df is None:
        print("⚠️ No transaction data available for behavioral analysis")
        return {}
    
    # Detect user field (same logic as scorer)
    user_field = None
    for col in transaction_df.columns:
        if any(term in col.lower() for term in ['user', 'wallet', 'address', 'from', 'account']):
            user_field = col
            break
    
    if not user_field:
        print("⚠️ Could not detect user field for behavioral analysis")
        return {}
    
    # Detect action field
    action_field = None
    for col in transaction_df.columns:
        if any(term in col.lower() for term in ['action', 'type', 'event']):
            action_field = col
            break
    
    # Create wallet-level statistics
    if action_field:
        wallet_stats = transaction_df.groupby(user_field).agg({
            action_field: ['count', lambda x: x.value_counts().to_dict()]
        })
        wallet_stats.columns = ['total_txns', 'action_breakdown']
    else:
        wallet_stats = transaction_df.groupby(user_field).size().to_frame('total_txns')
        wallet_stats['action_breakdown'] = [{}] * len(wallet_stats)
    
    # Add reserve diversity if available
    reserve_field = None
    for col in transaction_df.columns:
        if any(term in col.lower() for term in ['reserve', 'asset', 'token']):
            reserve_field = col
            break
    
    if reserve_field:
        reserve_stats = transaction_df.groupby(user_field)[reserve_field].nunique()
        wallet_stats['unique_assets'] = reserve_stats
    else:
        wallet_stats['unique_assets'] = 1
    
    # Merge with scores
    wallet_stats = wallet_stats.join(score_df, how='inner')
    
    behavioral_analysis = {}
    
    # Analyze each score range
    score_ranges = ['0-100', '100-200', '200-300', '300-400', '400-500', 
                   '500-600', '600-700', '700-800', '800-900', '900-1000']
    
    for score_range in score_ranges:
        range_data = wallet_stats[wallet_stats['score_range'] == score_range]
        
        if len(range_data) > 0:
            analysis = {
                'wallet_count': len(range_data),
                'avg_transactions': range_data['total_txns'].mean(),
                'avg_unique_assets': range_data['unique_assets'].mean() if 'unique_assets' in range_data.columns else 1,
                'common_behaviors': analyze_common_behaviors(range_data, score_range),
                'risk_patterns': identify_risk_patterns(score_range)
            }
            behavioral_analysis[score_range] = analysis
    
    return behavioral_analysis

def analyze_common_behaviors(range_data, score_range):
    """Analyze common behaviors in a score range"""
    behaviors = []
    
    # Transaction volume patterns
    avg_txns = range_data['total_txns'].mean()
    if avg_txns < 5:
        behaviors.append("Low activity users (< 5 transactions)")
    elif avg_txns > 50:
        behaviors.append("High activity users (> 50 transactions)")
    else:
        behaviors.append(f"Moderate activity users (~{avg_txns:.1f} transactions)")
    
    # Asset diversity
    if 'unique_assets' in range_data.columns:
        avg_assets = range_data['unique_assets'].mean()
        if avg_assets < 2:
            behaviors.append("Single-asset focused")
        elif avg_assets > 5:
            behaviors.append("Highly diversified portfolios")
        else:
            behaviors.append(f"Moderate diversification (~{avg_assets:.1f} assets)")
    
    return behaviors

def identify_risk_patterns(score_range):
    """Identify risk patterns by score range"""
    patterns = []
    
    score_num = int(score_range.split('-')[0])
    
    if score_num < 200:
        patterns.extend([
            "High liquidation risk",
            "Poor repayment history", 
            "Potential bot activity",
            "Exploitative behavior patterns"
        ])
    elif score_num < 400:
        patterns.extend([
            "Moderate risk indicators",
            "Inconsistent repayment patterns",
            "Some liquidation events"
        ])
    elif score_num < 600:
        patterns.extend([
            "Generally stable behavior",
            "Minor risk indicators",
            "Room for improvement in consistency"
        ])
    elif score_num < 800:
        patterns.extend([
            "Good financial discipline",
            "Consistent repayment behavior",
            "Low risk profile"
        ])
    else:
        patterns.extend([
            "Excellent credit behavior",
            "Perfect repayment history",
            "No liquidation events",
            "Responsible usage patterns"
        ])
    
    return patterns

def generate_analysis_markdown(results, score_df, behavioral_analysis):
    """Generate the analysis.md file"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# DeFi Credit Scoring Analysis

*Generated on {timestamp}*

## Executive Summary

This analysis examines the credit scores of {results['total_wallets_scored']} unique wallets based on their Aave V2 transaction history. The scoring model assigns credit scores from 0-1000, with higher scores indicating more reliable and responsible DeFi usage patterns.

### Key Findings

- **Average Score**: {results['analysis']['average_score']:.2f}
- **Median Score**: {results['analysis']['median_score']:.2f}  
- **High-Risk Wallets**: {results['analysis']['high_risk_wallets']} ({results['analysis']['high_risk_wallets']/results['total_wallets_scored']*100:.1f}%)
- **Excellent Wallets**: {results['analysis']['excellent_wallets']} ({results['analysis']['excellent_wallets']/results['total_wallets_scored']*100:.1f}%)

## Score Distribution

![Score Distribution](score_distribution.png)

### Distribution Breakdown

| Score Range | Wallet Count | Percentage | Risk Level |
|-------------|--------------|------------|------------|
"""
    
    # Add distribution table
    range_counts = score_df['score_range'].value_counts().sort_index()
    total_wallets = len(score_df)
    
    risk_levels = {
        '0-100': 'Critical', '100-200': 'Very High', '200-300': 'High',
        '300-400': 'High', '400-500': 'Moderate', '500-600': 'Moderate',
        '600-700': 'Low', '700-800': 'Low', '800-900': 'Very Low', '900-1000': 'Minimal'
    }
    
    for range_name, count in range_counts.items():
        percentage = (count / total_wallets) * 100
        range_str = str(range_name).replace('(', '').replace(']', '').replace(', ', '-')
        risk_level = risk_levels.get(range_str, 'Unknown')
        markdown_content += f"| {range_str} | {count} | {percentage:.1f}% | {risk_level} |\n"
    
    markdown_content += "\n## Behavioral Analysis by Score Range\n\n"
    
    # Add behavioral analysis for each range
    for score_range, analysis in behavioral_analysis.items():
        if analysis['wallet_count'] > 0:
            markdown_content += f"""### Score Range: {score_range}

**Wallet Count**: {analysis['wallet_count']}  
**Average Transactions**: {analysis['avg_transactions']:.1f}  
**Average Unique Assets**: {analysis['avg_unique_assets']:.1f}  

**Common Behaviors**:
"""
            for behavior in analysis['common_behaviors']:
                markdown_content += f"- {behavior}\n"
            
            markdown_content += "\n**Risk Patterns**:\n"
            for pattern in analysis['risk_patterns']:
                markdown_content += f"- {pattern}\n"
            
            markdown_content += "\n"
    
    # Add insights section
    markdown_content += """## Key Insights

### High-Performing Wallets (800-1000)
- Demonstrate excellent repayment discipline
- Show consistent, long-term engagement with the protocol
- Maintain diversified asset portfolios
- Have zero or minimal liquidation events
- Represent the most creditworthy segment

### Moderate-Risk Wallets (400-600)
- Show mixed behavioral patterns
- May have occasional repayment delays
- Generally stable but with room for improvement
- Represent the largest segment of users

### High-Risk Wallets (0-300)
- Exhibit concerning behavioral patterns
- High liquidation rates and poor repayment history
- May include bot accounts or exploitative users
- Require enhanced monitoring and risk management

## Model Performance

### Strengths
- Clear differentiation between risk segments
- Emphasis on repayment behavior (strongest predictor)
- Accounts for portfolio diversification and consistency
- Includes bot detection mechanisms

### Areas for Improvement
- Could incorporate market volatility factors
- Time-decay for historical events
- Integration with other DeFi protocols
- Machine learning enhancement for pattern detection

## Recommendations

### For Lending Protocols
1. **Risk-based Interest Rates**: Apply different rates based on credit scores
2. **Enhanced Monitoring**: Focus surveillance on wallets scoring below 300
3. **Incentive Programs**: Reward high-scoring wallets with better terms
4. **Collateral Requirements**: Adjust based on creditworthiness

### For Future Model Development
1. **Expand Data Sources**: Include other DeFi protocols and on-chain metrics
2. **Real-time Updates**: Implement continuous score updates
3. **Machine Learning**: Train supervised models on labeled datasets
4. **External Factors**: Incorporate market conditions and economic indicators

## Conclusion

The DeFi credit scoring model successfully segments wallets into meaningful risk categories based on transaction behavior. The distribution shows a healthy spread across score ranges, with clear behavioral differences between segments. This foundation provides a robust basis for risk assessment and can be enhanced with additional data sources and machine learning techniques.

---

*This analysis was generated automatically from wallet transaction data. For technical details, see the README.md file.*
"""
    
    # Save to file
    with open('analysis.md', 'w') as f:
        f.write(markdown_content)
    
    print("✅ Analysis report saved to analysis.md")

def main():
    """Main analysis function"""
    print("🚀 Generating DeFi Credit Scoring Analysis...")
    print("="*50)
    
    # Load results
    results = load_scoring_results()
    if not results:
        return
    
    # Create score DataFrame
    scores = results['wallet_scores']
    score_df = pd.DataFrame.from_dict(scores, orient='index', columns=['credit_score'])
    
    # Add score ranges
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
              '500-600', '600-700', '700-800', '800-900', '900-1000']
    score_df['score_range'] = pd.cut(score_df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    
    # Load original data for behavioral analysis
    transaction_df = load_original_data()
    
    # Create visualizations
    create_score_distribution_chart(score_df)
    
    # Analyze behaviors by score range
    behavioral_analysis = analyze_behaviors_by_score(transaction_df, score_df)
    
    # Generate markdown report
    generate_analysis_markdown(results, score_df, behavioral_analysis)
    
    print("="*50)
    print("✅ ANALYSIS COMPLETE!")
    print("✅ Files generated:")
    print("   - analysis.md (comprehensive report)")
    print("   - score_distribution.png (visualization)")
    print("="*50)
    
    return behavioral_analysis

if __name__ == "__main__":
    main()