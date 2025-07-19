# DeFi Credit Scoring Model

A machine learning-based credit scoring system that assigns scores (0-1000) to cryptocurrency wallets based on their Aave V2 transaction behavior.

## 🎯 Overview

This project analyzes DeFi transaction patterns to identify reliable vs. risky wallet behavior. Higher scores indicate trustworthy users; lower scores flag potential bots or exploitative behavior.

**Key Features:**
- One-step execution from any transaction JSON file
- Auto-detects different data formats and field names
- Transparent, rule-based scoring methodology
- Comprehensive behavioral analysis and visualizations

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run scoring (one command!)
python score_wallets.py user-transactions.json

# 3. Generate analysis (optional)
python generate_analysis.py
```

**That's it!** Your wallet scores are ready in `wallet_scores.json`.

## 📊 Sample Results

```json
{
  "total_wallets_scored": 15420,
  "average_score": 456.78,
  "wallet_scores": {
    "0x1234...": 750,  // Excellent user
    "0x5678...": 320,  // High-risk user
    "0x9abc...": 890   // Perfect user
  }
}
```

## 🧠 How It Works

### Scoring Methodology

The model evaluates 5 key behavioral factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Repayment Behavior** | 30% | Repay/borrow ratio (most important) |
| **Transaction Volume** | 25% | Total protocol engagement |
| **Risk Indicators** | 20% | Liquidation events, risky patterns |
| **Activity Consistency** | 15% | Regular usage over time |
| **Portfolio Diversity** | 10% | Number of different assets |

### Score Ranges

- **800-1000**: Excellent - Perfect repayment, no liquidations
- **600-799**: Good - Reliable with minor issues  
- **400-599**: Fair - Moderate risk, mixed patterns
- **200-399**: Poor - High risk, concerning behavior
- **0-199**: Critical - Very risky, potential bot activity

### Algorithm

```python
# Simplified scoring logic
score = (
    min(250, transactions * 5) +           # Volume (0-250)
    repayment_rate * 300 +                 # Repayment (0-300) 
    max(0, 200 - liquidations * 200) +     # Risk (0-200)
    consistency_ratio * 150 +              # Consistency (0-150)
    min(100, unique_assets * 20) +         # Diversity (0-100)
    responsibility_bonus                   # +50 if perfect
)

# Bot penalty: score *= 0.5 if >100 transactions/day
```

## 📁 Input Data Format

The script automatically handles different JSON structures:

**Format 1: Simple List**
```json
[
  {
    "user": "0x123...",
    "action": "deposit", 
    "amount": "1000000000000000000",
    "timestamp": 1640995200
  }
]
```

**Format 2: Nested Object**
```json
{
  "transactions": [
    {
      "wallet": "0x123...",
      "type": "borrow",
      "value": "500000000000000000"
    }
  ]
}
```

### Required Fields
- **Wallet ID**: `user`, `wallet`, `address`, `from`, or `account`
- **Action**: `action`, `type`, or `event` (deposit, borrow, repay, liquidationcall)

### Optional Fields  
- `timestamp`, `blockTimestamp` (for time-based analysis)
- `reserve`, `asset`, `token` (for portfolio diversity)
- `amount`, `value` (for transaction size analysis)

## 📊 Output Files

| File | Description |
|------|-------------|
| `wallet_scores.json` | Complete scoring results with metadata |
| `analysis.md` | Detailed behavioral analysis by score range |
| `score_distribution.png` | Score distribution charts |

## 🔧 Advanced Usage

### Custom Output Location
```bash
python score_wallets.py data.json -o custom/path/scores.json
```

### Batch Processing
```bash
# Score multiple files
for file in *.json; do
    python score_wallets.py "$file" -o "scores_${file}"
done
```

### API Integration
```python
from score_wallets import DeFiCreditScorer

scorer = DeFiCreditScorer()
df = scorer.load_transactions('data.json')
scores = scorer.calculate_scores(scorer.engineer_features(df))
```

## 📈 Model Performance

### Validation Approach
- **Rule-based scoring**: Transparent and interpretable
- **Feature importance**: Repayment behavior weighted highest (30%)
- **Outlier detection**: Bot penalty for suspicious patterns
- **Score distribution**: Balanced across all risk categories

### Key Assumptions
1. Repayment rate is the strongest creditworthiness indicator
2. Liquidation events signal poor risk management  
3. Consistent activity indicates genuine user behavior
4. Portfolio diversity shows financial sophistication

## 🛠 Technical Architecture

```
JSON Input → Field Detection → Data Preprocessing → Feature Engineering → Scoring Algorithm → Results Output
```

### Core Components
- **Auto Field Detector**: Identifies wallet/action fields in any JSON structure
- **Feature Engineer**: Extracts 15+ behavioral metrics per wallet
- **Credit Scorer**: Applies weighted scoring algorithm
- **Analysis Generator**: Creates behavioral insights and visualizations

## 🔍 Example Analysis

### High-Risk Wallets (0-300)
- High liquidation rates (>10% of transactions)
- Poor repayment ratios (<0.5)
- Potential bot activity (>100 transactions/day)
- Single-asset focus with no diversification

### Excellent Wallets (800-1000)  
- Perfect repayment history (1.0+ repay/borrow ratio)
- Zero liquidation events
- Consistent long-term usage
- Diversified asset portfolios (5+ different tokens)

## 🚀 Future Enhancements

- [ ] Machine learning models (Random Forest, XGBoost)
- [ ] Multi-protocol support (Compound, MakerDAO)
- [ ] Real-time scoring API
- [ ] Market volatility adjustments
- [ ] Time-decay for historical events

## 📋 Requirements

- Python 3.7+
- pandas, numpy, matplotlib, seaborn
- ~87MB disk space for sample data

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- [Sample Data](https://drive.google.com/file/d/14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor/view?usp=sharing)
- [Aave V2 Protocol](https://aave.com)

---

*Built for the DeFi Credit Scoring Challenge. Assigns credit scores to 15K+ wallets based on Aave V2 transaction patterns.*
