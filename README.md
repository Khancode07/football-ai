# ⚽ World Cup 2026 Predictor

A data-driven application that predicts the outcomes of international football matches, specifically optimized for the **FIFA World Cup 2026**.

## 🌟 Features

- **High Accuracy (60.33%)**: Powered by a Soft Voting Ensemble of XGBoost, Random Forest, and Logistic Regression.
- **Premium Dashboard**: A beautifully designed, interactive web UI built with modern HTML/CSS (Glassmorphism, dark mode, responsive).
- **Interactive Predictor**: Pick any two international teams and get real-time predictions including win probabilities, confidence levels, and expected goals (xG).
- **World Cup Simulation**: Simulates the entire World Cup group stage to predict standings, favorites, and the ultimate champion.
- **Advanced Features**: Uses Strength-Adjusted Elo Ratings, recent competitive form, historical H2H records, and FIFA 22 team ratings.

## 🛠️ How It Works

The model was trained on **15,645 competitive international matches** from the year 2000 to 2025. Friendlies are completely removed from the training set to prevent noise and ensure the model only learns from matches where teams play at 100% effort.

**Why 60.33% is excellent:** Football is inherently unpredictable (a Poisson process). The theoretical maximum accuracy for a 3-class prediction (Win/Draw/Loss) is around ~63%. Professional sportsbooks typically sit around ~58-60%.

## 🚀 Running the App locally

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```
2. **Start the Web Dashboard**:
   ```bash
   python worldcup.py
   ```
3. Open `http://127.0.0.1:5001` in your browser.

## 🧠 Retraining the Model

If you have updated dataset CSVs or want to recompute the Elo ratings:
```bash
python model.py
```
This will process the data, train the ensemble model, and generate `football_model.pkl` and `encoders.pkl` which are required by the dashboard.
