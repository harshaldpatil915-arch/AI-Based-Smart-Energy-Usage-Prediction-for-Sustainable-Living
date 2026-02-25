# ⚡ AI-Based Smart Energy Usage Prediction — Streamlit App

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
The app will automatically open at → http://localhost:8501

---

## What the App Does

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, KPIs, model comparison chart |
| 📊 Data Explorer | Hourly usage, heatmaps, patterns by hour/day |
| 🤖 Model Comparison | Train all 3 models, compare MAE/RMSE, Actual vs Predicted chart |
| 🔮 24h Forecast | Recursive 24-hour forecast with peak & off-peak highlights |
| 🔍 Feature Importance | Permutation importance chart, correlation matrix |

## Notes
- The dataset is auto-downloaded from UCI ML Repository on first run
- All models are cached — no re-training on page reload
- Dataset: ~130 MB download (only once)
