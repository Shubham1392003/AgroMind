
# AgroMind: Sustainable Agriculture with AI for Crop Yield Prediction 🌿🌾

## Hey there, Future Farmer! 👋
Welcome to **AgroMind** – your intelligent companion for smarter, more sustainable agriculture! This project harnesses the power of Artificial Intelligence to help farmers accurately predict crop yields. In a world facing climate change and increasing food demands, having a "crystal ball" for your fields isn't just cool – it's essential! 🔮

### 🌟 Our Mission
To empower farmers with data-driven insights, helping them optimize resources, minimize waste, and cultivate a greener future.

---

## 🚀 The Big Idea: AI in Action
Imagine a rural Indian farmer armed with AI-powered predictions for wheat and rice. By analyzing historical data, soil conditions, and real-time weather patterns, AgroMind forecasts yields and offers smart irrigation tips.

**Impact**:
- Up to 15% reduction in water usage
- Around 10% boost in crop yield

---

## 📊 What Data Are We Using?
The dataset: `clean_crop_data.csv` contains:

- `Area` 🗺️
- `Item` (crop type) 🌽
- `Year` 🗓️
- `hg/ha_yield` (crop yield)
- `average_rain_fall_mm_per_year` 🌧️
- `pesticides_tonnes` 🧪
- `avg_temp` 🌡️

🔍 **Future Improvement**: Add `Soil_Quality` data for improved predictions.

---

## 🧠 How AgroMind Works (`model_trainer.py`)
### 1. Data Loading
- Reads `clean_crop_data.csv`

### 2. Data Preparation
- **One-Hot Encoding** for crops
- Handles missing values
- Splits into 80% training and 20% testing data

### 3. Model Training
- **Random Forest Regressor**: Great for complex, non-linear relationships
- **RandomizedSearchCV**: Tunes hyperparameters for optimal performance

### 4. Evaluation Metrics
- **Mean Squared Error (MSE)** and **Root MSE (RMSE)**
- **R² Score**: Closer to 1 = better accuracy
- **Feature Importance** insights

### 5. Prediction and Visualization
- User input: rainfall, temp, pesticide, crop type → yield prediction 🔮
- Generates insightful graphs for evaluation and data exploration

---

## 🚀 Ready to Run AgroMind?
```bash
git clone https://github.com/Shubham1392003/AgroMind.git  # Replace if needed
cd AgroMind
```

Place your `clean_crop_data.csv` in the same directory.

Install required packages:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```

Launch:
```bash
python model_trainer.py
```

---

## 🌍 Why AgroMind Matters
- Boosts food production and reduces waste
- Builds climate resilience
- Empowers small-scale farmers with AI insights
- Promotes sustainable farming practices

---

## 💡 What’s Next for AgroMind?
- A web application 🌐
- Richer datasets: soil nutrients, sunlight hours, etc.
- Advanced AI: Neural networks 🤖
- Time-series forecasting 📈
- Satellite and geo-mapping integration 🛰️

---

## 🤝 Let’s Grow a Better Future!
Explore, contribute, or share your ideas. Together, we can cultivate a smarter and greener agricultural world! 🌱
