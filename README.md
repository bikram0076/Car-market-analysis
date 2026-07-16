# Car Market Intelligence Dashboard

A professional Streamlit dashboard built on top of your `carmarket.ipynb` EDA, covering the
**Cleaned_Car_Market_Dataset.csv**. Pure exploratory data analysis — no machine learning.

## Pages
1. **Home** — hero overview, KPIs, quick-look charts
2. **Overview** — data quality, dtypes, categorical distributions
3. **Price** — price distributions, drivers of price, trends over time
4. **Car Body** — body style popularity vs. price
5. **Driven** — kilometres, drivetrain, fuel-type usage patterns
6. **Statistical** — correlation heatmap, outlier boxplots, pairplot, brand x fuel-type heatmap
7. **Models** — top/most-listed car models, a side-by-side model comparison tool, and a value-for-money analysis

## Bonus features added
- Global sidebar filters (Brand, Body Type, Fuel Type, Condition, Year range, Price range) that apply live across every page
- Downloadable filtered CSV
- Auto-generated "Insight" callouts on each page summarizing the data shown
- **Model Comparison tab** — pick any set of models and compare price, kilometres, and year trends side-by-side
- **Value Analysis tab** — a transparent, formula-based (non-ML) value score combining price, kilometres, and age to surface the best-value models, plus a brand efficiency ranking (price per 1,000 km driven)

## Setup
```bash
pip install -r requirements.txt
```

## Run
Make sure `Cleaned_Car_Market_Dataset.csv` sits in the same folder as `app.py` (already included here), then:
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## Notes
- The custom theme lives in `.streamlit/config.toml` — feel free to tweak `primaryColor` etc.
- If the CSV is missing at runtime, the app will show a file uploader so you can supply it manually.
- The Value Score is a simple weighted average (50% price, 30% kilometres, 20% recency), not a predictive model — it's fully transparent and easy to explain to stakeholders.

