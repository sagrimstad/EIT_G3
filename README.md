# Day-Ahead Energy Consumption Forecasting

Day-ahead hourly electricity consumption forecasting for five Norwegian price areas (Bergen, Oslo, Stavanger, Tromsø, Trondheim), under a 5-day data lag constraint. EiT (TDT4861) collaboration with Aneo.

## Notebooks

- `lgbm_pipeline_minimal.ipynb` — end-to-end LightGBM training and test-window prediction
- `eda.ipynb` — exploratory data analysis
- `helsingfors_baseline.ipynb` — baseline analysis demonstrating Helsingfors is trivially predictable and was excluded from modeling

## Data

Place `consumption_temp.csv` in `data/`. The CSV is gitignored.
