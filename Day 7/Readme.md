# Geopolitical Events, Sanctions & Oil Market EDA

## 📌 Project Overview
This project explores the relationship between **geopolitical events**, **sanctions**, and **Iran’s oil exports** using exploratory data analysis (EDA).  
We analyze how wars, sanctions, and diplomatic shifts impact oil prices, risk indicators, and Iran’s export revenues.

---

## 📂 Datasets
1. **geopolitical_events.csv**  
   - Records of global and Middle East events (wars, sanctions, political shifts, market shocks).  
   - Key fields: `event_name`, `event_type`, `region`, `iran_involved`, `price_shock_pct`, `duration_days`, `severity`.

2. **iran_oil_exports.csv**  
   - Annual oil exports by destination country.  
   - Includes sanctions status and estimated revenues.

3. **risk_indicators.csv**  
   - Monthly risk metrics: geopolitical risk index, Iran-specific risk basis points, VIX, USD index, gold prices, OPEC spare capacity, Brent risk premium.

4. **sanctions_timeline.csv**  
   - Timeline of sanctions imposed on Iran.  
   - Includes GDP impact, oil/financial sector effects, and cumulative pressure score.

---

## ⚙️ Methodology
### Step 1: Data Inspection
- Load datasets with `pandas`.
- Check structure, data types, and missing values.

### Step 2: Descriptive Statistics
- Compute mean, median, min, max, and standard deviation for numeric fields.
- Summarize categorical distributions (event types, sanctions categories).

### Step 3: Visual Analysis
- **Histograms**: Distribution of oil price shocks.  
- **Line Charts**: Oil exports vs sanctions pressure score.  
- **Scatter Plots**: Iran-involved events vs Brent risk premium.  
- **Heatmaps**: Correlation between sanctions, risk indicators, and exports.  
- **Boxplots**: Event duration vs severity.

### Step 4: Extreme Events
- Identify outliers (e.g., 1990 Kuwait invasion, 2020 WTI crash).
- Link extreme shocks to geopolitical/military events.

---

## 📊 Key Insights
- Heavy sanctions (2012–2015, 2019–2021) slashed Iran’s exports by **60–70%**.  
- Relief periods (2015 JCPOA, 2016 sanctions lift) restored exports temporarily.  
- Military escalations (e.g., Abqaiq attack 2019, Iran-Israel 2024–2025) coincided with **Brent premiums > $5/bbl**.  
- Oil markets are resilient to small shocks but fragile to rare, extreme events.

---
