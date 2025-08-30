# app.py
import io
import json
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Environmental Load Optimization Agent", layout="wide")

# ----------------------
# Helpers / Config
# ----------------------
DEFAULT_CONFIG = {
    "grid_kgco2_per_kwh": 0.475,
    "fuel_kgco2_per_kwh": {"natural_gas": 0.202, "diesel": 0.27, "coal": 0.34},
    "energy_cost_per_kwh": {"grid": 0.12, "natural_gas": 0.045, "diesel": 0.20, "renewable": 0.06},
    "carbon_price_usd_per_ton": 65.0,
    "scope3_mult_of_s1s2": 0.25,
    "alert_energy_sigma": 3.0,
    "alert_intensity_quantile": 0.75
}

st.sidebar.title("Settings")
config = DEFAULT_CONFIG.copy()
# Allow user to tune grid factor quickly
config["grid_kgco2_per_kwh"] = st.sidebar.number_input("Grid kgCO2 per kWh", value=config["grid_kgco2_per_kwh"], step=0.01, format="%.3f")
config["carbon_price_usd_per_ton"] = st.sidebar.number_input("Carbon price $/tCO2", value=config["carbon_price_usd_per_ton"], step=1.0)
st.sidebar.markdown("---")
st.sidebar.caption("Upload your CSV in the main panel. The app auto-detects columns and adapts.")

# Normalization utilities
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("(", "", regex=False)
                  .str.replace(")", "", regex=False)
                  .str.replace("/", "_")
                  .str.replace("-", "_")
                  .str.lower()
    )
    return df

def first_available(df: pd.DataFrame, *names) -> Optional[str]:
    cols = set(df.columns)
    for n in names:
        if n and n.lower() in cols:
            return n.lower()
    return None

@st.cache_data
def load_and_prepare(uploaded_bytes) -> Dict:
    """Load CSV bytes and prepare unified columns tailored to your dataset."""
    raw = pd.read_csv(io.BytesIO(uploaded_bytes))
    df = normalize_columns(raw)

    # detect date
    date_candidates = ["timestamp", "date", "datetime", "time", "recorded_date"]
    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # energy columns
    energy_x = first_available(df, "energy_consumption_kwh_x", "energy_consumption_kwh", "energy_consumption_kwh_y", "energy_kwh")
    energy_y = first_available(df, "energy_consumption_kwh_y")
    if energy_x and energy_y:
        df["energy_kwh"] = df[[energy_x, energy_y]].mean(axis=1)
    elif energy_x:
        df["energy_kwh"] = df[energy_x]
    elif energy_y:
        df["energy_kwh"] = df[energy_y]
    else:
        # if none found, raise; but we can synthesize later
        df["energy_kwh"] = df.get("energy", np.nan)

    # carbon columns
    carbon_total = first_available(df, "carbon_footprint_total", "carbon_footprint", "carbon_emissions_kgco2")
    carbon_x = first_available(df, "carbon_emissions_kgco2_x")
    carbon_y = first_available(df, "carbon_emissions_kgco2_y")
    if carbon_total:
        df["carbon_kg"] = df[carbon_total]
    elif carbon_x and carbon_y:
        df["carbon_kg"] = df[[carbon_x, carbon_y]].mean(axis=1)
    elif carbon_x:
        df["carbon_kg"] = df[carbon_x]
    elif carbon_y:
        df["carbon_kg"] = df[carbon_y]
    else:
        # fallback estimate
        df["carbon_kg"] = df["energy_kwh"].fillna(0) * DEFAULT_CONFIG["grid_kgco2_per_kwh"]

    # production units
    prod_x = first_available(df, "production_output_units_x", "production_output_units_y", "production_output_units")
    if prod_x:
        df["production_units"] = df[prod_x]
    else:
        # synthesize: assume 4 kWh per unit average
        df["production_units"] = (df["energy_kwh"].fillna(0) / 4.0).round(2)

    # renewable pct
    ren = first_available(df, "renewable_energy_%", "renewableenergy", "renewable_pct", "renewable_share_percent")
    if ren:
        df["renewable_pct"] = df[ren]
    else:
        df["renewable_pct"] = df.get("renewableenergy", np.nan).fillna(np.random.uniform(10, 40, size=len(df)))

    # equipment efficiency
    eff = first_available(df, "equipment_efficiency_%", "equipment_efficiency")
    if eff:
        df["equipment_efficiency_%"] = df[eff]
    else:
        df["equipment_efficiency_%"] = np.random.uniform(70, 95, size=len(df))

    # fill numeric NaNs with median for stability
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    # derived metrics
    df["carbon_intensity_kg_per_unit"] = (df["carbon_kg"] / df["production_units"]).replace([np.inf, -np.inf], np.nan).fillna(0)

    # scope defaults if not present
    if "scope1_emissions" not in df.columns:
        df["scope1_emissions"] = 0.0
    if "scope2_emissions" not in df.columns:
        df["scope2_emissions"] = df["energy_kwh"] * DEFAULT_CONFIG["grid_kgco2_per_kwh"]  # coarse
    if "scope3_emissions" not in df.columns:
        df["scope3_emissions"] = (df["scope1_emissions"] + df["scope2_emissions"]) * DEFAULT_CONFIG["scope3_mult_of_s1s2"]

    df["total_lifecycle_kgco2e"] = df["scope1_emissions"] + df["scope2_emissions"] + df["scope3_emissions"]

    return {"df": df, "date_col": date_col}

# ----------------------
# App layout
# ----------------------
st.title("Environmental Load Optimization Agent — Streamlit Dashboard")
st.markdown("Upload your dataset (CSV). The app will adapt to the column names and build monitoring, prediction, optimization and recommendations.")

# Upload
uploaded_file = st.file_uploader("Upload final_carbon_project_dataset.csv", type=["csv"], accept_multiple_files=False)
if uploaded_file is None:
    st.info("Upload your CSV to continue. Use the dataset you've been working with.")
    st.stop()

with st.spinner("Loading and preparing dataset..."):
    payload = load_and_prepare(uploaded_file.read())
    df = payload["df"]
    date_col = payload["date_col"]

# Top KPIs
total_energy = df["energy_kwh"].sum()
total_co2 = df["carbon_kg"].sum()
avg_intensity = df["carbon_intensity_kg_per_unit"].mean()
med_renew = df["renewable_pct"].median()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total energy (kWh)", f"{total_energy:,.0f}")
k2.metric("Total CO₂ (kg)", f"{total_co2:,.0f}")
k3.metric("Avg intensity (kg CO₂/unit)", f"{avg_intensity:.2f}")
k4.metric("Median renewable %", f"{med_renew:.1f}%")

# Tabs: Overview, Monitoring, Optimization, Model, Recs, Reports
tabs = st.tabs(["Overview", "Monitoring", "Optimization", "Model", "Recommendations", "Reports"])

# ----------------------
# Overview Tab
# ----------------------
with tabs[0]:
    st.header("Overview & EDA")
    st.markdown("Time series, distributions and correlations")

    # Time series plots
    if date_col:
        tmp = df.sort_values(date_col)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tmp[date_col], y=tmp["energy_kwh"], name="Energy (kWh)"))
        fig.add_trace(go.Scatter(x=tmp[date_col], y=tmp["carbon_kg"], name="Carbon (kgCO₂)"))
        fig.update_layout(title="Energy & Carbon over time", xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True)

    # Renewable distribution
    fig2 = px.histogram(df, x="renewable_pct", nbins=20, title="Renewable % distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # Energy vs intensity scatter
    fig3 = px.scatter(df, x="energy_kwh", y="carbon_intensity_kg_per_unit", color=df.get("industry_sector", None),
                      hover_data=["production_units"], title="Energy kWh vs Carbon intensity")
    st.plotly_chart(fig3, use_container_width=True)

    # Correlation heatmap
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric correlation matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# ----------------------
# Monitoring Tab
# ----------------------
with tabs[1]:
    st.header("Monitoring & Alerts")
    st.markdown("Live-like monitoring visualizations and alert detection (static dataset simulated as stream).")

    # Energy time series split x/y if present
    if "energy_kwh" in df.columns:
        fig_e = px.line(df.sort_values(date_col) if date_col else df, x=date_col or df.index, y="energy_kwh", title="Energy Consumption (kWh) over time")
        st.plotly_chart(fig_e, use_container_width=True)

    # Carbon stack / footprint
    if "total_lifecycle_kgco2e" in df.columns and date_col:
        fig_c = px.line(df.sort_values(date_col), x=date_col, y=["scope1_emissions", "scope2_emissions", "scope3_emissions"],
                        labels={"value":"kgCO₂", "variable":"Scope"}, title="Scopes 1/2/3 over time")
        st.plotly_chart(fig_c, use_container_width=True)

    # Machine performance
    tcols = [c for c in ["machine_temperature_c_x", "machine_temperature_c_y", "equipment_efficiency_%", "operating_hours"] if c in df.columns]
    if tcols:
        st.subheader("Machine performance (sample)")
        st.dataframe(df[tcols + ([date_col] if date_col else [])].head(200))

    # Alerts detection
    st.subheader("Alerts")
    energy_thr = df["energy_kwh"].mean() + DEFAULT_CONFIG["alert_energy_sigma"] * df["energy_kwh"].std()
    intensity_thr = df["carbon_intensity_kg_per_unit"].quantile(DEFAULT_CONFIG["alert_intensity_quantile"])
    alerts = []
    for i, row in df.iterrows():
        t = row[date_col] if date_col else i
        if pd.notna(row["energy_kwh"]) and row["energy_kwh"] > energy_thr:
            alerts.append({"time": t, "type": "ENERGY_SPIKE", "value": row["energy_kwh"], "threshold": energy_thr})
        if pd.notna(row["carbon_intensity_kg_per_unit"]) and row["carbon_intensity_kg_per_unit"] > intensity_thr:
            alerts.append({"time": t, "type": "HIGH_INTENSITY", "value": row["carbon_intensity_kg_per_unit"], "threshold": intensity_thr})
        if "maintenance_status" in df.columns and isinstance(row.get("maintenance_status"), str) and row.get("maintenance_status").lower()=="required":
            alerts.append({"time": t, "type": "MAINTENANCE_REQUIRED", "value": row.get("maintenance_status")})

    alerts_df = pd.DataFrame(alerts)
    st.write(f"Detected {len(alerts_df)} alerts.")
    if not alerts_df.empty:
        st.dataframe(alerts_df.head(200))
        st.download_button("Download alerts CSV", data=alerts_df.to_csv(index=False), file_name="alerts.csv", mime="text/csv")
    else:
        st.info("No alerts detected with current thresholds.")

# ----------------------
# Optimization Tab
# ----------------------
with tabs[2]:
    st.header("Optimization Engine")
    st.markdown("Minimize cost + λ × carbon price subject to demand and renewable floor.")

    st.sidebar.subheader("Optimization controls")
    lambda_weight = st.sidebar.slider("Carbon weight (λ)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    renewable_floor = st.sidebar.slider("Min renewable share (%)", min_value=0, max_value=100, value=0, step=5)
    # Demand selection: user may pick a date row or use median
    if date_col:
        dsel = st.selectbox("Use demand from (select a date) or choose 'median' ", options=["median"] + list(df[date_col].dt.strftime("%Y-%m-%d").unique()))
        if dsel == "median":
            demand = float(df.sort_values(date_col)["energy_kwh"].tail(7).median())
        else:
            demand = float(df[df[date_col].dt.strftime("%Y-%m-%d")==dsel]["energy_kwh"].median())
    else:
        demand = float(df["energy_kwh"].median())

    st.write(f"Using demand (kWh): **{demand:.0f}**")

    @dataclass
    class OptInputs:
        demand_kwh: float
        lambda_carbon: float
        renewable_floor_pct: float

    def run_optimization(inputs: OptInputs):
        sources = ["grid", "natural_gas", "diesel", "renewable"]
        cost = config["energy_cost_per_kwh"]
        ef = {
            "grid": config["grid_kgco2_per_kwh"],
            "natural_gas": config["fuel_kgco2_per_kwh"]["natural_gas"],
            "diesel": config["fuel_kgco2_per_kwh"]["diesel"],
            "renewable": 0.0
        }
        x = pulp.LpVariable.dicts("kwh", sources, lowBound=0)
        prob = pulp.LpProblem("mix", pulp.LpMinimize)
        carbon_price_per_kg = config["carbon_price_usd_per_ton"] / 1000.0
        prob += pulp.lpSum([x[s]*cost[s] for s in sources]) + inputs.lambda_carbon * pulp.lpSum([x[s]*ef[s]*carbon_price_per_kg for s in sources])
        prob += pulp.lpSum([x[s] for s in sources]) >= inputs.demand_kwh
        if inputs.renewable_floor_pct > 0:
            prob += x["renewable"] >= inputs.demand_kwh * (inputs.renewable_floor_pct/100.0)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        sol = {s: max(0.0, pulp.value(x[s])) for s in sources}
        cost_usd = sum(sol[s] * cost[s] for s in sources)
        total_kg = sum(sol[s] * ef[s] for s in sources)
        return sol, cost_usd, total_kg

    if st.button("Run optimization"):
        inputs = OptInputs(demand_kwh=demand, lambda_carbon=lambda_weight, renewable_floor_pct=renewable_floor)
        sol, cost_usd, total_kg = run_optimization(inputs)
        sol_df = pd.DataFrame([{"source":k, "kWh":v} for k,v in sol.items()])
        st.subheader("Optimization result (kWh per source)")
        st.dataframe(sol_df)
        st.metric("Total cost (USD)", f"{cost_usd:,.2f}")
        st.metric("Total emissions (kg CO₂)", f"{total_kg:,.2f}")
        # chart
        fig = px.pie(sol_df, names="source", values="kWh", title="Resulting energy mix")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download optimization result CSV", data=sol_df.to_csv(index=False), file_name="optimization_result.csv", mime="text/csv")

# ----------------------
# Model Tab
# ----------------------
with tabs[3]:
    st.header("Carbon Emissions Model (RandomForest)")
    st.markdown("Train a model to predict carbon_kg from energy, renewable share, equipment efficiency, production.")

    features = [c for c in ["energy_kwh", "renewable_pct", "equipment_efficiency_%", "production_units", "ambienttemperature"] if c in df.columns]
    st.write("Candidate features:", features)
    test_size = st.slider("Test size (%)", min_value=10, max_value=40, value=25, step=5)
    if st.button("Train model"):
        X = df[features].fillna(0)
        y = df["carbon_kg"].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        st.success(f"Trained. R² = {r2:.3f}, MAE = {mae:.2f} kg CO₂")
        # feature importance
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.bar_chart(fi)
        joblib.dump(model, "carbon_model_rf.pkl")
        st.download_button("Download trained model", data=open("carbon_model_rf.pkl","rb"), file_name="carbon_model_rf.pkl")
    else:
        st.info("Train model to get predictive insights.")

# ----------------------
# Recommendations Tab
# ----------------------
with tabs[4]:
    st.header("Recommendations")
    st.markdown("Actionable suggestions derived from data and model-driven insights.")

    recs = []
    median_ren = df["renewable_pct"].median()
    if median_ren < 30:
        est = df["energy_kwh"].median() * 0.20 * config["grid_kgco2_per_kwh"]
        recs.append({"topic":"Increase renewable supply", "reason":f"Median renewable share {median_ren:.1f}%. Recommend +20pp", "est_kg_saved": est})
    median_eff = df["equipment_efficiency_%"].median()
    if median_eff < 85:
        est2 = df["energy_kwh"].median() * 0.05 * config["grid_kgco2_per_kwh"]
        recs.append({"topic":"Improve equipment efficiency", "reason":f"Median efficiency {median_eff:.1f}%. Recommend maintenance/controls to +5%", "est_kg_saved": est2})
    if "maintenance_status" in df.columns:
        cnt_req = (df["maintenance_status"].astype(str).str.lower()=="required").sum()
        if cnt_req>0:
            recs.append({"topic":"Address maintenance", "reason":f"{cnt_req} records flagged required", "est_kg_saved": None})
    rec_df = pd.DataFrame(recs)
    if not rec_df.empty:
        st.dataframe(rec_df)
        st.download_button("Download recommendations CSV", data=rec_df.to_csv(index=False), file_name="recommendations.csv")
    else:
        st.info("No recommendations generated (dataset already good or conditions not met).")

# ----------------------
# Reports Tab
# ----------------------
with tabs[5]:
    st.header("Export / Reports")
    st.markdown("Download enhanced dataset, alerts, optimization outputs and a simple HTML report.")
    st.download_button("Download enhanced dataset (CSV)", data=df.to_csv(index=False), file_name="enhanced_final_carbon_dataset.csv", mime="text/csv")
    if alerts_df is not None and not alerts_df.empty:
        st.download_button("Download alerts (CSV)", data=alerts_df.to_csv(index=False), file_name="alerts.csv", mime="text/csv")
    st.markdown("Simple HTML report (summary)")
    summary = {
        "total_energy_kwh": int(total_energy),
        "total_co2_kg": int(total_co2),
        "avg_intensity": float(avg_intensity)
    }
    html = f"<h2>Environmental Summary</h2><pre>{json.dumps(summary, indent=2)}</pre>"
    st.download_button("Download simple HTML report", data=html, file_name="env_report.html", mime="text/html")
