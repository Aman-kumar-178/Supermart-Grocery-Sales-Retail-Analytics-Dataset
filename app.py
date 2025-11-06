# ============================================================
# 🧠 Supermart Grocery Sales — Advanced Business Intelligence Platform (v8.6)
# ============================================================
# 🚀 Includes:
# ✅ Automated feature engineering + modeling + versioning
# ✅ Profit Growth & Sales Target Simulator
# ✅ Smart Sales Strategy & Market Insights
# ✅ Dynamic recommendations + Download Trained Model
# ✅ Final Conclusion & Action Plan
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 🧩 Helper Functions
# ============================================================

def preprocess_data(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for c in ["sales", "profit", "discount", "quantity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for col in df.columns:
        if "date" in col or "order" in col:
            try:
                df[col] = pd.to_datetime(df[col])
                df["year"] = df[col].dt.year
                df["month"] = df[col].dt.month
                df["weekday"] = df[col].dt.day_name()
                break
            except Exception:
                pass

    if "sales" in df.columns and "profit" in df.columns:
        df["profit_ratio"] = np.where(df["sales"] > 0, df["profit"] / df["sales"], 0)
    if "discount" in df.columns and "sales" in df.columns:
        df["discount_pct"] = np.where(df["sales"] > 0, df["discount"] / df["sales"], 0)
    if "sales" in df.columns and "quantity" in df.columns:
        df["price_per_unit"] = np.where(df["quantity"] > 0, df["sales"] / df["quantity"], 0)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


def save_model(model, prefix="supermart_model"):
    os.makedirs("models", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"models/{prefix}_{ts}.pkl"
    joblib.dump(model, path)
    return path


def build_pipeline(numeric_cols, categorical_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    selector = SelectKBest(score_func=f_regression, k=min(20, len(numeric_cols)))
    return preprocessor, selector


def get_models():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=150, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

# ============================================================
# 🖥️ Streamlit App
# ============================================================

st.set_page_config(page_title="Supermart BI v8.6", layout="wide")
st.title("🧠 Supermart Grocery Sales — Business Intelligence Dashboard (v8.6)")
st.caption("Includes profit growth simulation, model download, and AI recommendations")

# ------------------------------------------------------------
# 📂 Upload Data
# ------------------------------------------------------------
file = st.file_uploader("📂 Upload your Supermart dataset (CSV)", type=["csv"])
if file is None:
    st.info("Please upload your dataset to get insights and predictions.")
    st.stop()

df = pd.read_csv(file)
df = preprocess_data(df)
st.success("✅ Dataset Loaded Successfully!")
st.dataframe(df.head())

# ------------------------------------------------------------
# 📊 KPI Overview
# ------------------------------------------------------------
st.header("📈 Business Overview & KPIs")

total_sales = df["sales"].sum() if "sales" in df.columns else 0
total_profit = df["profit"].sum() if "profit" in df.columns else 0
avg_profit_ratio = (total_profit / total_sales * 100) if total_sales > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
col2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
col3.metric("🧾 Records", len(df))
col4.metric("💹 Avg Profit Ratio", f"{avg_profit_ratio:.2f}%")

# ------------------------------------------------------------
# 📊 Smart EDA
# ------------------------------------------------------------
st.header("🔍 Automated Data Insights")

if "sales" in df.columns:
    st.subheader("📦 Sales Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["sales"], bins=40, color="lightblue")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    if "category" in df.columns:
        top_sales = df.groupby("category")["sales"].sum().sort_values(ascending=False).head(5)
        st.subheader("🏆 Top 5 Categories by Sales")
        st.bar_chart(top_sales)

if "profit_ratio" in df.columns:
    fig, ax = plt.subplots()
    ax.hist(df["profit_ratio"] * 100, bins=30, color="salmon")
    ax.set_xlabel("Profit Ratio (%)")
    st.pyplot(fig)

# ------------------------------------------------------------
# 🤖 Model Training
# ------------------------------------------------------------
st.header("🤖 Predictive Model Training")

target_col = st.selectbox("🎯 Select Target Variable", df.select_dtypes(include=[np.number]).columns)
num_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col).tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

X = df[num_cols + cat_cols]
y = df[target_col]

test_size = st.slider("Test Set Size (%)", 5, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

pre, sel = build_pipeline(num_cols, cat_cols)
models = get_models()

if st.button("🚀 Train Models"):
    results, fitted = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("sel", sel), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "R2": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "CV_Score": cross_val_score(pipe, X_train, y_train, cv=3).mean()
        }
        fitted[name] = pipe

    res_df = pd.DataFrame(results).T
    st.subheader("📊 Model Comparison Results")
    st.dataframe(res_df)

    best = res_df["R2"].idxmax()
    st.success(f"✅ Best Model: {best} (R²={res_df.loc[best, 'R2']:.3f})")

    model_path = save_model(fitted[best], f"supermart_{best}")
    st.info(f"💾 Model saved at: {model_path}")

    # 🧩 NEW: Download button for model
    with open(model_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Trained Model (.pkl)",
            data=f,
            file_name=os.path.basename(model_path),
            mime="application/octet-stream"
        )

# ------------------------------------------------------------
# 💹 Profit & Sales Growth Simulator
# ------------------------------------------------------------
st.header("💹 Profit Growth & Sales Strategy Recommendations")

if "sales" in df.columns and "profit" in df.columns:
    st.subheader("🎯 Profitability Analysis")
    target_ratio = st.slider("Set Target Profit Margin (%)", min_value=avg_profit_ratio, max_value=50.0, value=avg_profit_ratio + 5)

    needed_profit = (target_ratio / 100) * total_sales
    increase = needed_profit - total_profit
    required_sales = total_sales * (target_ratio / avg_profit_ratio) if avg_profit_ratio > 0 else total_sales

    st.markdown(f"""
    - Current Profit: ₹{total_profit:,.0f}
    - Target Profit: ₹{needed_profit:,.0f}
    - Required Increase: ₹{increase:,.0f}
    - Needed Sales Volume: ₹{required_sales:,.0f}
    """)

    st.subheader("💡 Growth Recommendations")
    st.markdown("""
    **📊 Sales Strategies**
    - Launch targeted offers in top 3 cities or regions.
    - Increase marketing during low-sales months.
    - Use bundle sales to increase basket size.
    - Cross-sell complementary products.

    **💸 Profit Optimization**
    - Push high-margin categories.
    - Negotiate vendor deals and lower procurement cost.
    - Keep discounts under 15%.
    - Bundle low-profit SKUs with top sellers.

    **👥 Customer Insights**
    - Reward loyal customers.
    - Offer personalized recommendations.
    - Use churn prediction to re-engage lost users.
    """)

# ------------------------------------------------------------
# 📊 Conclusion & Action Plan
# ------------------------------------------------------------
st.header("📊 Final Insights & Action Plan")

if "sales" in df.columns and "profit" in df.columns:
    sales_growth = "increasing" if df["sales"].diff().mean() > 0 else "declining"
    profit_trend = "healthy" if avg_profit_ratio > 20 else "moderate" if avg_profit_ratio > 10 else "low"
    discount_effect = "⚠️ Discounts may reduce profits." if df["discount"].mean() > 0.15 * df["sales"].mean() else "✅ Discounts are reasonable."

    st.markdown(f"""
    - 🛍️ **Sales Trend:** {sales_growth.capitalize()}
    - 💸 **Profit Margin:** {profit_trend.capitalize()} ({avg_profit_ratio:.2f}%)
    - 💬 **Discount Impact:** {discount_effect}
    """)

    st.subheader("🧠 Strategic Recommendations")
    st.markdown("""
    **Short-Term (1–3 Months)**
    - Promote low-performing categories.
    - Focus ads on key profitable regions.

    **Mid-Term (3–6 Months)**
    - Launch loyalty programs and dynamic pricing.
    - Optimize stock based on trends.

    **Long-Term (6–12 Months)**
    - Automate demand prediction using trained models.
    - Expand to high-demand cities.
    """)

    st.success("✅ Implement these insights to maximize sales and profit sustainably!")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("Supermart BI v8.6 — Growth Intelligence Dashboard © 2025")
