"""
Car Market Intelligence Dashboard
==================================
A professional Streamlit dashboard for exploring the Cleaned Car Market Dataset.
Pure exploratory data analysis — no machine learning.

Pages: Home | Overview | Price | Car Body | Driven | Statistical | Models

Run with:
    streamlit run app.py

Author: Built for Bikramjit (O7 Services capstone project)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Car Market Intelligence Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#D62828"       # accent red
DARK = "#1B2430"          # near-black text
NAVY = "#003049"          # deep navy
GOLD = "#F77F00"          # amber accent
GREY = "#6C757D"
PALETTE = ["#003049", "#D62828", "#F77F00", "#FCBF49", "#669BBC", "#4B3F72", "#2A9D8F"]

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D9D9D9",
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlecolor": NAVY,
    "font.size": 10.5,
    "grid.color": "#ECECEC",
})

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: #EFF3F6; }}
    .main {{ background-color: #FAFBFC; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    h1, h2, h3 {{ color: {NAVY}; font-family: 'Segoe UI', sans-serif; }}

    .kpi-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f4f6f8 100%);
        border: 1px solid #E7EAEE;
        border-left: 5px solid {PRIMARY};
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}
    .kpi-value {{ font-size: 26px; font-weight: 700; color: {NAVY}; }}
    .kpi-label {{ font-size: 12.5px; color: {GREY}; text-transform: uppercase; letter-spacing: .5px; }}

    .section-banner {{
        background: linear-gradient(90deg, {NAVY} 0%, {PRIMARY} 100%);
        padding: 18px 24px;
        border-radius: 10px;
        color: white;
        margin-bottom: 18px;
    }}
    .section-banner h2 {{ color: white; margin: 0; }}
    .section-banner p {{ color: #E9EEF2; margin: 4px 0 0 0; font-size: 14px; }}

    .insight-box {{
        background: #FFF8F0;
        border-left: 4px solid {GOLD};
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: {DARK};
        margin: 10px 0;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def banner(title: str, subtitle: str = ""):
    st.markdown(
        f"""<div class="section-banner"><h2>{title}</h2><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def insight(text: str):
    st.markdown(f"""<div class="insight-box">💡 <b>Insight:</b> {text}</div>""", unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(
        f"""<div class="kpi-card"><div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div></div>""",
        unsafe_allow_html=True,
    )


def fmt_money(x):
    return f"${x:,.0f}"


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def style_plotly(fig, height=500):
    """Apply a consistent, branded theme to Plotly figures (sunburst / 3D scatter)."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=DARK, family="Segoe UI, sans-serif"),
        title_font=dict(color=NAVY, size=15, family="Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=55, b=10),
        height=height,
        legend=dict(bgcolor="rgba(255,255,255,0.6)"),
    )
    return fig


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
DATA_PATH = "Cleaned_Car_Market_Dataset.csv"


@st.cache_data(show_spinner="Loading car market data...")
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Numeric extraction for fuel consumption, e.g. "8.7 L / 100 km"
    if "FuelConsumption" in df.columns:
        df["FuelConsumption_num"] = (
            df["FuelConsumption"].str.extract(r"([\d.]+)").astype(float)
        )

    # Age of the car relative to the newest year present in the data
    if "Year" in df.columns:
        df["CarAge"] = df["Year"].max() - df["Year"]

    return df


try:
    raw_df = load_data(DATA_PATH)
    data_source_ok = True
except FileNotFoundError:
    data_source_ok = False
    raw_df = None

if not data_source_ok:
    st.error(
        "Could not find **Cleaned_Car_Market_Dataset.csv** next to app.py. "
        "Please upload it below to continue."
    )
    uploaded = st.file_uploader("Upload Cleaned_Car_Market_Dataset.csv", type="csv")
    if uploaded is not None:
        raw_df = load_data(uploaded)
        data_source_ok = True
    else:
        st.stop()

# ------------------------------------------------------------------
# SIDEBAR — NAVIGATION + GLOBAL FILTERS
# ------------------------------------------------------------------
st.sidebar.markdown(
    f"<h1 style='color:{NAVY}; font-size:26px;'>🚗 Car Market</h1>"
    f"<p style='color:{GREY}; margin-top:-10px;'>Intelligence Dashboard</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGES = ["🏠 Home", "📊 Overview", "💰 Price", "🚙 Car Body", "🛣️ Driven", "📈 Statistical", "🧩 Models"]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Global Filters")

brands = sorted(raw_df["Brand"].unique().tolist())
sel_brands = st.sidebar.multiselect("Brand", brands, default=[])

body_types = sorted(raw_df["BodyType"].unique().tolist())
sel_body = st.sidebar.multiselect("Body Type", body_types, default=[])

fuel_types = sorted(raw_df["FuelType"].unique().tolist())
sel_fuel = st.sidebar.multiselect("Fuel Type", fuel_types, default=[])

used_new = sorted(raw_df["UsedOrNew"].unique().tolist())
sel_condition = st.sidebar.multiselect("Condition", used_new, default=[])

year_min, year_max = int(raw_df["Year"].min()), int(raw_df["Year"].max())
sel_year = st.sidebar.slider("Manufacturing Year", year_min, year_max, (year_min, year_max))

price_min, price_max = int(raw_df["Price"].min()), int(raw_df["Price"].max())
sel_price = st.sidebar.slider("Price Range (AUD)", price_min, price_max, (price_min, price_max), step=500)

st.sidebar.markdown("---")
if st.sidebar.button("♻️ Reset Filters"):
    st.rerun()

# Apply filters
df = raw_df.copy()
if sel_brands:
    df = df[df["Brand"].isin(sel_brands)]
if sel_body:
    df = df[df["BodyType"].isin(sel_body)]
if sel_fuel:
    df = df[df["FuelType"].isin(sel_fuel)]
if sel_condition:
    df = df[df["UsedOrNew"].isin(sel_condition)]
df = df[(df["Year"] >= sel_year[0]) & (df["Year"] <= sel_year[1])]
df = df[(df["Price"] >= sel_price[0]) & (df["Price"] <= sel_price[1])]

st.sidebar.markdown(
    f"<p style='color:{GREY}; font-size:12px;'>Showing <b>{len(df):,}</b> of "
    f"{len(raw_df):,} listings</p>",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No listings match the current filters. Please broaden your filter selection in the sidebar.")
    st.stop()

# ==================================================================
# PAGE: HOME
# ==================================================================
if page == "🏠 Home":
    st.markdown(
        f"""
        <div style="background: linear-gradient(120deg, {NAVY} 0%, #012A4A 60%, {PRIMARY} 150%);
                    padding: 42px 36px; border-radius: 16px; color: white; margin-bottom: 24px;">
            <h1 style="color:white; margin:0; font-size:38px;">Car Market Intelligence Dashboard</h1>
            <p style="color:#DDE7EE; font-size:16px; max-width:720px; margin-top:10px;">
                An end-to-end exploratory data analysis platform covering pricing trends, body styles,
                usage patterns, and statistical relationships across {len(raw_df):,} vehicle listings —
                built for O7 Services car market analysis. Includes interactive sunburst and 3D
                visuals for deeper multi-dimensional exploration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Total Listings", f"{len(df):,}")
    kpi(c2, "Average Price", fmt_money(df["Price"].mean()))
    kpi(c3, "Median Price", fmt_money(df["Price"].median()))
    kpi(c4, "Avg. Kilometres", f"{df['Kilometres'].mean():,.0f} km")
    kpi(c5, "Brands Covered", f"{df['Brand'].nunique()}")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("What's inside this dashboard")
        st.markdown(
            """
- **📊 Overview** — dataset structure, data quality, and a Brand → Body Type sunburst
- **💰 Price** — price distributions, drivers of price, and a 3D price/kilometres/year view
- **🚙 Car Body** — body style popularity, price, and a Body Type → Fuel Type sunburst
- **🛣️ Driven** — kilometres, drivetrains, and usage-related patterns
- **📈 Statistical** — correlations, outlier checks, and a 3D multivariate scatter
- **🧩 Models** — top car models, a Brand → Model sunburst, and value-for-money analysis
            """
        )

    with right:
        st.subheader("Snapshot of the data")
        st.dataframe(df.head(8), use_container_width=True, height=280)
        st.caption(f"Columns: {', '.join(raw_df.columns[:19])}")

    st.markdown("---")
    st.subheader("Quick Look")
    q1, q2, q3 = st.columns(3)

    with q1:
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        top_brands = df["Brand"].value_counts().head(6)
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax, palette=PALETTE)
        ax.set_title("Top Brands by Listings")
        ax.set_xlabel("Listings"); ax.set_ylabel("")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with q2:
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        body_counts = df["BodyType"].value_counts().head(5)
        ax.pie(body_counts.values, labels=body_counts.index, autopct="%1.0f%%",
               colors=PALETTE, startangle=90, wedgeprops={"edgecolor": "white"})
        ax.set_title("Body Type Mix")
        st.pyplot(fig, use_container_width=True)

    with q3:
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        sns.histplot(df["Price"], bins=25, kde=True, ax=ax, color=PRIMARY)
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price (AUD)")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    insight(
        f"The best-represented brand is **{df['Brand'].value_counts().idxmax()}** and the most common "
        f"body style is **{df['BodyType'].value_counts().idxmax()}**, together shaping the bulk of "
        f"this market segment."
    )

# ==================================================================
# PAGE: OVERVIEW
# ==================================================================
elif page == "📊 Overview":
    banner("Dataset Overview", "Structure, data quality, and category-level distributions")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Rows", f"{df.shape[0]:,}")
    kpi(c2, "Columns", f"{df.shape[1]}")
    kpi(c3, "Missing Values", f"{int(df.isnull().sum().sum())}")
    kpi(c4, "Duplicate Rows", f"{int(df.duplicated().sum())}")

    st.write("")
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "🧬 Column Types", "📈 Summary Statistics"])

    with tab1:
        st.dataframe(df, use_container_width=True, height=380)
        st.download_button(
            "⬇️ Download filtered data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_car_data.csv",
            mime="text/csv",
        )

    with tab2:
        dtypes_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Type": df.dtypes.values.astype(str),
            "Unique Values": [df[c].nunique() for c in df.columns],
            "Missing": [df[c].isnull().sum() for c in df.columns],
        })
        st.dataframe(dtypes_df, use_container_width=True, height=380)

    with tab3:
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("Categorical Distributions")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        top_brands = df["Brand"].value_counts().head(10)
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax, palette=PALETTE)
        ax.set_title("Top 10 Car Brands by Listings")
        ax.set_xlabel("Number of Listings"); ax.set_ylabel("")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with r1c2:
        top_brand_list = df["Brand"].value_counts().head(8).index
        sunburst_df = df[df["Brand"].isin(top_brand_list)]
        fig = px.sunburst(
            sunburst_df, path=["Brand", "BodyType"],
            color="Brand", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(title="Market Structure: Brand → Body Type (Top 8 Brands)")
        style_plotly(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Transmission", ax=ax, palette=PALETTE)
        ax.set_title("Transmission Type")
        ax.set_xlabel(""); ax.set_ylabel("Listings")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with r2c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="UsedOrNew", ax=ax, palette=PALETTE,
                      order=df["UsedOrNew"].value_counts().index)
        ax.set_title("Used vs New Cars")
        ax.set_xlabel(""); ax.set_ylabel("Listings")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    insight(
        f"**{df['Transmission'].value_counts(normalize=True).idxmax()}** transmission dominates "
        f"the market at {df['Transmission'].value_counts(normalize=True).max()*100:.1f}% of listings, "
        f"and **{df['UsedOrNew'].value_counts(normalize=True).idxmax()}** vehicles make up "
        f"{df['UsedOrNew'].value_counts(normalize=True).max()*100:.1f}% of the selection."
    )

# ==================================================================
# PAGE: PRICE
# ==================================================================
elif page == "💰 Price":
    banner("Price Analysis", "Distributions, drivers, and trends behind vehicle pricing")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Average Price", fmt_money(df["Price"].mean()))
    kpi(c2, "Median Price", fmt_money(df["Price"].median()))
    kpi(c3, "Min Price", fmt_money(df["Price"].min()))
    kpi(c4, "Max Price", fmt_money(df["Price"].max()))

    st.write("")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.histplot(df["Price"], bins=30, kde=True, ax=ax, color=PRIMARY)
    ax.set_title("Distribution of Car Prices")
    ax.set_xlabel("Price (AUD)"); ax.set_ylabel("Number of Cars")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        top10 = df.groupby("Brand")["Price"].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=top10.values, y=top10.index, ax=ax, palette=PALETTE)
        ax.set_title("Top 10 Brands by Average Price")
        ax.set_xlabel("Average Price (AUD)"); ax.set_ylabel("")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df, x="FuelType", y="Price", ax=ax, palette=PALETTE, errorbar=None)
        ax.set_title("Average Price by Fuel Type")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel(""); ax.set_ylabel("Average Price (AUD)")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Price vs Kilometres vs Year (3D View)")
    sample = df if len(df) < 3000 else df.sample(3000, random_state=42)
    fig3d = px.scatter_3d(
        sample, x="Kilometres", y="Year", z="Price", color="FuelType",
        color_discrete_sequence=PALETTE, opacity=0.65,
        hover_data=["Brand", "Model"],
    )
    fig3d.update_traces(marker=dict(size=4))
    fig3d.update_layout(
        title="Price vs Kilometres vs Year, coloured by Fuel Type",
        scene=dict(
            xaxis_title="Kilometres", yaxis_title="Year", zaxis_title="Price (AUD)",
            xaxis=dict(backgroundcolor="white", gridcolor="#ECECEC"),
            yaxis=dict(backgroundcolor="white", gridcolor="#ECECEC"),
            zaxis=dict(backgroundcolor="white", gridcolor="#ECECEC"),
        ),
    )
    style_plotly(fig3d, height=580)
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("Drag to rotate and scroll to zoom — explore how kilometres, year, and fuel type jointly shape price.")

    st.markdown("#### Price Trend Over Manufacturing Years")
    year_price = df.groupby("Year")["Price"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.lineplot(data=year_price, x="Year", y="Price", marker="o", ax=ax, color=PRIMARY)
    ax.set_title("Average Price by Manufacturing Year")
    ax.set_xlabel("Year"); ax.set_ylabel("Average Price (AUD)")
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x="FuelType", y="Price", ax=ax, palette=PALETTE)
        ax.set_title("Price Distribution by Fuel Type")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel(""); ax.set_ylabel("Price (AUD)")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x="Transmission", y="Price", ax=ax, palette=PALETTE)
        ax.set_title("Price Distribution by Transmission")
        ax.set_xlabel(""); ax.set_ylabel("Price (AUD)")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    corr_price_km = df["Price"].corr(df["Kilometres"])
    top_brand_name = df.groupby("Brand")["Price"].mean().idxmax()
    insight(
        f"Price and kilometres driven have a correlation of **{corr_price_km:.2f}** — "
        f"{'a moderate-to-strong negative relationship, meaning higher-mileage cars tend to be cheaper' if corr_price_km < -0.3 else 'a relatively weak linear relationship'}. "
        f"**{top_brand_name}** commands the highest average price in the current selection."
    )

# ==================================================================
# PAGE: CAR BODY
# ==================================================================
elif page == "🚙 Car Body":
    banner("Car Body Analysis", "Body style popularity and its relationship with price")

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Body Styles", f"{df['BodyType'].nunique()}")
    kpi(c2, "Most Common", df["BodyType"].value_counts().idxmax())
    kpi(c3, "Highest Avg. Price", df.groupby("BodyType")["Price"].mean().idxmax())

    st.write("")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        sns.countplot(data=df, x="BodyType", order=df["BodyType"].value_counts().index,
                      ax=ax, palette=PALETTE)
        ax.set_title("Body Type Distribution")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel(""); ax.set_ylabel("Listings")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        top_body_list = df["BodyType"].value_counts().head(6).index
        sunburst_body = df[df["BodyType"].isin(top_body_list)]
        fig = px.sunburst(
            sunburst_body, path=["BodyType", "FuelType"],
            color="BodyType", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(title="Body Type → Fuel Type Breakdown")
        style_plotly(fig, height=460)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Price by Body Type")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = df.groupby("BodyType")["Price"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="BodyType", y="Price", order=order, ax=ax, palette=PALETTE, errorbar=None)
    ax.set_title("Average Price by Body Type")
    ax.tick_params(axis="x", rotation=40)
    ax.set_xlabel(""); ax.set_ylabel("Average Price (AUD)")
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x="BodyType", y="Price", order=order, ax=ax, palette=PALETTE)
        ax.set_title("Price Spread by Body Type")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel(""); ax.set_ylabel("Price (AUD)")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cross = pd.crosstab(df["BodyType"], df["Doors"])
        sns.heatmap(cross, cmap="YlGnBu", linewidths=0.5, ax=ax, annot=True, fmt="d", cbar=False)
        ax.set_title("Body Type vs Number of Doors")
        ax.set_xlabel("Doors"); ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)

    top_body = df["BodyType"].value_counts().idxmax()
    priciest_body = df.groupby("BodyType")["Price"].mean().idxmax()
    insight(
        f"**{top_body}** is the most listed body style, while **{priciest_body}** commands the "
        f"highest average asking price — a useful signal for inventory and pricing strategy."
    )

# ==================================================================
# PAGE: DRIVEN (Kilometres & Drive Type)
# ==================================================================
elif page == "🛣️ Driven":
    banner("Driven Analysis", "Kilometres, drivetrains, fuel efficiency, and usage patterns")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Avg. Kilometres", f"{df['Kilometres'].mean():,.0f} km")
    kpi(c2, "Median Kilometres", f"{df['Kilometres'].median():,.0f} km")
    kpi(c3, "Most Common Drive Type", df["DriveType"].value_counts().idxmax())
    kpi(c4, "Avg. Fuel Consumption", f"{df['FuelConsumption_num'].mean():.1f} L/100km"
        if "FuelConsumption_num" in df.columns else "N/A")

    st.write("")
    
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.histplot(df["Kilometres"], bins=30, kde=True, ax=ax, color=NAVY)
    ax.set_title("Distribution of Kilometres Driven")
    ax.set_xlabel("Kilometres"); ax.set_ylabel("Number of Cars")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fuel_km = df.groupby("FuelType")["Kilometres"].mean().sort_values()
        sns.barplot(x=fuel_km.index, y=fuel_km.values, ax=ax, palette=PALETTE)
        ax.set_title("Average Kilometres by Fuel Type")
        ax.tick_params(axis="x", rotation=40)
        ax.set_xlabel(""); ax.set_ylabel("Average Kilometres")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df, x="DriveType", order=df["DriveType"].value_counts().index,
                      ax=ax, palette=PALETTE)
        ax.set_title("Drive Type Distribution")
        ax.set_xlabel(""); ax.set_ylabel("Listings")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    drive_price = df.groupby("DriveType")["Price"].mean().sort_values(ascending=False)
    sns.barplot(x=drive_price.index, y=drive_price.values, ax=ax, palette=PALETTE)
    ax.set_title("Average Price by Drive Type")
    ax.set_xlabel(""); ax.set_ylabel("Average Price (AUD)")
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    lowest_km_fuel = df.groupby("FuelType")["Kilometres"].mean().idxmin()
    top_drive = df.groupby("DriveType")["Price"].mean().idxmax()
    insight(
        f"**{lowest_km_fuel}** vehicles tend to have the lowest average kilometres in this selection, "
        f"while **{top_drive}** drivetrains fetch the highest average price — often reflecting "
        f"performance or premium SUV segments."
    )

# ==================================================================
# PAGE: STATISTICAL
# ==================================================================
elif page == "📈 Statistical":
    banner("Statistical Analysis", "Correlations, outlier checks, and multivariate relationships")

    numeric_df = df.select_dtypes(include=np.number)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Between Numeric Features")
    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x=df["Price"], ax=ax, color=PRIMARY)
        ax.set_title("Boxplot of Price (Outlier Check)")
        ax.set_xlabel("Price (AUD)")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x=df["Kilometres"], ax=ax, color=NAVY)
        ax.set_title("Boxplot of Kilometres (Outlier Check)")
        ax.set_xlabel("Kilometres")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Price Distribution by Number of Seats")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.violinplot(data=df, x="Seats", y="Price", ax=ax, palette=PALETTE)
    ax.set_title("Price Distribution by Seats")
    ax.set_xlabel("Seats"); ax.set_ylabel("Price (AUD)")
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    st.markdown("#### Multivariate Relationships (3D)")
    sample_n = min(len(df), 3000)
    sample_df = df.sample(sample_n, random_state=42)
    fig3d_stat = px.scatter_3d(
        sample_df, x="Kilometres", y="CarAge", z="Price", color="BodyType",
        color_discrete_sequence=PALETTE, opacity=0.6,
        hover_data=["Brand", "Model", "FuelType"],
    )
    fig3d_stat.update_traces(marker=dict(size=3.5))
    fig3d_stat.update_layout(
        title=f"Price, Kilometres & Car Age by Body Type (sample of {sample_n:,})",
        scene=dict(
            xaxis_title="Kilometres", yaxis_title="Car Age (yrs)", zaxis_title="Price (AUD)",
            xaxis=dict(backgroundcolor="white", gridcolor="#ECECEC"),
            yaxis=dict(backgroundcolor="white", gridcolor="#ECECEC"),
            zaxis=dict(backgroundcolor="white", gridcolor="#ECECEC"),
        ),
    )
    style_plotly(fig3d_stat, height=600)
    st.plotly_chart(fig3d_stat, use_container_width=True)
    st.caption("Rotate freely — newer, low-kilometre cars cluster toward the higher-price end regardless of body type.")

    st.markdown("#### Engine Cylinders")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df, x="CylindersinEngine", ax=ax, palette=PALETTE)
        ax.set_title("Engine Cylinders Distribution")
        ax.set_xlabel("Cylinders"); ax.set_ylabel("Listings")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        pivot = pd.pivot_table(df, index="Brand", columns="FuelType", values="Price", aggfunc="mean")
        top_index = df["Brand"].value_counts().head(12).index
        sns.heatmap(pivot.loc[pivot.index.intersection(top_index)], cmap="YlGnBu",
                    linewidths=0.4, ax=ax, cbar_kws={"label": "Avg Price"})
        ax.set_title("Avg Price: Brand x Fuel Type (Top Brands)")
        ax.set_xlabel(""); ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)

    strongest = numeric_df.corr()["Price"].drop("Price").abs().idxmax()
    strongest_val = numeric_df.corr()["Price"].drop("Price")[strongest]
    insight(
        f"Among numeric features, **{strongest}** shows the strongest linear relationship with "
        f"Price (r = {strongest_val:.2f}). Both Price and Kilometres retain some right-skew even "
        f"after IQR-based outlier removal, which is typical for used-car marketplaces."
    )

# ==================================================================
# PAGE: MODELS  (Car Models deep-dive analysis)
# ==================================================================
elif page == "🧩 Models":
    banner("Car Models Analysis", "Rankings, side-by-side comparison, and value-for-money insights")

    tab1, tab2, tab3 = st.tabs(["🏆 Top Car Models", "🔍 Model Comparison", "💎 Value Analysis"])

    # ---------------- TAB 1: Top car models ----------------
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            top_models = (
                df.groupby("Model")["Price"].mean().sort_values(ascending=False).head(10)
            )
            sns.barplot(x=top_models.values, y=top_models.index, ax=ax, palette=PALETTE)
            ax.set_title("Top 10 Most Expensive Car Models")
            ax.set_xlabel("Average Price (AUD)"); ax.set_ylabel("")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)

        with c2:
            top_brands_models = df["Brand"].value_counts().head(6).index
            sb_df = df[df["Brand"].isin(top_brands_models)]
            top_models_per_brand = (
                sb_df.groupby(["Brand", "Model"]).size().reset_index(name="Listings")
                .sort_values("Listings", ascending=False)
                .groupby("Brand").head(4)
            )
            fig = px.sunburst(
                top_models_per_brand, path=["Brand", "Model"], values="Listings",
                color="Brand", color_discrete_sequence=PALETTE,
            )
            fig.update_layout(title="Top Models by Brand (Top 6 Brands)")
            style_plotly(fig, height=520)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Explore a Specific Model")
        model_choice = st.selectbox("Choose a model", sorted(df["Model"].unique()))
        model_df = df[df["Model"] == model_choice]
        m1, m2, m3, m4 = st.columns(4)
        kpi(m1, "Listings", f"{len(model_df):,}")
        kpi(m2, "Avg. Price", fmt_money(model_df["Price"].mean()))
        kpi(m3, "Avg. Kilometres", f"{model_df['Kilometres'].mean():,.0f} km")
        kpi(m4, "Common Body Type", model_df["BodyType"].mode()[0] if not model_df.empty else "N/A")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(model_df["Price"], bins=15, kde=True, ax=ax, color=PRIMARY)
            ax.set_title(f"Price Distribution — {model_choice}")
            ax.set_xlabel("Price (AUD)")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            year_counts = model_df["Year"].value_counts().sort_index()
            sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", ax=ax, color=NAVY)
            ax.set_title(f"Listings by Year — {model_choice}")
            ax.set_xlabel("Year"); ax.set_ylabel("Listings")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)

        insight(
            f"The **{model_choice}** has an average listed price of "
            f"{fmt_money(model_df['Price'].mean())} across {len(model_df):,} listings in the "
            f"current filter selection."
        )

    # ---------------- TAB 2: Model comparison ----------------
    with tab2:
        st.markdown("Pick two or more models to compare side-by-side.")
        default_models = df["Model"].value_counts().head(3).index.tolist()
        compare_models = st.multiselect(
            "Models to compare", sorted(df["Model"].unique()), default=default_models
        )

        if len(compare_models) < 2:
            st.info("Select at least two models above to see a comparison.")
        else:
            comp_df = df[df["Model"].isin(compare_models)]

            summary = comp_df.groupby("Model").agg(
                Listings=("Price", "count"),
                Avg_Price=("Price", "mean"),
                Median_Price=("Price", "median"),
                Avg_Kilometres=("Kilometres", "mean"),
                Avg_Year=("Year", "mean"),
                Most_Common_Body=("BodyType", lambda x: x.mode()[0] if not x.mode().empty else "N/A"),
            ).round(0)
            st.dataframe(
                summary.style.format({
                    "Avg_Price": "${:,.0f}", "Median_Price": "${:,.0f}",
                    "Avg_Kilometres": "{:,.0f}", "Avg_Year": "{:.0f}",
                }),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                sns.boxplot(data=comp_df, x="Model", y="Price", ax=ax, palette=PALETTE)
                ax.set_title("Price Comparison")
                ax.tick_params(axis="x", rotation=20)
                ax.set_xlabel(""); ax.set_ylabel("Price (AUD)")
                style_ax(ax)
                st.pyplot(fig, use_container_width=True)

            with c2:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                sns.boxplot(data=comp_df, x="Model", y="Kilometres", ax=ax, palette=PALETTE)
                ax.set_title("Kilometres Comparison")
                ax.tick_params(axis="x", rotation=20)
                ax.set_xlabel(""); ax.set_ylabel("Kilometres")
                style_ax(ax)
                st.pyplot(fig, use_container_width=True)

            fig, ax = plt.subplots(figsize=(11, 4.3))
            trend = comp_df.groupby(["Year", "Model"])["Price"].mean().reset_index()
            sns.lineplot(data=trend, x="Year", y="Price", hue="Model", marker="o", ax=ax, palette=PALETTE)
            ax.set_title("Average Price by Year, by Model")
            ax.set_xlabel("Year"); ax.set_ylabel("Average Price (AUD)")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)

            cheapest = summary["Avg_Price"].idxmin()
            priciest = summary["Avg_Price"].idxmax()
            insight(
                f"Among the selected models, **{priciest}** is the most expensive on average "
                f"({fmt_money(summary.loc[priciest, 'Avg_Price'])}), while **{cheapest}** is the "
                f"most affordable ({fmt_money(summary.loc[cheapest, 'Avg_Price'])})."
            )

    # ---------------- TAB 3: Value analysis ----------------
    with tab3:
        st.markdown(
            "A simple, non-ML **value score** highlighting which listings offer the most car "
            "for the money: low price, low kilometres, and a newer year all raise the score."
        )

        value_df = df.copy()
        # Normalize each factor 0-1, combine into a single value score (higher = better value)
        def norm(s, invert=False):
            rng = s.max() - s.min()
            if rng == 0:
                return pd.Series(0.5, index=s.index)
            n = (s - s.min()) / rng
            return 1 - n if invert else n

        value_df["ValueScore"] = (
            0.5 * norm(value_df["Price"], invert=True)
            + 0.3 * norm(value_df["Kilometres"], invert=True)
            + 0.2 * norm(value_df["Year"], invert=False)
        ) * 100

        best_value = (
            value_df.groupby("Model")
            .agg(Avg_Price=("Price", "mean"), Avg_Kilometres=("Kilometres", "mean"),
                 Avg_ValueScore=("ValueScore", "mean"), Listings=("Price", "count"))
            .query("Listings >= 5")
            .sort_values("Avg_ValueScore", ascending=False)
            .head(10)
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            sns.barplot(x=best_value["Avg_ValueScore"], y=best_value.index, ax=ax, palette=PALETTE)
            ax.set_title("Top 10 Best-Value Models (min. 5 listings)")
            ax.set_xlabel("Average Value Score (0–100)"); ax.set_ylabel("")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)

        with c2:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            age_price = df.groupby("CarAge")["Price"].mean().reset_index()
            sns.lineplot(data=age_price, x="CarAge", y="Price", marker="o", ax=ax, color=PRIMARY)
            ax.set_title("Average Price by Car Age")
            ax.set_xlabel("Car Age (years)"); ax.set_ylabel("Average Price (AUD)")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)

        st.markdown("#### Price per Kilometre-Year Efficiency, by Brand")
        brand_eff = (
            df.groupby("Brand")
            .agg(Avg_Price=("Price", "mean"), Avg_Kilometres=("Kilometres", "mean"), Listings=("Price", "count"))
            .query("Listings >= 10")
        )
        brand_eff["Price_per_1000km"] = brand_eff["Avg_Price"] / (brand_eff["Avg_Kilometres"] / 1000)
        brand_eff = brand_eff.sort_values("Price_per_1000km").head(10)

        fig, ax = plt.subplots(figsize=(11, 4.3))
        sns.barplot(x=brand_eff["Price_per_1000km"], y=brand_eff.index, ax=ax, palette=PALETTE)
        ax.set_title("Lowest Price per 1,000 km Driven, by Brand (min. 10 listings)")
        ax.set_xlabel("AUD per 1,000 km"); ax.set_ylabel("")
        style_ax(ax)
        st.pyplot(fig, use_container_width=True)

        top_value_model = best_value.index[0] if not best_value.empty else "N/A"
        insight(
            f"**{top_value_model}** currently scores highest on value (balancing low price, low "
            f"kilometres, and newer year), making it a strong pick for budget-conscious buyers in "
            f"this segment."
        )

# ==================================================================
# FOOTER
# ==================================================================
st.markdown("---")
st.markdown(
    f"""<p style='text-align:center; color:{GREY}; font-size:12.5px;'>
    Car Market Intelligence Dashboard &middot; Built with Streamlit, Pandas, NumPy, Matplotlib, Seaborn &amp; Plotly
    &middot; Data: Cleaned Car Market Dataset
    </p>""",
    unsafe_allow_html=True,
)