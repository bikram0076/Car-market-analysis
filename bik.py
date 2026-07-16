
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Car Market Intelligence Dashboard",
    layout="wide",
)
NAVY = "#003049"  
RED = "#D62828"     
PALETTE = [NAVY, RED, "#F77F00", "#4B3F72", "#2A9D8F"]

sns.set_theme(style="whitegrid")

st.markdown("""
<style>
    /* Light background for the main canvas */
    .stApp {
        background-color: #f0f4f8;
    }
    /* Deep Navy Sidebar to match the NAVY variable */
    [data-testid="stSidebar"] {
        background-color: #003049;
    }
    /* Make sidebar text white for readability against navy */
    [data-testid="stSidebar"] * {
        color: #DCDCDC !important;
    }
    /* Style the metric boxes with a white background and shadow */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #D62828;
    }
    
    /* FIX: Prevent metric numbers from truncating with '...' */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; /* Shrinks the font slightly */
        white-space: normal !important; /* Allows text to wrap if necessary */
        word-break: break-word !important; 
    }

    /* Make main titles match your NAVY accent color */
    h1, h2, h3 {
        color: #003049 !important;
    }
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def clean_axes(ax):
    """Remove the top and right border lines from a chart (just for a cleaner look)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

DATA_PATH = "Cleaned_Car_Market_Dataset.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()


    if "FuelConsumption" in df.columns:
        df["FuelConsumption_num"] = df["FuelConsumption"].str.extract(r"([\d.]+)").astype(float)

    if "Year" in df.columns:
        df["CarAge"] = df["Year"].max() - df["Year"]

    return df



try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("Could not find 'Cleaned_Car_Market_Dataset.csv'. Please upload it below.")
    uploaded_file = st.file_uploader("Upload Cleaned_Car_Market_Dataset.csv", type="csv")
    if uploaded_file is None:
        st.stop()
    raw_df = load_data(uploaded_file)


st.sidebar.title("Car Market Dashboard")

page = st.sidebar.radio(
    "Go to page",
    ["Home", "Overview", "Price", "Car Body", "Driven", "Statistical", "Models"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

sel_brands = st.sidebar.multiselect("Brand", sorted(raw_df["Brand"].unique()))
sel_body = st.sidebar.multiselect("Body Type", sorted(raw_df["BodyType"].unique()))
sel_fuel = st.sidebar.multiselect("Fuel Type", sorted(raw_df["FuelType"].unique()))
sel_condition = st.sidebar.multiselect("Condition", sorted(raw_df["UsedOrNew"].unique()))

year_min, year_max = int(raw_df["Year"].min()), int(raw_df["Year"].max())
sel_year = st.sidebar.slider("Manufacturing Year", year_min, year_max, (year_min, year_max))

price_min, price_max = int(raw_df["Price"].min()), int(raw_df["Price"].max())
sel_price = st.sidebar.slider("Price Range (AUD)", price_min, price_max, (price_min, price_max), step=500)

if st.sidebar.button("Reset Filters"):
    st.rerun()

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

st.sidebar.caption(f"Showing {len(df):,} of {len(raw_df):,} listings")

if df.empty:
    st.warning("No listings match your filters. Please change the filters in the sidebar.")
    st.stop()


def money(x):
    return f"${x:,.0f}"


if page == "Home":
    st.title("🚗 Car Market Intelligence Dashboard")
    st.write(
        f"This dashboard explores {len(raw_df):,} car listings — covering pricing, "
        "body styles, and usage patterns. Use the sidebar to filter the data and "
        "navigate between pages."
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Listings", f"{len(df):,}")
    col2.metric("Average Price", money(df["Price"].mean()))
    col3.metric("Median Price", money(df["Price"].median()))
    col4.metric("Avg. Kilometres", f"{df['Kilometres'].mean():,.0f} km")
    col5.metric("Brands", df["Brand"].nunique())

    st.subheader("What's in this dashboard")
    st.markdown(
        """
        - **Overview** — dataset structure and category breakdowns
        - **Price** — price distributions and what drives price
        - **Car Body** — body style popularity and pricing
        - **Driven** — kilometres, drivetrains, and fuel type patterns
        - **Statistical** — correlations and outlier checks
        - **Models** — top models, comparisons, and value for money
        """
    )

    st.subheader("Quick Look")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        top_brands = df["Brand"].value_counts().head(6)
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax, palette=PALETTE)
        ax.set_title("Top Brands by Listings")
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        body_counts = df["BodyType"].value_counts().head(5)
        ax.pie(body_counts.values, labels=body_counts.index, autopct="%1.0f%%", colors=PALETTE)
        ax.set_title("Body Type Mix")
        st.pyplot(fig)

    with c3:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.histplot(df["Price"], bins=25, kde=True, ax=ax, color=RED)
        ax.set_title("Price Distribution")
        clean_axes(ax)
        st.pyplot(fig)

    st.info(
        f"The most listed brand is **{df['Brand'].value_counts().idxmax()}**, and the "
        f"most common body style is **{df['BodyType'].value_counts().idxmax()}**."
    )


elif page == "Overview":
    st.header("Dataset Overview")
    st.caption("Structure, data quality, and category-level distributions")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Types", "Summary Statistics"])

    with tab1:
        st.dataframe(df, use_container_width=True, height=380)
        st.download_button(
            "Download filtered data (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_car_data.csv",
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
        st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.subheader("Categorical Distributions")
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        top_brands = df["Brand"].value_counts().head(10)
        sns.barplot(x=top_brands.values, y=top_brands.index, ax=ax, palette=PALETTE)
        ax.set_title("Top 10 Brands by Listings")
        clean_axes(ax)
        st.pyplot(fig)

    with r1c2:
    
        top_brand_list = df["Brand"].value_counts().head(8).index
        sunburst_df = df[df["Brand"].isin(top_brand_list)]
        fig = px.sunburst(
            sunburst_df, path=["Brand", "BodyType"],
            color="Brand", color_discrete_sequence=PALETTE,
            title="Brand → Body Type (Top 8 Brands)",
        )
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Transmission", ax=ax, palette=PALETTE)
        ax.set_title("Transmission Type")
        clean_axes(ax)
        st.pyplot(fig)

    with r2c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        order = df["UsedOrNew"].value_counts().index
        sns.countplot(data=df, x="UsedOrNew", order=order, ax=ax, palette=PALETTE)
        ax.set_title("Used vs New Cars")
        clean_axes(ax)
        st.pyplot(fig)

    st.info(
        f"**{df['Transmission'].value_counts().idxmax()}** transmission is the most common, "
        f"and **{df['UsedOrNew'].value_counts().idxmax()}** vehicles make up most of the listings."
    )

elif page == "Price":
    st.header("Price Analysis")
    st.caption("Distributions and trends behind vehicle pricing")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Price", money(df["Price"].mean()))
    col2.metric("Median Price", money(df["Price"].median()))
    col3.metric("Min Price", money(df["Price"].min()))
    col4.metric("Max Price", money(df["Price"].max()))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.histplot(df["Price"], bins=30, kde=True, ax=ax, color=RED)
    ax.set_title("Distribution of Car Prices")
    ax.set_xlabel("Price (AUD)")
    clean_axes(ax)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        top10 = df.groupby("Brand")["Price"].mean().sort_values(ascending=False).head(10)
        sns.barplot(x=top10.values, y=top10.index, ax=ax, palette=PALETTE)
        ax.set_title("Top 10 Brands by Average Price")
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df, x="FuelType", y="Price", ax=ax, palette=PALETTE, errorbar=None)
        ax.set_title("Average Price by Fuel Type")
        ax.tick_params(axis="x", rotation=40)
        clean_axes(ax)
        st.pyplot(fig)

    st.subheader("Price Trend Over Manufacturing Years")
    year_price = df.groupby("Year")["Price"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    sns.lineplot(data=year_price, x="Year", y="Price", marker="o", ax=ax, color=RED)
    ax.set_title("Average Price by Manufacturing Year")
    clean_axes(ax)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x="FuelType", y="Price", ax=ax, palette=PALETTE)
        ax.set_title("Price Spread by Fuel Type")
        ax.tick_params(axis="x", rotation=40)
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x="Transmission", y="Price", ax=ax, palette=PALETTE)
        ax.set_title("Price Spread by Transmission")
        clean_axes(ax)
        st.pyplot(fig)

    corr_price_km = df["Price"].corr(df["Kilometres"])
    top_brand_name = df.groupby("Brand")["Price"].mean().idxmax()
    st.info(
        f"Price and kilometres driven have a correlation of **{corr_price_km:.2f}** "
        f"(negative means higher-mileage cars tend to cost less). "
        f"**{top_brand_name}** has the highest average price in the current selection."
    )


elif page == "Car Body":
    st.header("Car Body Analysis")
    st.caption("Body style popularity and its relationship with price")

    col1, col2, col3 = st.columns(3)
    col1.metric("Body Styles", df["BodyType"].nunique())
    col2.metric("Most Common", df["BodyType"].value_counts().idxmax())
    col3.metric("Highest Avg. Price", df.groupby("BodyType")["Price"].mean().idxmax())

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        order = df["BodyType"].value_counts().index
        sns.countplot(data=df, x="BodyType", order=order, ax=ax, palette=PALETTE)
        ax.set_title("Body Type Distribution")
        ax.tick_params(axis="x", rotation=40)
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        top_body_list = df["BodyType"].value_counts().head(6).index
        sunburst_body = df[df["BodyType"].isin(top_body_list)]
        fig = px.sunburst(
            sunburst_body, path=["BodyType", "FuelType"],
            color="BodyType", color_discrete_sequence=PALETTE,
            title="Body Type → Fuel Type",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price by Body Type")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = df.groupby("BodyType")["Price"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="BodyType", y="Price", order=order, ax=ax, palette=PALETTE, errorbar=None)
    ax.set_title("Average Price by Body Type")
    ax.tick_params(axis="x", rotation=40)
    clean_axes(ax)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(data=df, x="BodyType", y="Price", order=order, ax=ax, palette=PALETTE)
        ax.set_title("Price Spread by Body Type")
        ax.tick_params(axis="x", rotation=40)
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cross = pd.crosstab(df["BodyType"], df["Doors"])
        sns.heatmap(cross, cmap="Blues", linewidths=0.5, ax=ax, annot=True, fmt="d", cbar=False)
        ax.set_title("Body Type vs Number of Doors")
        st.pyplot(fig)

    top_body = df["BodyType"].value_counts().idxmax()
    priciest_body = df.groupby("BodyType")["Price"].mean().idxmax()
    st.info(
        f"**{top_body}** is the most listed body style, while **{priciest_body}** has the "
        f"highest average asking price."
    )


elif page == "Driven":
    st.header("Driven Analysis")
    st.caption("Kilometres, drivetrains, and usage patterns")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Kilometres", f"{df['Kilometres'].mean():,.0f} km")
    col2.metric("Median Kilometres", f"{df['Kilometres'].median():,.0f} km")
    col3.metric("Most Common Drive Type", df["DriveType"].value_counts().idxmax())
    if "FuelConsumption_num" in df.columns:
        col4.metric("Avg. Fuel Use", f"{df['FuelConsumption_num'].mean():.1f} L/100km")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.histplot(df["Kilometres"], bins=30, kde=True, ax=ax, color=NAVY)
    ax.set_title("Distribution of Kilometres Driven")
    clean_axes(ax)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fuel_km = df.groupby("FuelType")["Kilometres"].mean().sort_values()
        sns.barplot(x=fuel_km.index, y=fuel_km.values, ax=ax, palette=PALETTE)
        ax.set_title("Average Kilometres by Fuel Type")
        ax.tick_params(axis="x", rotation=40)
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        order = df["DriveType"].value_counts().index
        sns.countplot(data=df, x="DriveType", order=order, ax=ax, palette=PALETTE)
        ax.set_title("Drive Type Distribution")
        clean_axes(ax)
        st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    drive_price = df.groupby("DriveType")["Price"].mean().sort_values(ascending=False)
    sns.barplot(x=drive_price.index, y=drive_price.values, ax=ax, palette=PALETTE)
    ax.set_title("Average Price by Drive Type")
    clean_axes(ax)
    st.pyplot(fig)

    lowest_km_fuel = df.groupby("FuelType")["Kilometres"].mean().idxmin()
    top_drive = df.groupby("DriveType")["Price"].mean().idxmax()
    st.info(
        f"**{lowest_km_fuel}** vehicles tend to have the lowest average kilometres, while "
        f"**{top_drive}** drivetrains have the highest average price."
    )


elif page == "Statistical":
    st.header("Statistical Analysis")
    st.caption("Correlations and outlier checks")

    numeric_df = df.select_dtypes(include=np.number)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Between Numeric Features")
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x=df["Price"], ax=ax, color=RED)
        ax.set_title("Boxplot of Price (Outlier Check)")
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.boxplot(x=df["Kilometres"], ax=ax, color=NAVY)
        ax.set_title("Boxplot of Kilometres (Outlier Check)")
        clean_axes(ax)
        st.pyplot(fig)

    st.subheader("Price Distribution by Number of Seats")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.violinplot(data=df, x="Seats", y="Price", ax=ax, palette=PALETTE)
    ax.set_title("Price Distribution by Seats")
    clean_axes(ax)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.countplot(data=df, x="CylindersinEngine", ax=ax, palette=PALETTE)
        ax.set_title("Engine Cylinders Distribution")
        clean_axes(ax)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        pivot = pd.pivot_table(df, index="Brand", columns="FuelType", values="Price", aggfunc="mean")
        top_index = df["Brand"].value_counts().head(12).index
        sns.heatmap(pivot.loc[pivot.index.intersection(top_index)], cmap="Blues", linewidths=0.4, ax=ax)
        ax.set_title("Avg Price: Brand x Fuel Type (Top Brands)")
        st.pyplot(fig)

    strongest = numeric_df.corr()["Price"].drop("Price").abs().idxmax()
    strongest_val = numeric_df.corr()["Price"].drop("Price")[strongest]
    st.info(
        f"**{strongest}** has the strongest linear relationship with Price "
        f"(correlation = {strongest_val:.2f})."
    )

elif page == "Models":
    st.header("Car Models Analysis")
    st.caption("Rankings, comparisons, and value-for-money insights")

    tab1, tab2, tab3 = st.tabs(["Top Models", "Model Comparison", "Value Analysis"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            top_models = df.groupby("Model")["Price"].mean().sort_values(ascending=False).head(10)
            sns.barplot(x=top_models.values, y=top_models.index, ax=ax, palette=PALETTE)
            ax.set_title("Top 10 Most Expensive Car Models")
            clean_axes(ax)
            st.pyplot(fig)

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
                title="Top Models by Brand",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Explore a Specific Model")
        model_choice = st.selectbox("Choose a model", sorted(df["Model"].unique()))
        model_df = df[df["Model"] == model_choice]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Listings", f"{len(model_df):,}")
        m2.metric("Avg. Price", money(model_df["Price"].mean()))
        m3.metric("Avg. Kilometres", f"{model_df['Kilometres'].mean():,.0f} km")
        m4.metric("Common Body Type", model_df["BodyType"].mode()[0] if not model_df.empty else "N/A")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(model_df["Price"], bins=15, kde=True, ax=ax, color=RED)
            ax.set_title(f"Price Distribution — {model_choice}")
            clean_axes(ax)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            year_counts = model_df["Year"].value_counts().sort_index()
            sns.lineplot(x=year_counts.index, y=year_counts.values, marker="o", ax=ax, color=NAVY)
            ax.set_title(f"Listings by Year — {model_choice}")
            clean_axes(ax)
            st.pyplot(fig)

        st.info(
            f"The **{model_choice}** has an average listed price of "
            f"{money(model_df['Price'].mean())} across {len(model_df):,} listings."
        )

    with tab2:
        st.write("Pick two or more models to compare side-by-side.")
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
            ).round(0)
            st.dataframe(summary, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                sns.boxplot(data=comp_df, x="Model", y="Price", ax=ax, palette=PALETTE)
                ax.set_title("Price Comparison")
                ax.tick_params(axis="x", rotation=20)
                clean_axes(ax)
                st.pyplot(fig)

            with c2:
                fig, ax = plt.subplots(figsize=(6.5, 4.5))
                sns.boxplot(data=comp_df, x="Model", y="Kilometres", ax=ax, palette=PALETTE)
                ax.set_title("Kilometres Comparison")
                ax.tick_params(axis="x", rotation=20)
                clean_axes(ax)
                st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(11, 4.3))
            trend = comp_df.groupby(["Year", "Model"])["Price"].mean().reset_index()
            sns.lineplot(data=trend, x="Year", y="Price", hue="Model", marker="o", ax=ax, palette=PALETTE)
            ax.set_title("Average Price by Year, by Model")
            clean_axes(ax)
            st.pyplot(fig)

            cheapest = summary["Avg_Price"].idxmin()
            priciest = summary["Avg_Price"].idxmax()
            st.info(
                f"**{priciest}** is the most expensive on average "
                f"({money(summary.loc[priciest, 'Avg_Price'])}), while **{cheapest}** is the "
                f"most affordable ({money(summary.loc[cheapest, 'Avg_Price'])})."
            )


    with tab3:
        st.write(
            "A simple value score highlights which listings offer the most car for the "
            "money: lower price, lower kilometres, and a newer year all raise the score."
        )

        def normalize(series, invert=False):
            """Scale a column to a 0-1 range so different units can be combined fairly."""
            spread = series.max() - series.min()
            if spread == 0:
                return pd.Series(0.5, index=series.index)
            scaled = (series - series.min()) / spread
            return 1 - scaled if invert else scaled

        value_df = df.copy()
        value_df["ValueScore"] = (
            0.5 * normalize(value_df["Price"], invert=True)
            + 0.3 * normalize(value_df["Kilometres"], invert=True)
            + 0.2 * normalize(value_df["Year"])
        ) * 100

        best_value = (
            value_df.groupby("Model")
            .agg(Avg_Price=("Price", "mean"), Avg_ValueScore=("ValueScore", "mean"), Listings=("Price", "count"))
            .query("Listings >= 5")
            .sort_values("Avg_ValueScore", ascending=False)
            .head(10)
        )

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            sns.barplot(x=best_value["Avg_ValueScore"], y=best_value.index, ax=ax, palette=PALETTE)
            ax.set_title("Top 10 Best-Value Models (min. 5 listings)")
            ax.set_xlabel("Value Score (0–100)")
            clean_axes(ax)
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(6.5, 5))
            age_price = df.groupby("CarAge")["Price"].mean().reset_index()
            sns.lineplot(data=age_price, x="CarAge", y="Price", marker="o", ax=ax, color=RED)
            ax.set_title("Average Price by Car Age")
            ax.set_xlabel("Car Age (years)")
            clean_axes(ax)
            st.pyplot(fig)

        top_value_model = best_value.index[0] if not best_value.empty else "N/A"
        st.info(
            f"**{top_value_model}** currently scores highest on value (balancing low price, "
            f"low kilometres, and newer year)."
        )

st.markdown("---")
st.caption("Car Market Intelligence Dashboard — built with Streamlit, Pandas, and Seaborn.")
