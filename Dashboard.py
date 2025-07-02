# Leading Causes of Death Dashboard using ARIMA

import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Leading Causes of Death Dashboard", layout="wide")

# Session initialization
if 'data_uploaded' not in st.session_state:
    st.session_state['data_uploaded'] = False
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Upload Dataset
if not st.session_state['data_uploaded']:
    st.markdown("<h1 style='text-align:center;'>📤 Upload Your CSV Dataset</h1>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align:center; font-size:18px; color:#555;'>
        Please upload a valid CSV with columns like <code>Year</code>, <code>Cause Name</code>, <code>State</code>, <code>Deaths</code>.
        </div><br><br>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = {"Year", "Cause Name", "State", "Deaths"}
            if not required_cols.issubset(df.columns):
                st.error("The CSV is missing one or more required columns.")
            else:
                df = df[~df['State'].isin(['United States', 'District of Columbia'])]
                if 'Age-adjusted Death Rate' in df.columns:
                    df.drop(columns=['Age-adjusted Death Rate'], inplace=True)
                df.columns = [col.strip().title().replace("-", " ") for col in df.columns]

                st.session_state['data'] = df
                st.session_state['data_uploaded'] = True
                st.success("Upload successful! Redirecting to dashboard...")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    st.stop()

# Dashboard
df = st.session_state['data']
st.markdown("<h1 style='text-align:center;'> Leading Causes of Death Dashboard using ARIMA</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Historical Data Filter")
state_options = ["All States"] + sorted(df['State'].unique())
year_options = sorted(df['Year'].unique())

selected_state = st.sidebar.selectbox("State", state_options)
selected_year = st.sidebar.selectbox("Year", year_options)
top_option = st.sidebar.radio("Causes to Display (Bar Chart)", ["Top 3 Causes", "All Causes"])

# Filters
filtered_df = df.copy()
if selected_state != "All States":
    filtered_df = filtered_df[filtered_df['State'] == selected_state]
filtered_df = filtered_df[filtered_df['Year'] == selected_year]
filtered_df = filtered_df[filtered_df['Cause Name'].str.lower() != "all causes"]

# Summary for Chart
summary = (
    filtered_df.groupby("Cause Name")["Deaths"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

if top_option == "Top 3 Causes":
    summary = summary.head(3)

total_deaths = summary["Deaths"].sum()
summary["Percentage"] = (summary["Deaths"] / total_deaths * 100).round(2)
summary["Label"] = summary.apply(lambda row: f"{row['Deaths']:,} ({row['Percentage']}%)", axis=1)
summary["Cause Name"] = summary["Cause Name"].apply(lambda x: "<br>".join(x.split(" ")))

# Side-by-side layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='font-weight: bold; font-size: 24px;'>Dataset Viewer</h3>", unsafe_allow_html=True)

    search_query = st.text_input("Search Dataset", "")

    df_display = filtered_df.copy()
    if search_query:
        search_query_lower = search_query.lower()
        df_display = df_display[df_display.apply(
            lambda row: row.astype(str).str.lower().str.contains(search_query_lower).any(), axis=1
        )]

    df_display["Cause Name"] = df_display["Cause Name"].apply(
        lambda x: '<br>'.join(x[i:i+30] for i in range(0, len(x), 30)) if len(x) > 30 else x
    )
    styled_html = df_display.to_html(escape=False, index=False)

    st.markdown(f"""
        <div style="height:500px; overflow-y:auto; border:1px solid #ccc; padding:10px; border-radius:10px;">
            {styled_html}
        </div>
    """, unsafe_allow_html=True)

    csv = df_display.to_csv(index=False).encode("utf-8")
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.download_button("Download Filtered Data as CSV", csv, file_name="filtered_dataset.csv", mime="text/csv")

with col2:
    st.markdown("<h3 style='font-weight: bold; font-size: 24px;'>Death by Cause (Bar Chart)</h3>", unsafe_allow_html=True)

    fig = px.bar(
        summary,
        x="Deaths",
        y="Cause Name",
        text="Label",
        orientation="h",
        color="Cause Name",  # Add this line
        title=f"{top_option} in {selected_state}, {selected_year}<br>Total Deaths: {total_deaths:,}",
        labels={"Deaths": "Number of Deaths", "Cause Name": "Cause"},
        height=500
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(yaxis_title=None, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Forecasting with ARIMA model
st.sidebar.markdown("---")
st.sidebar.title("Prediction of Causes")
predict_state = st.sidebar.selectbox("Forecast for State", sorted(df['State'].unique()))
years_ahead = st.sidebar.selectbox("Years to Predict (after 2017)", [1, 2, 3, 5, 10])
show_all_toggle = st.sidebar.checkbox("Show all causes (line chart)", value=False)

if st.sidebar.button("Run Forecast"):
    with st.spinner("Running ARIMA models, please wait..."):
        df_pred = df.copy()
        df_pred = df_pred[df_pred['Cause Name'].str.lower() != "all causes"]
        df_pred = df_pred[df_pred['Year'] <= 2017]

        future_years = list(range(2018, 2018 + years_ahead))
        predictions = []
        causes = df_pred['Cause Name'].unique()
        skipped_causes = []
        progress = st.progress(0)

        for i, cause in enumerate(causes):
            sub_df = df_pred[(df_pred['State'] == predict_state) & (df_pred['Cause Name'] == cause)].sort_values('Year')
            if len(sub_df) < 3:
                skipped_causes.append(cause)
                progress.progress((i + 1) / len(causes))
                continue

            try:
                model = auto_arima(sub_df['Deaths'], seasonal=False, suppress_warnings=True, error_action="ignore")
                forecast = model.predict(n_periods=years_ahead)
                forecast_values = forecast.values if hasattr(forecast, "values") else forecast

                for j, year in enumerate(future_years):
                    predictions.append({
                        'Year': year,
                        'State': predict_state,
                        'Cause Name': cause,
                        'Predicted Deaths': max(0, round(forecast_values[j]))
                    })
            except Exception as e:
                skipped_causes.append(cause)
            progress.progress((i + 1) / len(causes))

    st.success("Forecasting complete!")
    if skipped_causes:
        st.warning(f" Skipped {len(skipped_causes)} causes due to insufficient data or model failure.")
        st.warning(f" Skipped {len(skipped_causes)} causes due to insufficient data or model failure.")

    pred_df = pd.DataFrame(predictions)

    if pred_df.empty:
        st.error("No predictions available.")
    else:
        idx = pred_df.groupby('Year')['Predicted Deaths'].idxmax()
        top_causes = pred_df.loc[idx].reset_index(drop=True)

        st.subheader(f"Top Predicted Causes in {predict_state}")
        fig_forecast = px.bar(
            top_causes,
            x="Year",
            y="Predicted Deaths",
            color="Cause Name",
            title=f"Top Predicted Causes in {predict_state}",
            height=500
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        if show_all_toggle:
            st.subheader(f"All Predicted Causes in {predict_state}")
            fig_all = px.line(
                pred_df,
                x="Year",
                y="Predicted Deaths",
                color="Cause Name",
                markers=True,
                title="Predicted Deaths by Cause",
                height=600
            )
            fig_all.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_all, use_container_width=True)

# To track if new file has been processed
if 'new_file_processed' not in st.session_state:
    st.session_state['new_file_processed'] = False

st.sidebar.markdown("---")
st.sidebar.title("Upload New Dataset (CSV)")
new_file = st.sidebar.file_uploader("", type=["csv"], key="new_file")

if new_file and not st.session_state['new_file_processed']:
    try:
        df_new = pd.read_csv(new_file)
        required_cols = {"Year", "Cause Name", "State", "Deaths"}
        if not required_cols.issubset(df_new.columns):
            st.sidebar.error("Missing required columns.")
        else:
            df_new = df_new[~df_new['State'].isin(['United States', 'District of Columbia'])]
            if 'Age-adjusted Death Rate' in df_new.columns:
                df_new.drop(columns=['Age-adjusted Death Rate'], inplace=True)
            df_new.columns = [col.strip().title().replace("-", " ") for col in df_new.columns]

            st.session_state['data'] = df_new
            st.session_state['new_file_processed'] = True  # Set flag to avoid loop
            st.sidebar.success("Dataset updated!")
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

# Reset if the user removes the file
if not new_file:
    st.session_state['new_file_processed'] = False
    
