# Leading Causes of Death Dashboard using ARIMA

import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Leading Causes of Death Dashboard", layout="wide")

# Session Initialization
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






# Khisshore
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

# Side-by-Side Layout
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
