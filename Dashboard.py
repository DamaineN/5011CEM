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

