
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("Finance Collections Dashboard")

# Load data
file_path = "sample_data.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    st.error("sample_data.xlsx not found. Please upload the cleaned Excel file.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filter Data")
selected_status = st.sidebar.multiselect("Collection Status", options=df['collection_status'].unique(), default=list(df['collection_status'].unique()))
df = df[df['collection_status'].isin(selected_status)]

min_date, max_date = df['invoice_post_date'].min(), df['invoice_post_date'].max()
selected_range = st.sidebar.date_input("Invoice Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df['invoice_post_date'] >= pd.to_datetime(selected_range[0])) & (df['invoice_post_date'] <= pd.to_datetime(selected_range[1]))]

st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub Repo](https://github.com/negroniO/finance_dashboard)")

# Convert dates
df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
df['invoice_post_date'] = pd.to_datetime(df['invoice_post_date'], errors='coerce')
df['collection_month'] = df['collection_date'].dt.to_period('M')
df['invoiced_month'] = df['invoice_post_date'].dt.to_period('M')

# Summary Data
monthly_collected = df[df['collection_status'] == 'Paid'].groupby('collection_month')['order_amount'].sum()
monthly_invoiced = df[df['invoice'].str.startswith('II', na=False)].groupby('invoiced_month')['order_amount'].sum()

monthly_summary = pd.merge(monthly_invoiced.reset_index(name='invoiced_amount'),
                           monthly_collected.reset_index(name='collected_amount'),
                           left_on='invoiced_month', right_on='collection_month', how='outer')
monthly_summary = monthly_summary.rename(columns={'invoiced_month': 'month'}).drop(columns=['collection_month'])
monthly_summary = monthly_summary.fillna(0)
monthly_summary['month'] = monthly_summary['month'].astype(str)

# KPIs
st.markdown("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Invoiced", f"€{df['order_amount'].sum():,.0f}")
col2.metric("Total Collected", f"€{df[df['collection_status'] == 'Paid']['order_amount'].sum():,.0f}")
monthly_summary['DSO'] = ((monthly_summary['invoiced_amount'].cumsum() - monthly_summary['collected_amount'].cumsum()) / monthly_summary['invoiced_amount'].replace(0, np.nan) * 30).round(2)
col3.metric("Avg DSO", f"{monthly_summary['DSO'].mean():.1f} days")

# Tabs
tab1, tab2 = st.tabs(["Performance", "Forecasts"])

with tab1:
    st.markdown("Invoiced vs Collected (Interactive)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly_summary['month'], y=monthly_summary['invoiced_amount'], name='Invoiced', marker_color='skyblue'))
    fig.add_trace(go.Bar(x=monthly_summary['month'], y=monthly_summary['collected_amount'], name='Collected', marker_color='seagreen'))
    fig.update_layout(barmode='group', xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Net Cash Flow")
    monthly_summary['net_cashflow'] = monthly_summary['collected_amount'] - monthly_summary['invoiced_amount']
    net_fig = px.bar(monthly_summary, x='month', y='net_cashflow', title='Monthly Net Cash Flow (Collected - Invoiced)',
                     color=monthly_summary['net_cashflow'] >= 0, color_discrete_map={True: 'seagreen', False: 'salmon'})
    net_fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(net_fig, use_container_width=True)

    st.markdown("Cumulative Uncollected Debt")
    monthly_summary['cum_invoiced'] = monthly_summary['invoiced_amount'].cumsum()
    monthly_summary['cum_collected'] = monthly_summary['collected_amount'].cumsum()
    monthly_summary['cumulative_debt'] = monthly_summary['cum_invoiced'] - monthly_summary['cum_collected']
    fig_debt = px.area(monthly_summary, x='month', y='cumulative_debt', title="Cumulative Uncollected Debt Over Time", color_discrete_sequence=["orange"])
    st.plotly_chart(fig_debt, use_container_width=True)

with tab2:
    st.markdown("Forecast Settings")
    forecast_period = st.slider("Forecast Horizon (months)", min_value=3, max_value=24, value=6)
    training_months = st.slider("Train Model On Most Recent (months)", min_value=6, max_value=len(monthly_summary), value=len(monthly_summary))

    # Collected Amount Forecast
    st.markdown("Collected Amount Forecast")
    df_prophet = monthly_summary[['month', 'collected_amount']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.tail(training_months)

    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_period, freq='M')
    forecast = model.predict(future)

    fig1 = px.line(forecast, x='ds', y='yhat', title='Forecasted Monthly Collected Amount')
    fig1.add_traces([
        go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(dash='dot'), name='Upper'),
        go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(dash='dot'), name='Lower')
    ])
    st.plotly_chart(fig1, use_container_width=True)

    # DSO Forecast
    st.markdown("#### DSO Forecast")
    df_prophet_dso = monthly_summary[['month', 'DSO']].copy()
    df_prophet_dso.columns = ['ds', 'y']
    df_prophet_dso['ds'] = pd.to_datetime(df_prophet_dso['ds'])
    df_prophet_dso = df_prophet_dso.tail(training_months)

    model_dso = Prophet()
    model_dso.fit(df_prophet_dso)
    future_dso = model_dso.make_future_dataframe(periods=forecast_period, freq='M')
    forecast_dso = model_dso.predict(future_dso)

    fig2 = px.line(forecast_dso, x='ds', y='yhat', title='Forecasted DSO')
    fig2.add_traces([
        go.Scatter(x=forecast_dso['ds'], y=forecast_dso['yhat_upper'], mode='lines', line=dict(dash='dot'), name='Upper'),
        go.Scatter(x=forecast_dso['ds'], y=forecast_dso['yhat_lower'], mode='lines', line=dict(dash='dot'), name='Lower')
    ])
    st.plotly_chart(fig2, use_container_width=True)
