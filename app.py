import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("📊 Finance Collections Dashboard")

# Load data
file_path = "sample_data.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    st.error("sample_data.xlsx not found. Please upload the cleaned Excel file.")
    st.stop()

# Granularity logic
st.sidebar.subheader("Time Granularity")
granularity = st.sidebar.radio("Group Data By", ["Month", "Quarter", "Year"], index=0)

def format_period(date_col, level):
    if level == "Month":
        return date_col.dt.to_period("M").astype(str)
    elif level == "Quarter":
        return date_col.dt.to_period("Q").astype(str)
    elif level == "Year":
        return date_col.dt.to_period("Y").astype(str)

# Filter
df['invoice_post_date'] = pd.to_datetime(df['invoice_post_date'], errors='coerce')
df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
df = df.dropna(subset=['invoice_post_date'])

st.sidebar.header("Filter Data")
selected_status = st.sidebar.multiselect(
    "Collection Status", 
    options=df['collection_status'].dropna().unique(), 
    default=list(df['collection_status'].dropna().unique())
)
df = df[df['collection_status'].isin(selected_status)]

min_date, max_date = df['invoice_post_date'].min(), df['invoice_post_date'].max()
selected_range = st.sidebar.date_input("Invoice Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df['invoice_post_date'] >= pd.to_datetime(selected_range[0])) & 
        (df['invoice_post_date'] <= pd.to_datetime(selected_range[1]))]

st.sidebar.markdown("---")
st.sidebar.markdown("[GitHub Repo](https://github.com/negroniO/finance_dashboard)")

# Period columns
df['invoice_period'] = format_period(df['invoice_post_date'], granularity)
df['collection_period'] = format_period(df['collection_date'], granularity)

# Summary Data
monthly_collected = df[df['collection_status'] == 'Paid'].groupby('collection_period')['order_amount'].sum()
monthly_invoiced = df[df['invoice'].str.startswith('II', na=False)].groupby('invoice_period')['order_amount'].sum()

monthly_summary = pd.merge(
    monthly_invoiced.reset_index(name='invoiced_amount'),
    monthly_collected.reset_index(name='collected_amount'),
    left_on='invoice_period', right_on='collection_period', how='outer'
)

monthly_summary = monthly_summary.rename(columns={'invoice_period': 'period'}).drop(columns=['collection_period'])
monthly_summary = monthly_summary.fillna(0)
monthly_summary['period'] = monthly_summary['period'].astype(str)

# KPIs
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Invoiced", f"€{df['order_amount'].sum():,.0f}")
col2.metric("Total Collected", f"€{df[df['collection_status'] == 'Paid']['order_amount'].sum():,.0f}")
monthly_summary['DSO'] = ((monthly_summary['invoiced_amount'].cumsum() - monthly_summary['collected_amount'].cumsum()) / 
                          monthly_summary['invoiced_amount'].replace(0, np.nan) * 30).round(2)
col3.metric("Avg DSO", f"{monthly_summary['DSO'].mean():.1f} days")

# Tabs
tab1, tab2, tab3 = st.tabs(["Performance", "Forecasts", "Risk Insights"])

with tab1:
    st.markdown(f"### Invoiced vs Collected ({granularity})")
    fig = go.Figure(data=[
        go.Bar(name="Invoiced", x=monthly_summary['period'], y=monthly_summary['invoiced_amount'], 
               hovertemplate='Invoiced: €%{y:,.2f}<extra></extra>', marker_color='skyblue'),
        go.Bar(name="Collected", x=monthly_summary['period'], y=monthly_summary['collected_amount'], 
               hovertemplate='Collected: €%{y:,.2f}<extra></extra>', marker_color='seagreen')
    ])
    fig.update_layout(barmode='group', xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Net Cash Flow")
    monthly_summary['net_cashflow'] = monthly_summary['collected_amount'] - monthly_summary['invoiced_amount']
    monthly_summary['cashflow_label'] = np.where(monthly_summary['net_cashflow'] >= 0, 'Positive', 'Negative')
    fig_cf = px.bar(
        monthly_summary,
        x='period',
        y='net_cashflow',
        color='cashflow_label',
        color_discrete_map={'Positive': 'seagreen', 'Negative': 'salmon'},
        title='Net Cash Flow',
        hover_data={'net_cashflow': ':.2f'}
    )
    fig_cf.update_traces(hovertemplate='Net Cashflow: €%{y:,.2f}')
    fig_cf.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cf, use_container_width=True)

    st.markdown("### Cumulative Uncollected Debt")
    monthly_summary['cum_invoiced'] = monthly_summary['invoiced_amount'].cumsum()
    monthly_summary['cum_collected'] = monthly_summary['collected_amount'].cumsum()
    monthly_summary['cumulative_debt'] = monthly_summary['cum_invoiced'] - monthly_summary['cum_collected']
    fig_debt = px.area(monthly_summary, x='period', y='cumulative_debt',
                       title="Cumulative Uncollected Debt", hover_data={'cumulative_debt': ':.2f'})
    fig_debt.update_traces(hovertemplate='Debt: €%{y:,.2f}')
    st.plotly_chart(fig_debt, use_container_width=True)

with tab2:
    st.markdown("### Forecast Settings")
    forecast_period = st.slider("Forecast Horizon (months)", min_value=3, max_value=24, value=6)
    training_months = st.slider("Training Window", min_value=6, max_value=len(monthly_summary), value=len(monthly_summary))

    st.markdown("### Collected Amount Forecast")
    df_prophet = monthly_summary[['period', 'collected_amount']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.tail(training_months)
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=forecast_period, freq='M')
    forecast = model.predict(future)
    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Collected Amount')
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper', line=dict(dash='dot'))
    fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower', line=dict(dash='dot'))
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.markdown("### DSO Forecast")
    df_dso = monthly_summary[['period', 'DSO']].copy()
    df_dso.columns = ['ds', 'y']
    df_dso['ds'] = pd.to_datetime(df_dso['ds'])
    df_dso = df_dso.tail(training_months)
    model_dso = Prophet()
    model_dso.fit(df_dso)
    future_dso = model_dso.make_future_dataframe(periods=forecast_period, freq='M')
    forecast_dso = model_dso.predict(future_dso)
    fig_dso = px.line(forecast_dso, x='ds', y='yhat', title='Forecasted DSO')
    fig_dso.add_scatter(x=forecast_dso['ds'], y=forecast_dso['yhat_upper'], mode='lines', name='Upper', line=dict(dash='dot'))
    fig_dso.add_scatter(x=forecast_dso['ds'], y=forecast_dso['yhat_lower'], mode='lines', name='Lower', line=dict(dash='dot'))
    st.plotly_chart(fig_dso, use_container_width=True)

with tab3:
    st.markdown("### Collection Rate by Invoice Period")
    df['invoice_period'] = format_period(df['invoice_post_date'], granularity)
    paid_df = df[df['collection_status'] == 'Paid'].copy()
    paid_df['invoice_period'] = format_period(paid_df['invoice_post_date'], granularity)

    monthly_inv = df.groupby('invoice_period')['order_amount'].sum()
    monthly_col = paid_df.groupby('invoice_period')['order_amount'].sum()

    cohort = pd.DataFrame({'invoiced_amount': monthly_inv, 'collected_amount': monthly_col}).fillna(0)
    cohort['collection_rate'] = (cohort['collected_amount'] / cohort['invoiced_amount']).round(2)
    cohort = cohort.reset_index()
    fig_cohort = px.bar(
        cohort,
        x='invoice_period',
        y='collection_rate',
        title="Collection Rate by Invoice Period",
        hover_data={'collection_rate': ':.2f'},
        color_discrete_sequence=['seagreen']
    )
    st.plotly_chart(fig_cohort, use_container_width=True)

    st.markdown("### Days to Payment vs Outstanding")
    today = pd.to_datetime('today')
    df['effective_end_date'] = df['collection_date'].fillna(today)
    df['days_to_payment'] = (df['collection_date'] - df['invoice_post_date']).dt.days
    df['days_outstanding'] = (df['effective_end_date'] - df['invoice_post_date']).dt.days
    metrics = df.groupby('invoice_period').agg(
        avg_days_to_payment=('days_to_payment', 'mean'),
        avg_days_outstanding=('days_outstanding', 'mean')
    ).reset_index()
    fig_days = go.Figure()
    fig_days.add_trace(go.Scatter(x=metrics['invoice_period'], y=metrics['avg_days_to_payment'], name='To Payment', mode='lines+markers'))
    fig_days.add_trace(go.Scatter(x=metrics['invoice_period'], y=metrics['avg_days_outstanding'], name='Outstanding', mode='lines+markers'))
    fig_days.update_layout(title="Average Days to Payment vs. Outstanding", xaxis_title="Period", yaxis_title="Days")
    st.plotly_chart(fig_days, use_container_width=True)

    st.markdown("### Aging Buckets")
    bins = [0, 31, 61, 91, 121, float('inf')]
    labels = ['0–30', '31–60', '61–90', '91–120', '120+']
    df['aging_bucket'] = pd.cut(df['days_outstanding'], bins=bins, labels=labels, right=False)
    aging = df[df['collection_date'].isna()].groupby('aging_bucket')['order_amount'].sum().reset_index(name='unpaid_amount')
    fig_aging = px.bar(aging, x='aging_bucket', y='unpaid_amount', title="Unpaid Amount by Aging Bucket",
                       hover_data={'unpaid_amount': ':.2f'}, color_discrete_sequence=['tomato'])
    st.plotly_chart(fig_aging, use_container_width=True)
