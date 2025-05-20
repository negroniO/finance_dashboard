
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from datetime import datetime

st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("📊 Finance Collections Dashboard")

# Load data
file_path = "sample_data.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    st.error("sample_data.xlsx not found. Please upload the cleaned Excel file.")
    st.stop()

# Filters
st.sidebar.header("🔍 Filter Data")
selected_status = st.sidebar.multiselect("Collection Status", options=df['collection_status'].unique(), default=list(df['collection_status'].unique()))
df = df[df['collection_status'].isin(selected_status)]

min_date, max_date = df['invoice_post_date'].min(), df['invoice_post_date'].max()
selected_range = st.sidebar.date_input("Invoice Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df['invoice_post_date'] >= pd.to_datetime(selected_range[0])) & (df['invoice_post_date'] <= pd.to_datetime(selected_range[1]))]

# Date handling
df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
df['invoice_post_date'] = pd.to_datetime(df['invoice_post_date'], errors='coerce')
df['collection_month'] = df['collection_date'].dt.to_period('M')
df['invoiced_month'] = df['invoice_post_date'].dt.to_period('M')

# Summary
monthly_collected = df[df['collection_status'] == 'Paid'].groupby('collection_month')['order_amount'].sum()
monthly_invoiced = df[df['invoice'].str.startswith('II', na=False)].groupby('invoiced_month')['order_amount'].sum()

monthly_summary = pd.merge(monthly_invoiced.reset_index(name='invoiced_amount'),
                           monthly_collected.reset_index(name='collected_amount'),
                           left_on='invoiced_month', right_on='collection_month', how='outer')

monthly_summary = monthly_summary.rename(columns={'invoiced_month': 'month'}).drop(columns=['collection_month'])
monthly_summary = monthly_summary.fillna(0)
monthly_summary['month'] = monthly_summary['month'].astype(str)

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔮 Forecasts", "📊 Risk Insights"])

with tab1:
    st.subheader("Monthly Invoiced vs Collected Amounts")
    x = np.arange(len(monthly_summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, monthly_summary['invoiced_amount'], width, label='Invoiced', color='skyblue')
    ax.bar(x + width/2, monthly_summary['collected_amount'], width, label='Collected', color='seagreen')
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_summary['month'], rotation=45)
    ax.set_ylabel("Amount (EUR)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Monthly Net Cash Flow")
    monthly_summary['net_cashflow'] = monthly_summary['collected_amount'] - monthly_summary['invoiced_amount']
    fig_cf, ax_cf = plt.subplots(figsize=(12, 5))
    colors = monthly_summary['net_cashflow'].apply(lambda x: 'skyblue' if x >= 0 else '#FAF3A0')
    ax_cf.bar(monthly_summary['month'], monthly_summary['net_cashflow'], color=colors)
    ax_cf.axhline(0, linestyle='--', color='black')
    ax_cf.set_ylabel("Net Cash Flow (€)")
    ax_cf.set_xlabel("Month")
    ax_cf.set_title("Net Cash Flow (Collected - Invoiced)")
    ax_cf.tick_params(axis='x', rotation=45)
    st.pyplot(fig_cf)

    st.subheader("Cumulative Uncollected Debt")
    monthly_summary['cum_invoiced'] = monthly_summary['invoiced_amount'].cumsum()
    monthly_summary['cum_collected'] = monthly_summary['collected_amount'].cumsum()
    monthly_summary['cumulative_debt'] = monthly_summary['cum_invoiced'] - monthly_summary['cum_collected']
    fig_debt, ax_debt = plt.subplots(figsize=(12, 5))
    ax_debt.plot(monthly_summary['month'], monthly_summary['cumulative_debt'], marker='o', color='darkorange')
    ax_debt.fill_between(monthly_summary['month'], monthly_summary['cumulative_debt'], color='#ffdd99', alpha=0.5)
    ax_debt.axhline(0, linestyle='--', color='black')
    ax_debt.set_ylabel("Cumulative Debt (€)")
    ax_debt.tick_params(axis='x', rotation=45)
    st.pyplot(fig_debt)

with tab2:
    st.subheader("Forecast: Collected Amount (Prophet)")
    df_prophet = monthly_summary[['month', 'collected_amount']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=6, freq='M')
    forecast = model.predict(future)
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("Forecast: DSO")
    monthly_summary['DSO'] = ((monthly_summary['cumulative_debt'] / monthly_summary['invoiced_amount'].replace(0, np.nan)) * 30).round(2)
    df_prophet_dso = monthly_summary[['month', 'DSO']].copy()
    df_prophet_dso.columns = ['ds', 'y']
    df_prophet_dso['ds'] = pd.to_datetime(df_prophet_dso['ds'])
    model_dso = Prophet()
    model_dso.fit(df_prophet_dso)
    future_dso = model_dso.make_future_dataframe(periods=6, freq='M')
    forecast_dso = model_dso.predict(future_dso)
    fig2 = model_dso.plot(forecast_dso)
    st.pyplot(fig2)

with tab3:
    st.subheader("Collection Rate by Invoice Month")
    paid_df = df[df['collection_status'] == 'Paid'].copy()
    df['invoice_month'] = df['invoice_post_date'].dt.to_period('M')
    paid_df['invoice_month'] = paid_df['invoice_post_date'].dt.to_period('M')
    monthly_invoiced = df.groupby('invoice_month')['order_amount'].sum()
    monthly_collected = paid_df.groupby('invoice_month')['order_amount'].sum()
    cohort_summary = pd.DataFrame({
        'invoiced_amount': monthly_invoiced,
        'collected_amount': monthly_collected
    }).fillna(0)
    cohort_summary['collection_rate'] = (cohort_summary['collected_amount'] / cohort_summary['invoiced_amount']).round(2)
    cohort_summary_plot = cohort_summary.reset_index()
    fig_cohort, ax_cohort = plt.subplots(figsize=(12, 6))
    ax_cohort.bar(cohort_summary_plot['invoice_month'].astype(str), cohort_summary_plot['collection_rate'], color='seagreen')
    ax_cohort.axhline(1.0, linestyle='--', color='gray')
    ax_cohort.set_ylim(0, 1.1)
    ax_cohort.set_ylabel("Collection Rate")
    ax_cohort.tick_params(axis='x', rotation=45)
    st.pyplot(fig_cohort)

    st.subheader("Avg Days to Payment vs Outstanding")
    today = pd.to_datetime('today')
    df['effective_end_date'] = df['collection_date'].fillna(today)
    df['days_to_payment'] = (df['collection_date'] - df['invoice_post_date']).dt.days
    df['days_outstanding'] = (df['effective_end_date'] - df['invoice_post_date']).dt.days
    monthly_metrics = df.groupby('invoice_month').agg(
        avg_days_to_payment=('days_to_payment', 'mean'),
        avg_days_outstanding=('days_outstanding', 'mean')
    ).reset_index()
    fig_days, ax_days = plt.subplots(figsize=(12, 6))
    ax_days.plot(monthly_metrics['invoice_month'].astype(str), monthly_metrics['avg_days_to_payment'], label='To Payment', marker='o')
    ax_days.plot(monthly_metrics['invoice_month'].astype(str), monthly_metrics['avg_days_outstanding'], label='Outstanding', marker='o')
    ax_days.set_ylabel("Days")
    ax_days.legend()
    ax_days.tick_params(axis='x', rotation=45)
    st.pyplot(fig_days)

    st.subheader("Unpaid Amount by Aging Bucket")
    bins = [0, 31, 61, 91, 121, float('inf')]
    labels = ['0–30', '31–60', '61–90', '91–120', '120+']
    df['aging_bucket'] = pd.cut(df['days_outstanding'], bins=bins, labels=labels, right=False)
    aging_summary = df[df['collection_date'].isna()].groupby('aging_bucket')['order_amount'].sum().reset_index(name='unpaid_amount')
    fig_aging, ax_aging = plt.subplots(figsize=(8, 5))
    ax_aging.bar(aging_summary['aging_bucket'], aging_summary['unpaid_amount'], color='tomato')
    ax_aging.set_ylabel("Amount (€)")
    st.pyplot(fig_aging)
