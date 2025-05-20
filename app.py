import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Finance Collections Dashboard", layout="wide")
st.title("📊 Finance Collections Dashboard")

# --- LOAD CLEANED DATA ---
file_path = "sample_data.xlsx"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    st.error("sample_data.xlsx not found. Please upload the cleaned Excel file.")
    st.stop()

# --- FILTERS ---
st.sidebar.header("1. Filter Data")
selected_status = st.sidebar.multiselect("Select Collection Status", options=df['collection_status'].unique(), default=list(df['collection_status'].unique()))
df = df[df['collection_status'].isin(selected_status)]

min_date, max_date = df['invoice_post_date'].min(), df['invoice_post_date'].max()
selected_range = st.sidebar.date_input("Invoice Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df['invoice_post_date'] >= pd.to_datetime(selected_range[0])) & (df['invoice_post_date'] <= pd.to_datetime(selected_range[1]))]

# --- DATA PREP ---
df['collection_date'] = pd.to_datetime(df['collection_date'], errors='coerce')
df['invoice_post_date'] = pd.to_datetime(df['invoice_post_date'], errors='coerce')
df['collection_month'] = df['collection_date'].dt.to_period('M')
df['invoiced_month'] = df['invoice_post_date'].dt.to_period('M')

monthly_collected = df[df['collection_status'] == 'Paid'].groupby('collection_month')['order_amount'].sum()
monthly_invoiced = df[df['invoice'].str.startswith('II', na=False)].groupby('invoiced_month')['order_amount'].sum()

monthly_summary = pd.merge(monthly_invoiced.reset_index(name='invoiced_amount'),
                           monthly_collected.reset_index(name='collected_amount'),
                           left_on='invoiced_month', right_on='collection_month', how='outer')

monthly_summary = monthly_summary.rename(columns={'invoiced_month': 'month'}).drop(columns=['collection_month'])
monthly_summary = monthly_summary.fillna(0)
monthly_summary['month'] = monthly_summary['month'].astype(str)

# --- PLOT: Invoiced vs Collected ---
st.subheader("Monthly Invoiced vs Collected Amounts")
x = np.arange(len(monthly_summary))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - width/2, monthly_summary['invoiced_amount'], width, label='Invoiced', color='skyblue')
ax.bar(x + width/2, monthly_summary['collected_amount'], width, label='Collected', color='seagreen')
ax.set_xticks(x)
ax.set_xticklabels(monthly_summary['month'], rotation=45)
ax.set_ylabel("Amount (EUR)")
ax.set_title("Invoiced vs Collected")
ax.legend()
st.pyplot(fig)

# --- NET CASHFLOW ---
st.subheader("Monthly Net Cash Flow")
monthly_summary['net_cashflow'] = monthly_summary['collected_amount'] - monthly_summary['invoiced_amount']
fig_cf, ax_cf = plt.subplots(figsize=(12, 5))
colors = monthly_summary['net_cashflow'].apply(lambda x: 'skyblue' if x >= 0 else '#FAF3A0')
ax_cf.bar(monthly_summary['month'], monthly_summary['net_cashflow'], color=colors)
ax_cf.axhline(0, linestyle='--', color='black', linewidth=0.8)
ax_cf.set_title('Monthly Net Cash Flow (Collected - Invoiced)')
ax_cf.set_ylabel('Net Cash Flow (€)')
ax_cf.set_xlabel('Month')
ax_cf.tick_params(axis='x', rotation=45)
st.pyplot(fig_cf)

# --- CUMULATIVE DEBT ---
st.subheader("Cumulative Uncollected Debt Over Time")
monthly_summary['cum_invoiced'] = monthly_summary['invoiced_amount'].cumsum()
monthly_summary['cum_collected'] = monthly_summary['collected_amount'].cumsum()
monthly_summary['cumulative_debt'] = monthly_summary['cum_invoiced'] - monthly_summary['cum_collected']
fig_debt, ax_debt = plt.subplots(figsize=(12, 5))
ax_debt.plot(monthly_summary['month'], monthly_summary['cumulative_debt'], marker='o', color='darkorange', label='Cumulative Debt')
ax_debt.fill_between(monthly_summary['month'], monthly_summary['cumulative_debt'], color='#ffdd99', alpha=0.5)
ax_debt.set_title('Cumulative Uncollected Debt Over Time')
ax_debt.set_xlabel('Month')
ax_debt.set_ylabel('Cumulative Debt (€)')
ax_debt.axhline(0, linestyle='--', color='black', linewidth=0.8)
ax_debt.legend()
ax_debt.tick_params(axis='x', rotation=45)
st.pyplot(fig_debt)

# --- FORECAST: Prophet on Collected Amount ---
st.subheader("Forecast: Monthly Collected Amount")
df_prophet = monthly_summary[['month', 'collected_amount']].copy()
df_prophet.columns = ['ds', 'y']
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)
fig1 = model.plot(forecast)
st.pyplot(fig1)

# --- FORECAST: Prophet on DSO ---
st.subheader("Forecast: Days Sales Outstanding (DSO)")
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

# --- COLLECTION RATE COHORT ---
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
ax_cohort.set_title('Monthly Collection Rate by Invoice Month (Cohort)')
ax_cohort.set_xlabel('Invoice Month')
ax_cohort.set_ylabel('Collection Rate')
ax_cohort.set_ylim(0, 1.1)
ax_cohort.axhline(1.0, linestyle='--', color='gray', linewidth=0.8)
ax_cohort.tick_params(axis='x', rotation=45)
st.pyplot(fig_cohort)

# --- DAYS OUTSTANDING ---
st.subheader("Average Days to Payment vs. Days Outstanding")
today = pd.to_datetime('today')
df['effective_end_date'] = df['collection_date'].fillna(today)
df['days_to_payment'] = (df['collection_date'] - df['invoice_post_date']).dt.days
df['days_outstanding'] = (df['effective_end_date'] - df['invoice_post_date']).dt.days
monthly_metrics = df.groupby('invoice_month').agg(
    avg_days_to_payment=('days_to_payment', 'mean'),
    avg_days_outstanding=('days_outstanding', 'mean')
).reset_index()
fig_days, ax_days = plt.subplots(figsize=(12, 6))
ax_days.plot(monthly_metrics['invoice_month'].astype(str), monthly_metrics['avg_days_to_payment'], label='Avg Days to Payment', marker='o')
ax_days.plot(monthly_metrics['invoice_month'].astype(str), monthly_metrics['avg_days_outstanding'], label='Avg Days Outstanding', marker='o')
ax_days.set_title('Average Days to Payment vs. Days Outstanding')
ax_days.set_xlabel('Invoice Month')
ax_days.set_ylabel('Days')
ax_days.legend()
ax_days.tick_params(axis='x', rotation=45)
st.pyplot(fig_days)

# --- AGING BUCKETS ---
st.subheader("Unpaid Amount by Aging Bucket")
bins = [0, 31, 61, 91, 121, float('inf')]
labels = ['0–30', '31–60', '61–90', '91–120', '120+']
df['aging_bucket'] = pd.cut(df['days_outstanding'], bins=bins, labels=labels, right=False)
aging_summary = df[df['collection_date'].isna()].groupby('aging_bucket')['order_amount'].sum().reset_index(name='unpaid_amount')
fig_aging, ax_aging = plt.subplots(figsize=(8, 5))
ax_aging.bar(aging_summary['aging_bucket'], aging_summary['unpaid_amount'], color='tomato')
ax_aging.set_title('Unpaid Amount by Aging Bucket')
ax_aging.set_xlabel('Days Outstanding')
ax_aging.set_ylabel('Amount (€)')
st.pyplot(fig_aging)
