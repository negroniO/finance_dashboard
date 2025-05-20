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

# --- MONTHLY SUMMARY ---
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
monthly_summary['cumulative_debt'] = monthly_summary['invoiced_amount'].cumsum() - monthly_summary['collected_amount'].cumsum()
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