# 📊 Finance Collections Dashboard (Streamlit)

This interactive Streamlit dashboard provides visual analytics and forecasting for order collections and invoices. Built using Python and Prophet, it allows finance teams to monitor cashflow, identify collection trends, and forecast DSO and payment behavior.

---

## 🔧 Features

- Upload and analyze your Excel data (`Workbook.xlsx`)
- Monthly breakdown of invoiced vs. collected amounts
- Net cashflow visualizations
- Prophet-based forecasts for:
  - Collected amounts
  - Days Sales Outstanding (DSO)
- Filters by collection status and invoice date range

---

## 📂 File Structure

```
finance_dashboard/
├── app.py               # Streamlit app
├── Workbook.xlsx        # Your local data file
├── requirements.txt     # Package dependencies
├── README.md            # Project overview
```

---

## ▶️ How to Run

### 1. Clone this repo

```bash
git clone https://github.com/your-username/finance-dashboard.git
cd finance-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit

```bash
streamlit run app.py
```

Then open the local URL shown (usually `http://localhost:8501`) in your browser.

---

## 📈 Example Use Cases

- Monitor and compare monthly invoice and payment performance
- Forecast future collections and DSO
- Spot seasonal collection dips or invoice surges
- Export results and share insights with finance teams

---

## 📦 Built With

- [Streamlit](https://streamlit.io)
- [Prophet](https://facebook.github.io/prophet/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)

---

## 📬 Feedback & Ideas?

Feel free to open an issue or connect with me on [LinkedIn](https://www.linkedin.com/). I'd love to hear how you're using this or what you'd like to see next.
