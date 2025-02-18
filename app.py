from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Define Uploads Directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return process_excel(filepath)

def process_excel(filepath):
    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower()
    except Exception as e:
        return jsonify({"error": f"Error reading file: {str(e)}"}), 400

    required_columns = {'product', 'date', 'sales', 'cogs'}
    if not required_columns.issubset(df.columns):
        return jsonify({"error": f"Missing required columns: {list(required_columns - set(df.columns))}"}), 400

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    df['cogs'] = pd.to_numeric(df['cogs'], errors='coerce').fillna(0)
    
    if df.empty or df['date'].isnull().all():
        return jsonify({"error": "No valid data found."}), 400

    df['profit'] = df['sales'] - df['cogs']
    df['day'] = df['date'].dt.strftime('%Y-%m-%d')
    df['week'] = df['date'].dt.strftime('%Y-%U')
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['year'] = df['date'].dt.year.astype(str)

    total_sales = df['sales'].sum()
    total_cogs = df['cogs'].sum()
    total_profit_or_loss = total_sales - total_cogs

    product_sales = df.groupby('product')['sales'].sum().reset_index().values.tolist()
    low_sales_products = df[df['sales'] < df['sales'].mean()].groupby('product')['sales'].sum().reset_index().values.tolist()

    daily_sales = df.groupby('day')['sales'].sum().reset_index()
    monthly_sales = df.groupby('month')['sales'].sum().reset_index()
    yearly_sales = df.groupby('year')['sales'].sum().reset_index()

    def percent_change(series):
        return series.pct_change().fillna(0).replace([np.inf, -np.inf], 0).astype(float).tolist()

    growth_data = {
        "daily_growth": percent_change(daily_sales['sales']),
        "monthly_growth": percent_change(monthly_sales['sales']),
        "yearly_growth": percent_change(yearly_sales['sales'])
    }

    def predict_sales(data):
        if len(data) < 2:
            return 0
        data = data.reset_index(drop=True)
        data['index'] = np.arange(len(data))
        X, y = data[['index']], data['sales']
        model = LinearRegression().fit(X, y)
        return float(model.predict([[len(data)]])[0])  # Ensure it's a native float

    predictions = {
        "daily": predict_sales(daily_sales),
        "monthly": predict_sales(monthly_sales),
        "yearly": predict_sales(yearly_sales)
    }

    def convert_numpy(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    response_data = {
        "totalSales": total_sales,
        "totalCOGS": total_cogs,
        "profitOrLoss": total_profit_or_loss,
        "productSales": {"headers": ["Product", "Total Sales"], "rows": product_sales},
        "lowSalesProducts": {"headers": ["Product", "Total Sales"], "rows": low_sales_products},
        "growthAnalysis": growth_data,
        "predictedSales": predictions,
        "summary": "Financial analysis completed."
    }

    return jsonify(convert_numpy(response_data))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
