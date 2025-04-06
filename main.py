from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import kagglehub
from google import genai
from google.genai import types
from flask_cors import CORS
import joblib

# Define paths for saving model and preprocessing objects
MODEL_PATH = "lstm_stock_model.keras"  # Native Keras format
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

app = Flask(_name_)
CORS(app)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values(by=["Date"])
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[df.index >= "2021-01-01"]
    
    encoder = LabelEncoder()
    df["Stock"] = encoder.fit_transform(df["Stock"])
    
    # Since "Stock" is the first column, scale the remaining columns
    scaler = MinMaxScaler()
    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
    
    # Save encoder and scaler for later reuse
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return df, encoder, scaler

def prepare_dataset(df, time_steps=20):
    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i : i + time_steps])
            y.append(data[i + time_steps, 3])  # Predict "Close" price
        return np.array(X), np.array(y)
    
    X_list, y_list = [], []
    for stock_id in df["Stock"].unique():
        stock_data = df[df["Stock"] == stock_id].drop(["Stock"], axis=1).values
        if len(stock_data) > time_steps:
            X_stock, y_stock = create_sequences(stock_data, time_steps)
            X_list.append(X_stock)
            y_list.append(y_stock)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def build_and_train_model(X, y, time_steps=20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True)
    
    model = Sequential([
        Input(shape=(time_steps, X.shape[2])),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X_train[-100000:], y_train[-100000:], epochs=10, batch_size=64,
              validation_data=(X_test, y_test))
    
    # Save the trained model using the native Keras format
    model.save(MODEL_PATH, save_format='keras')
    return model

def generate_stock_insights(stock_name, df, encoder, scaler, model, time_steps=20):
    try:
        stock_id = encoder.transform([stock_name])[0]
    except Exception:
        return {"error": "Stock name not found in dataset."}
    
    stock_data = df[df["Stock"] == stock_id].drop(["Stock"], axis=1).values
    if len(stock_data) <= time_steps:
        return {"error": "Insufficient data for this stock."}
    
    # Prepare sequences for the specific stock
    X_stock, _ = prepare_dataset(df[df["Stock"] == stock_id], time_steps)
    y_pred = model.predict(X_stock[-10:])
    # There are 6 features after dropping "Stock", so repeat y_pred 6 times for inverse transform
    y_pred_rescaled = scaler.inverse_transform(np.repeat(y_pred.reshape(-1, 1), 6, axis=1))[:, 3]
    
    insights = generate_gemini_insights(y_pred_rescaled, stock_name)
    return {"predicted_prices": y_pred_rescaled.tolist(), "insights": insights}

def generate_gemini_insights(y_pred_sample, stock_name):
    # Configure Gemini API using environment variable for security
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_id = "gemini-2.0-flash"
    prompt = f"""
    Analyzing future trends for stock: {stock_name}
    Predicted Prices: {y_pred_sample}
    Provide investment insights and actionable recommendations for buying or selling {stock_name}.
    """
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")
    response_text = ""
    for chunk in genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content_stream(
            model=model_id, contents=contents, config=generate_content_config):
        response_text += chunk.text
    return response_text

def generate_market_insights(df, encoder, scaler, model, time_steps=20):
    # Filter for recent data (last 6 months)
    recent_df = df[df.index >= (df.index.max() - pd.DateOffset(months=6))]
    top_stocks_ids = recent_df['Stock'].value_counts().nlargest(20).index.tolist()
    
    stock_predictions = {}
    for stock_id in top_stocks_ids:
        stock_data = df[df["Stock"] == stock_id].drop(["Stock"], axis=1).values
        if len(stock_data) > time_steps:
            X_stock, _ = prepare_dataset(df[df["Stock"] == stock_id], time_steps)
            y_pred = model.predict(X_stock[-1:])
            # Inverse transform using 6 features (since "Stock" is dropped)
            price_pred = scaler.inverse_transform(np.repeat(y_pred.reshape(-1, 1), 6, axis=1))[:, 3][0]
            stock_name_str = encoder.inverse_transform([stock_id])[0]
            stock_predictions[stock_name_str] = price_pred
    insights = generate_gemini_market_analysis(stock_predictions)
    return {"market_predictions": stock_predictions, "insights": insights}

def generate_gemini_market_analysis(stock_predictions):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_id = "gemini-2.0-flash"
    prompt = f"""
    Future stock market analysis based on predicted 'Close' prices:
    {stock_predictions}
    Identify the most profitable and risky stocks for investment with reasons.
    Provide actionable investment recommendations.
    """
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")
    response_text = ""
    for chunk in genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models.generate_content_stream(
            model=model_id, contents=contents, config=generate_content_config):
        response_text += chunk.text
    return response_text

# Flask API Routes
@app.route('/predict_stock', methods=['GET'])
def predict_stock():
    stock_name = request.args.get('stock_name')
    if not stock_name:
        return jsonify({"error": "Stock name is required"}), 400
    result = generate_stock_insights(stock_name, df, encoder, scaler, model)
    return jsonify({"data": result}), 200

@app.route('/market_insights', methods=['GET'])
def market_insights():
    result = generate_market_insights(df, encoder, scaler, model)
    return jsonify({"data": result}), 200

# Startup: Download dataset, load data, and train or load model
if _name_ == '_main_':
    print("Downloading dataset...")
    path = kagglehub.dataset_download("andrewmvd/india-stock-market")
    filepath = None
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename == "stocks_df.csv":
                filepath = os.path.join(dirname, filename)
                break
        if filepath:
            break
    if not filepath:
        raise FileNotFoundError("stocks_df.csv not found in downloaded dataset.")
    print(f"Using dataset file: {filepath}")
    
    df, encoder, scaler = load_and_preprocess_data(filepath)
    X, y = prepare_dataset(df)
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print("Training new model...")
        model = build_and_train_model(X, y)
    
    app.run(host='0.0.0.0', port=8080, debug=True)