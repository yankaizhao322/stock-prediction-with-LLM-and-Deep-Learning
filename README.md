# Stock Prediction with LLM, LSTM-GAN, and CNN


This project implements a hybrid stock forecasting and interpretation pipeline that combines deep learning models (LSTM + GAN) with a lightweight CNN discriminator and a Large Language Model (LLM) for natural-language explanations. It leverages historical market data and social media sentiment to predict near-term stock movements and produces an analyst-style summary that explains the quantitative outputs in a traceable, evidence-based way.

   *⚠️Disclaimer: This project is for educational and research purposes only. The predictions and LLM-generated explanations should not be construed as financial investment advice.*


## Project Overview

The system follows a multi-stage workflow:

* **Quantitative Data Processing:** Loads historical OHLCV data and engineered features (e.g., returns, moving averages).
* **Sentiment Feature Engineering:** Extracts daily sentiment signals from tweets using VADER (NLTK).
* **Deep Learning Forecasting (LSTM-GAN):**
    * **Generator (LSTM):** Learns temporal patterns from time-series inputs and produces forecasts (e.g., next-day price / next-day change).
    * **Discriminator (CNN):** Distinguishes real vs. generated sequences to encourage realistic predictions and reduce mode collapse.
* **LLM Interpretation:** Summarizes the prediction, sentiment context, and evidence signals into a concise, verifiable explanation (no investment advice).

## Methodology and Technical Details

### 1. Data Preprocessing(Kaggle)

The pipeline uses pandas and sklearn.preprocessing for data handling:

* **Data Cleaning:** Handles missing values, aligns timestamps, and standardizes the trading-day index.
* **Normalization:** Uses MinMaxScaler to scale features (e.g., Open/High/Low/Close/Volume and engineered indicators) into a stable range for neural network training.
* **Windowing:** Converts the time series into supervised samples via sliding windows (lookback T → predict t+1).

### 2. Sentiment Analysis (Twitter)

Market sentiment is derived from social media signals using NLTK VADER:

* **Compound Score:** Each tweet yields a compound sentiment score in [-1, +1].
* **Daily Aggregation:** Tweets are grouped by trading day; daily mean sentiment and positive ratio are computed.
* **Feature Fusion:** Sentiment features are merged into the model input as exogenous variables to complement price/volume dynamics.

### 3. Deep Learning Models (LSTM + GAN + CNN)

This project focuses on a GAN-style forecasting framework:

* **Generator (LSTM):**
    * Learns temporal dependencies in stock features + sentiment signals.
    * Produces predicted values (e.g., next-day close or next-day return).
* **Discriminator (CNN):**
    * A CNN-based discriminator evaluates whether a sequence is real or generated.
    * Provides adversarial feedback to the LSTM generator, improving realism and stability.

**Key Training Setup (typical):**

* **Loss:** MSE (forecasting) + adversarial objective (GAN)
* **Optimizer:** Adam
* **Regularization:** Dropout and early stopping (if used)

### 4. LLM Integration (Explanation Layer)

An LLM is used as an interpretation layer, not as a trading system. It consumes:

* The model output (predicted price / predicted direction / probability if available)
* A structured evidence summary (sentiment statistics, keywords, representative tweets, prior-day price/volume signals)

It then generates a restrained, analyst-style explanation:

* Drivers supporting the predicted direction
* Uncertainty and opposing signals
* Near-term events to monitor (earnings, macro, policy, product news, etc.)
* Evidence coverage assessment (data sufficiency and bias risk)

## Installation and Requirements

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn nltk statsmodels tqdm transformers torch pytz
```

## Structure and Workflow
### Step 1: Stock Data Analysis
Load OHLCV data, compute engineered features (e.g., returns, moving averages), and visualize trends.

### Step 2: Train the LSTM-GAN Model
Train the LSTM generator with a CNN discriminator using historical windows and sentiment-enhanced inputs. Monitor training stability and validation behavior to reduce overfitting.

### Step 3: Generate Predictions
Inverse-transform model outputs back to the original scale and compare against real price movements.

### Step 4: LLM Explanation Generation
Produce a structured text explanation for each prediction date based on sentiment summary and market evidence (no investment advice).

# LLM output Results
2022-01-20

<img width="600" height="500" alt="image" src="https://github.com/user-attachments/assets/fd80737b-2109-41b0-8f2a-9145a2f35711" />


# Contact:
Author: Yankai Zhao

Email: yaz624@lghigh.edu

Institution: Lehigh University, Department of Electrical and Computer Engineering
