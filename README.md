# The Adaptive Brain: Pollution Trend Prediction

**Veridian Project - Advanced Agentic Coding**

This project explores Machine Learning approaches to predict environmental pollution trends. Specifically, it implements an **Adaptive Subsequence Pattern Alignment (ASPA)** model to predict Air Quality Index (AQI) dynamics for 2026 based on 10 years of historical data.

## 🚀 The Core Idea
Standard correlation fails in environmental science because nature is "warped" (e.g., winter peaks shift by weeks each year). 
**ASPA** uses **Dynamic Time Warping (DTW)** to "stretch" and "compress" time, identifying that two pollution events are *conceptually* the same even if they occur on different dates.

## 📂 Project Structure

### 1. The Frontend (Visualization)
- `index.html`: The main user interface.
- `style.css`: Industrial/Earthy styling with glassmorphism effects.
- `script.js`: Handles logic and rendering charts (Chart.js).
- `predictions.json`: Stores the pre-computed 2026 predictions for fast loading.

### 2. The Backend (ASPA Engine)
- `aspa_engine.py`: The core Python logic using `dtaidistance`.
    - **Module A**: Patterns Searcher (DTW) - Finds best historical matches.
    - **Module B**: Trend Classifier (K-Means) - *[In Progress]*
    - **Module C**: Probabilistic Projection - *[In Progress]*
- `demo_module_a.py`: A verification script to test the DTW engine.

## 🛠️ How to Run
1.  **Frontend**: Simply open `index.html` in any web browser. No server required for the static demo.
2.  **Backend`:
    ```bash
    pip install dtaidistance pandas scipy scikit-learn
    python demo_module_a.py
    ```

## 📊 Dataset
Uses `city_day.csv` (Kaggle Indian Cities Pollution Database) covering 2015-2020+.

---
*Created by Antigravity & User*
