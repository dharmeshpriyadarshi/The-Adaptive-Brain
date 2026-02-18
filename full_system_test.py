from aspa_engine import VeridianASPAEngine
import numpy as np
import pandas as pd

def full_system_test():
    print("--- ASPA Full System Test (Modules A+B+C) ---")
    
    engine = VeridianASPAEngine('./data/city_day.csv')
    city = 'Delhi'
    
    print(f"\n1. Training Trend Classifier for {city}...")
    engine.train_trend_classifier(city)
    
    # Simulate a "Current 30 Days"
    # Using a known chunk from 2018 (Jan 1-30)
    print(f"\n2. Simulating 'Current' 30 Days (Jan 2018 Data)...")
    df = pd.read_csv('./data/city_day.csv')
    if 'Datetime' in df.columns: df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    mask = (df['City'] == city) & (df['Date'] >= '2018-01-01') & (df['Date'] <= '2018-01-30')
    current_30_days = df[mask]['AQI'].interpolate(method='linear').bfill().ffill().values[:30]
    
    # Calculate current slope for Module C
    x = np.arange(30)
    current_slope, _ = np.polyfit(x, current_30_days, 1)
    print(f"   Current Slope/Momentum: {current_slope:.2f}")

    # Module A
    print("\n3. Running Module A (Pattern Match)...")
    match_result = engine.get_pattern_match(city, current_30_days)
    if not match_result:
        print("No match found.")
        return

    print(f"   Best Match: {pd.to_datetime(match_result['match_start_date']).date()} (Dist: {match_result['distance']:.2f})")
    
    # Module B
    print("\n4. Running Module B (Trend Classification)...")
    trend_state = engine.get_trend_state(current_30_days)
    print(f"   Identified State: {trend_state}")
    
    # Module C
    print("\n5. Running Module C (Probabilistic Projection)...")
    forecast = engine.synthesize_prediction(match_result, current_trend_slope=current_slope)
    
    print("\n--- FINAL FORECAST (Next 14 Days) ---")
    print(np.round(forecast, 2))
    print(f"Mean Predicted AQI: {np.mean(forecast):.2f}")

if __name__ == "__main__":
    full_system_test()
