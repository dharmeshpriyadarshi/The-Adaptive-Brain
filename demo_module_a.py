from aspa_engine import VeridianASPAEngine
import pandas as pd
import numpy as np

def demo_module_a():
    print("--- ASPA Module A Demo ---")
    
    # Initialize Engine
    # Assuming data is in 'data/city_day.csv'
    engine = VeridianASPAEngine('./data/city_day.csv')
    
    # Pick a dummy 'Current 30 Days' to test
    # Let's pretend we are in Delhi and "current" days are some random chunk from the dataset 
    # (e.g., first 30 days of 2020) just to see if it finds itself or a similar pattern.
    # In a real scenario, this would be live API data.
    
    city = 'Delhi'
    print(f"\nSimulating 'Current 30 Days' for {city}...")
    
    # For demo, grab strict values from file to serve as "current"
    # Let's say we grab Jan 1 2018 to Jan 30 2018
    # We expect the engine to find this exact window (dist=0) or a very close one.
    df = pd.read_csv('./data/city_day.csv')
    if 'Datetime' in df.columns: df.rename(columns={'Datetime': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    mask = (df['City'] == city) & (df['Date'] >= '2018-01-01') & (df['Date'] <= '2018-01-30')
    current_30_days_data = df[mask]['AQI'].interpolate(method='linear').bfill().ffill().values[:30]
    
    if len(current_30_days_data) < 30:
        print("Not enough data for demo. Exiting.")
        return

    print(f"Query Pattern (First 5 values): {current_30_days_data[:5]}...")
    
    # Run Module A
    print("\nRunning Pattern Searcher (DTW)... please wait...")
    result = engine.get_pattern_match(city, current_30_days_data)
    
    if result:
        print("\n--- MATCH FOUND! ---")
        print(f"Best Historical Match Date: {pd.to_datetime(result['match_start_date']).strftime('%Y-%m-%d')}")
        print(f"DTW Distance Score: {result['distance']:.4f}")
        print(f"Look-Ahead (Next 14 Days Mean AQI): {np.mean(result['look_ahead_data']):.2f}")
        print("--------------------")
        
        # Verification: If distance is close to 0, it found itself (which is correct behavior for this test)
        if result['distance'] < 0.1:
            print("(Success: The model correctly identified the exact historical period used as query)")
    else:
        print("No match found.")

if __name__ == "__main__":
    demo_module_a()
