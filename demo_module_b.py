from aspa_engine import VeridianASPAEngine
import numpy as np
import pandas as pd

def demo_module_b():
    print("--- ASPA Module B Demo (Trend Classifier) ---")
    
    engine = VeridianASPAEngine('./data/city_day.csv')
    city = 'Delhi'
    
    # 1. Train the Classifier
    print(f"\nTraining Trend Classifier for {city}...")
    engine.train_trend_classifier(city)
    
    # 2. Test Classification on a "Severe" looking window (High Mean, Rising Slope)
    # Creating a synthetic fake window 
    fake_severe_window = np.linspace(300, 500, 30) # rising from 300 to 500
    
    # 3. Test on a "Clean" looking window
    fake_clean_window = np.linspace(50, 60, 30) # steady low
    
    print("\nClassifying Synthetic 'Severe Rising' Data...")
    state_severe = engine.get_trend_state(fake_severe_window)
    print(f"Identified State: {state_severe}")
    
    print("\nClassifying Synthetic 'Clean Stable' Data...")
    state_clean = engine.get_trend_state(fake_clean_window)
    print(f"Identified State: {state_clean}")

if __name__ == "__main__":
    demo_module_b()
