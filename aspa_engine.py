import pandas as pd
import numpy as np
try:
    from dtaidistance import dtw
except ImportError:
    print("Warning: dtaidistance benchmark not found, falling back to pure python or numpy version if available.")
    import dtaidistance.dtw as dtw

class VeridianASPAEngine:
    def __init__(self, csv_path):
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Standardize column names (User's dataset might use 'Date' or 'Datetime')
        if 'Datetime' in self.df.columns:
            self.df.rename(columns={'Datetime': 'Date'}, inplace=True)
            
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Sort by date just in case
        self.df.sort_values('Date', inplace=True)
        
        # Simple data cleaning: interpolate to remove NaNs which DTW hates
        # We do this per city to avoid bleeding data between cities
        self.df['AQI'] = self.df.groupby('City')['AQI'].transform(lambda x: x.interpolate(method='linear').bfill().ffill())
        print("Data loaded and cleaned.")

    def get_pattern_match(self, city, current_30_days):
        """
        Module A: The Pattern Searcher
        Takes the last 30 days of AQI data (query) and finds the best historical match.
        """
        # Filter for the specific city
        city_df = self.df[self.df['City'] == city]
        
        if city_df.empty:
            raise ValueError(f"No data found for city: {city}")
            
        city_data = city_df['AQI'].values
        dates = city_df['Date'].values
        
        # Normalize the query (Z-score) as per requirements
        # Note: We technically should normalize the candidate windows too for a shape-only match,
        # but for simplicity/direct-port of user code we 'll start raw or check requirement 4.
        # User REQ 4: "All data should be Z-score normalized before DTW."
        
        query = np.array(current_30_days, dtype=np.double)
        # normalize query
        query_mean = np.mean(query)
        query_std = np.std(query)
        if query_std == 0: query_std = 1 # Avoid div by zero
        query_norm = (query - query_mean) / query_std
        
        best_dist = float('inf')
        best_window_data = None
        best_start_idx = -1
        
        # History Search
        # We need at least 30 days + 14 days lookahead = 44 days margin
        search_limit = len(city_data) - 44 
        
        # Slide across history (Step 5 days to save computation)
        # Using a simple Python loop. For production, the library's built-in sequence search is faster, 
        # but this is transparent for the "learning" process.
        for i in range(0, search_limit, 5):
            candidate = np.array(city_data[i:i+30], dtype=np.double)
            
            # Normalize candidate
            cand_mean = np.mean(candidate)
            cand_std = np.std(candidate)
            if cand_std == 0: cand_std = 1
            candidate_norm = (candidate - cand_mean) / cand_std
            
            # DTW Distance Calculation
            # Try to use the standard distance function which should auto-select best method
            try:
                dist = dtw.distance(query_norm, candidate_norm)
            except Exception:
                # If that fails (likely C library missing), force pure Python
                dist = dtw.distance(query_norm, candidate_norm, use_c=False)
            
            if dist < best_dist:
                best_dist = dist
                best_start_idx = i
                
        if best_start_idx != -1:
            # The 'Look-Ahead' is the 14 days following this match
            # We return the raw (non-normalized) values for specific context
            best_match_dates = dates[best_start_idx : best_start_idx+30]
            look_ahead_data = city_data[best_start_idx+30 : best_start_idx+44]
            look_ahead_dates = dates[best_start_idx+30 : best_start_idx+44]
            
            return {
                "match_start_date": best_match_dates[0],
                "match_end_date": best_match_dates[-1],
                "distance": best_dist,
                "historical_data": city_data[best_start_idx : best_start_idx+30],
                "look_ahead_data": look_ahead_data,
                "look_ahead_dates": look_ahead_dates
            }
            
        return None
