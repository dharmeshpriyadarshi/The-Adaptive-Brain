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

    def train_trend_classifier(self, city, n_clusters=4):
        """
        Module B: The Trend Classifier (Unsupervised)
        Technique: K-Means Clustering on DTW Distances
        
        Since computing a full N x N DTW distance matrix for all 30-day windows in 10 years 
        is computationally expensive (O(N^2)), we will use a simplified approach for this demo:
        1. Extract random sample of 30-day windows from history.
        2. Cluster these windows using Euclidean distance (as a proxy for shape in vector space) 
           or fit K-Means on their statistical features (mean, std, slope).
           
        For the 'Pro' version we would use K-Means with custom DTW distance, but scikit-learn's 
        KMeans assumes Euclidean space. We will use a feature-based clustering for speed and stability.
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        city_df = self.df[self.df['City'] == city]
        if city_df.empty: return None
        
        data = city_df['AQI'].values
        
        # Create a dataset of rolling windows
        windows = []
        for i in range(0, len(data)-30, 30): # Non-overlapping for diverse training
            window = data[i:i+30]
            if len(window) == 30:
                # Feature Engineering for "Trend Shape"
                # 1. Mean (Level)
                # 2. Slope (Trend Direction)
                # 3. Std Dev (Volatility)
                mean_val = np.mean(window)
                std_val = np.std(window)
                
                # Simple linear regression for slope
                x = np.arange(30)
                slope, _ = np.polyfit(x, window, 1)
                
                windows.append([mean_val, std_val, slope])
        
        if not windows: return None
        
        X = np.array(windows)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        self.trend_model = kmeans
        self.trend_scaler = scaler
        self.trend_labels = {
            0: "Stable / Moderate",
            1: "Volatile Spike", 
            2: "Severe Accumulation",
            3: "Rapid Clearing"
        }
        # Note: In a real unsupervised system, we'd need to manually label clusters 
        # by inspecting their centroids. Here we use placeholders that we might adjust dynamically.
        
        print(f"Trend Classifier trained for {city} with {n_clusters} clusters.")
        
    def get_trend_state(self, window_data):
        """
        Returns the 'Trend State' (e.g., 'Severe Accumulation') for a given 30-day window.
        """
        if not hasattr(self, 'trend_model'):
            return "Unknown (Model Not Trained)"
            
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        x = np.arange(30)
        slope, _ = np.polyfit(x, window_data, 1)
        
        features = np.array([[mean_val, std_val, slope]])
        features_scaled = self.trend_scaler.transform(features)
        
        cluster_id = self.trend_model.predict(features_scaled)[0]
        
        # Dynamic Relabeling based on cluster centroid characteristics could go here
        # For now, return the ID or a mapped name
        return self.trend_labels.get(cluster_id, f"Trend Pattern {cluster_id}")

    def synthesize_prediction(self, historical_match_data, current_trend_slope=None):
        """
        Module C: Probabilistic Projection
        The "Look-Ahead": Takes the 14 days following the match.
        Adjustment: Applies the Velocity of the current year to that historical trend.
        """
        if historical_match_data is None:
            return None
            
        historical_projection = historical_match_data['look_ahead_data']
        historical_mean = np.mean(historical_projection)
        
        # If we have current trend info (slope), we can adjust the projection
        # For this version, we'll use a simplified weighted average as requested
        # Method 1 + Method 2 Synthesis
        
        # Calculate a "Trend Projection" based on continuing the current slope 
        # starting from the last known value
        last_known_val = historical_match_data['historical_data'][-1]
        
        # Use provided slope or 0 (flat) if unknown
        slope = current_trend_slope if current_trend_slope is not None else 0
        
        # Create a linear projection for the next 14 days based on current momentum
        trend_projection = np.array([last_known_val + (slope * i) for i in range(1, 15)])
        
        # Dynamic weighting: If trend is very strong (low distance match), trust history more?
        # User Algo: final_forecast = (alpha * trend_projection) + (beta * historical_mean vector)
        
        # Let's say we trust history shape (beta) more but adjust level via alpha?
        # Actually user formulation was:
        # final_forecast = (alpha * trend_projection) + (beta * historical_mean)
        # We will interpret 'historical_mean' as the historical vector itself to keep the shape.
        
        alpha = 0.4  # Weight for the Linear Momentum vs
        beta = 0.6   # Weight for the Historical Shape
        
        final_forecast = (alpha * trend_projection) + (beta * historical_projection)
        
        return final_forecast
