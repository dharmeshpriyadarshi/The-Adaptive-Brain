import pandas as pd
import numpy as np
try:
    from dtaidistance import dtw
except ImportError:
    print("Warning: dtaidistance benchmark not found, falling back to pure python or numpy version if available.")
    import dtaidistance.dtw as dtw

class VeridianASPAEngine:
    def __init__(self, csv_path, train_end_date=None):
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
        
        # Apply training cutoff for rigorous backtesting evaluation
        if train_end_date:
            cutoff = pd.to_datetime(train_end_date)
            self.df = self.df[self.df['Date'] < cutoff]
            print(f"Rigorous ML Split: Historical training data restricted to before {cutoff.date()}.")
            
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
        """
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        city_df = self.df[self.df['City'] == city]
        if city_df.empty: return None
        
        data = city_df['AQI'].values
        
        windows = []
        for i in range(0, len(data)-30, 30):
            window = data[i:i+30]
            if len(window) == 30:
                mean_val = np.mean(window)
                std_val = np.std(window)
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
        
        # Dynamically map labels meaning based on actual centroid properties (mean, std, slope)
        # to prevent "Severe Accumulation" being assigned to arbitrary ID=2 which could be low AQI.
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Simple heuristic labeling based on centroids
        # centroids[:, 0] is mean, centroids[:, 1] is std, centroids[:, 2] is slope
        self.trend_labels = {}
        for i, c in enumerate(centroids):
            c_mean, c_std, c_slope = c[0], c[1], c[2]
            
            if c_mean > 200:
                if c_slope > 1: self.trend_labels[i] = "Severe Accumulation"
                elif c_slope < -1: self.trend_labels[i] = "Clearing from Severe Crisis"
                else: self.trend_labels[i] = "Sustained Severe Pollution"
            elif c_mean < 100:
                if c_slope > 1: self.trend_labels[i] = "Rising Moderate"
                else: self.trend_labels[i] = "Stable / Satisfactory"
            else:
                if c_std > 50: self.trend_labels[i] = "Volatile Spike"
                elif c_slope > 0: self.trend_labels[i] = "Gradual Accumulation"
                else: self.trend_labels[i] = "Gradual Clearing"
                
            # Fallback if unassigned
            if i not in self.trend_labels:
                self.trend_labels[i] = "Moderate Fluctuation"
        
        print(f"Trend Classifier trained for {city} with {n_clusters} clusters.")
        
    def get_trend_state(self, window_data):
        """
        Returns the 'Trend State' and structural ML metrics for transparency.
        """
        if not hasattr(self, 'trend_model'):
            return {"label": "Unknown", "cluster_id": -1, "mean": 0, "std": 0, "slope": 0}
            
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        x = np.arange(30)
        slope, _ = np.polyfit(x, window_data, 1)
        
        features = np.array([[mean_val, std_val, slope]])
        features_scaled = self.trend_scaler.transform(features)
        
        cluster_id = int(self.trend_model.predict(features_scaled)[0])
        label = self.trend_labels.get(cluster_id, f"Trend Pattern {cluster_id}")
        
        return {
            "label": label,
            "cluster_id": cluster_id,
            "mean": round(float(mean_val), 2),
            "std": round(float(std_val), 2),
            "slope": round(float(slope), 2)
        }

    def synthesize_prediction(self, historical_match_data, current_trend_slope=0):
        """
        Module C: Probabilistic Projection
        Returns the final forecast array + transparency dict
        """
        if historical_match_data is None:
            return None, {}
            
        historical_projection = historical_match_data['look_ahead_data']
        last_known_val = historical_match_data['historical_data'][-1]
        
        slope = current_trend_slope if current_trend_slope is not None else 0
        trend_projection = np.array([last_known_val + (slope * i) for i in range(1, 15)])
        
        alpha = 0.4  # Weight for Linear Momentum
        beta = 0.6   # Weight for Historical Shape
        
        final_forecast = (alpha * trend_projection) + (beta * historical_projection)
        
        details = {
            "alpha": alpha,
            "beta": beta,
            "momentum_v": round(float(slope), 2),
            "base_hist_pred": round(float(historical_projection[0]), 2),
            "base_trend_pred": round(float(trend_projection[0]), 2)
        }
        
        return final_forecast, details
