from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import numpy as np
import itertools
import json
from datetime import datetime, date

class ForecastJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d')
        return super().default(obj)

def convert_to_serializable(data):
    """Convert a pandas DataFrame to JSON serializable format"""
    return json.loads(json.dumps(data, cls=ForecastJSONEncoder))

def detect_and_clean_anomalies(df, sigma_threshold=3.0):
    """Detect and handle anomalies in time series data"""
    from scipy import stats
    import numpy as np
    
    # Copy the dataframe
    df_clean = df.copy()
    
    # Calculate z-scores
    z_scores = stats.zscore(df_clean['y'])
    abs_z_scores = np.abs(z_scores)
    
    # Find anomalies
    anomaly_indices = np.where(abs_z_scores > sigma_threshold)[0]
    
    if len(anomaly_indices) > 0:
        # Replace anomalies with median or moving average
        median_value = df_clean['y'].median()
        
        for idx in anomaly_indices:
            # Use local median if possible
            window = 5
            start = max(0, idx - window)
            end = min(len(df_clean), idx + window)
            
            local_values = df_clean['y'].iloc[start:end].values
            # Remove the anomaly itself from local values
            local_values = local_values[local_values != df_clean['y'].iloc[idx]]
            
            if len(local_values) > 0:
                # Use local median
                df_clean.loc[df_clean.index[idx], 'y'] = np.median(local_values)
            else:
                # Fallback to global median
                df_clean.loc[df_clean.index[idx], 'y'] = median_value
    
    # Return cleaned dataframe and anomaly info
    return df_clean, {
        'anomaly_count': len(anomaly_indices),
        'anomaly_indices': anomaly_indices.tolist() if len(anomaly_indices) > 0 else []
    }

def optimize_hyperparameters(historical_df, forecast_periods, seasonality_mode='additive'):
    """Find optimal hyperparameters for Prophet model"""
    
    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }
    
    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Storage for results
    results = []
    
    # Use cross validation to evaluate all parameters
    for params in all_params:
        model = Prophet(
            seasonality_mode=seasonality_mode,
            **params
        )
        model.fit(historical_df)
        
        # Cross-validate
        horizon = f"{min(forecast_periods, 30)} days"  # Use smaller of forecast_periods or 30 days
        period = f"{min(int(len(historical_df)/3), 90)} days"
        
        df_cv = cross_validation(model, horizon=horizon, period=period, disable_tqdm=True)
        df_p = performance_metrics(df_cv)
        
        # Collect results
        results.append({
            'params': params,
            'rmse': df_p['rmse'].mean(),
            'mape': df_p['mape'].mean()
        })
    
    # Find the best parameters
    best_params = min(results, key=lambda x: x['mape'])
    return best_params['params']

def ensemble_forecast(df, forecast_periods, methods=['prophet', 'arima', 'ets']):
    """Generate forecasts using multiple methods and combine them"""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    # Convert to pandas datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    forecasts = {}
    
    # Generate Prophet forecast
    if 'prophet' in methods:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        forecasts['prophet'] = forecast[['ds', 'yhat']].tail(forecast_periods).set_index('ds')
    
    # Generate ARIMA forecast
    if 'arima' in methods:
        # Prepare data for ARIMA
        ts = df.set_index('ds')['y']
        
        # Fit ARIMA model
        try:
            model = ARIMA(ts, order=(5,1,0))
            model_fit = model.fit()
            
            # Make forecast
            arima_forecast = model_fit.forecast(steps=forecast_periods)
            arima_dates = pd.date_range(
                start=df['ds'].iloc[-1] + pd.Timedelta(days=1),
                periods=forecast_periods
            )
            forecasts['arima'] = pd.DataFrame({
                'yhat': arima_forecast
            }, index=arima_dates)
        except:
            # ARIMA may fail, just ignore if it does
            pass
    
    # Generate ETS forecast
    if 'ets' in methods:
        # Prepare data for ETS
        ts = df.set_index('ds')['y']
        
        try:
            # Fit ETS model
            model = ExponentialSmoothing(
                ts,
                trend='add',
                seasonal='add',
                seasonal_periods=7  # Assuming daily data with weekly seasonality
            )
            model_fit = model.fit()
            
            # Make forecast
            ets_forecast = model_fit.forecast(forecast_periods)
            ets_dates = pd.date_range(
                start=df['ds'].iloc[-1] + pd.Timedelta(days=1),
                periods=forecast_periods
            )
            forecasts['ets'] = pd.DataFrame({
                'yhat': ets_forecast
            }, index=ets_dates)
        except:
            # ETS may fail, just ignore if it does
            pass
    
    # Combine forecasts (simple average)
    result = pd.DataFrame()
    for method, forecast_df in forecasts.items():
        if result.empty:
            result = forecast_df.copy()
            result.columns = [method]
        else:
            result = result.join(forecast_df, rsuffix=f'_{method}')
            result.rename(columns={'yhat': method}, inplace=True)
    
    # Calculate ensemble average
    result['ensemble'] = result.mean(axis=1)
    
    # Create final output format
    final_output = []
    for date, row in result.iterrows():
        final_output.append({
            'ds': date.strftime('%Y-%m-%d'),
            'yhat': row['ensemble'],
            'yhat_lower': row['ensemble'] * 0.9,  # Simple approximation for confidence intervals
            'yhat_upper': row['ensemble'] * 1.1,
            'methods': methods,
            'individual_forecasts': {method: float(row[method]) for method in methods if method in row}
        })
    
    return final_output

