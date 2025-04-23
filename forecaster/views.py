from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, viewsets
import pandas as pd
import json
from prophet import Prophet
import numpy as np
from datetime import datetime
from .models import DemandForecast, CashFlowForecast, InventoryForecast
from .serializers import (DemandForecastSerializer, CashFlowForecastSerializer, 
                         InventoryForecastSerializer)
from .utils import optimize_hyperparameters, detect_and_clean_anomalies, ensemble_forecast, convert_to_serializable

class DemandPrediction(APIView):
    def post(self, request):
        try:
            serializer = DemandForecastSerializer(data=request.data)
            if not serializer.is_valid():
                return Response({
                    'errors': serializer.errors,
                    'success': False
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Extract validated data
            data = serializer.validated_data
            df = pd.DataFrame(data['historical_data'])
            df.columns = ['ds', 'y']  # Prophet requires these column names
            
            # Clean anomalies if requested
            if data.get('clean_anomalies', True):  # On by default
                df, anomaly_info = detect_and_clean_anomalies(df, sigma_threshold=data.get('anomaly_threshold', 3.0))
                # Optionally store anomaly info
                anomalies_detected = anomaly_info['anomaly_count'] > 0
            
            # Use ensemble forecast if requested
            if data.get('use_ensemble', False):
                periods = data.get('forecast_periods', 30)
                forecast_results = ensemble_forecast(
                    df, 
                    periods, 
                    methods=data.get('ensemble_methods', ['prophet', 'arima', 'ets'])
                )
                
                # Save forecast to database
                forecast_obj = serializer.save(forecast_data=forecast_results)
                
                return Response({
                    'forecast': forecast_results,
                    'forecast_id': forecast_obj.id,
                    'success': True
                })
            
            # Optimize parameters if requested
            if data.get('optimize_parameters', False):
                # This will be slower but more accurate
                optimal_params = optimize_hyperparameters(df, data.get('forecast_periods', 30))
                model = Prophet(**optimal_params)
            else:
                model = Prophet()
            
            model.fit(df)
            
            # Make future dataframe for predictions
            periods = data.get('forecast_periods', 30)
            future = model.make_future_dataframe(periods=periods)
            
            # Forecast
            forecast = model.predict(future)
            
            # Prepare response
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # Convert timestamps to strings before converting to dict
            forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
            response_data = forecast_data.tail(periods).to_dict('records')
            
            # Save forecast to database
            forecast_obj = serializer.save(forecast_data=response_data)
            
            # Perform cross-validation if requested
            if data.get('perform_validation', False):
                cv_metrics = self.perform_cross_validation(
                    model, 
                    df,
                    horizon=f"{min(30, int(len(df)/3))} days",
                    period=f"{min(90, int(len(df)/2))} days"
                )
                
                # Save accuracy metrics
                forecast_obj.accuracy = cv_metrics.get('mape')
                forecast_obj.save()
                
                # Include in response
                return Response({
                    'forecast': response_data,
                    'forecast_id': forecast_obj.id,
                    'accuracy_metrics': cv_metrics,
                    'success': True
                })
            
            return Response({
                'forecast': response_data,
                'forecast_id': forecast_obj.id,
                'success': True
            })
            
        except Exception as e:
            import traceback
            return Response({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)
    
    def perform_cross_validation(self, model, historical_df, horizon='30 days', period='90 days'):
        # Perform cross-validation
        from prophet.diagnostics import cross_validation, performance_metrics
        
        df_cv = cross_validation(model, horizon=horizon, period=period)
        df_p = performance_metrics(df_cv)
        
        # Return accuracy metrics
        return {
            'mape': float(df_p['mape'].mean()),  # Mean Absolute Percentage Error
            'rmse': float(df_p['rmse'].mean()),   # Root Mean Squared Error
            'coverage': float(df_p['coverage'].mean()),  # Coverage of prediction intervals
        }

class CashFlowPrediction(APIView):
    def post(self, request):
        try:
            serializer = CashFlowForecastSerializer(data=request.data)
            if not serializer.is_valid():
                return Response({
                    'errors': serializer.errors,
                    'success': False
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Extract validated data
            data = serializer.validated_data
            df = pd.DataFrame(data['historical_data'])
            df.columns = ['ds', 'y']  # Prophet requires these column names
            
            # Clean anomalies if requested
            if data.get('clean_anomalies', True):  # On by default
                df, anomaly_info = detect_and_clean_anomalies(df, sigma_threshold=data.get('anomaly_threshold', 3.0))
                # Optionally store anomaly info
                anomalies_detected = anomaly_info['anomaly_count'] > 0
            
            # Use ensemble forecast if requested
            if data.get('use_ensemble', False):
                periods = data.get('forecast_periods', 90)
                forecast_results = ensemble_forecast(
                    df, 
                    periods, 
                    methods=data.get('ensemble_methods', ['prophet', 'arima', 'ets'])
                )
                
                # Save forecast to database
                forecast_obj = serializer.save(forecast_data=forecast_results)
                
                return Response({
                    'forecast': forecast_results,
                    'forecast_id': forecast_obj.id,
                    'success': True
                })
            
            # Optimize parameters if requested
            if data.get('optimize_parameters', False):
                # This will be slower but more accurate
                optimal_params = optimize_hyperparameters(df, data.get('forecast_periods', 90))
                model = Prophet(**optimal_params)
            else:
                model = Prophet()
            
            # Add seasonality if provided
            if 'seasonality' in data and data['seasonality']:
                for seasonality in data['seasonality']:
                    model.add_seasonality(
                        name=seasonality['name'],
                        period=seasonality['period'],
                        fourier_order=seasonality.get('fourier_order', 5)
                    )
            
            model.fit(df)
            
            # Make future dataframe for predictions
            periods = data.get('forecast_periods', 90)
            future = model.make_future_dataframe(periods=periods)
            
            # Forecast
            forecast = model.predict(future)
            
            # Prepare response
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # Convert timestamps to strings before converting to dict
            forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
            response_data = forecast_data.tail(periods).to_dict('records')
            
            # Save forecast to database
            forecast_obj = serializer.save(forecast_data=response_data)
            
            # Perform cross-validation if requested
            if data.get('perform_validation', False):
                cv_metrics = self.perform_cross_validation(
                    model, 
                    df,
                    horizon=f"{min(30, int(len(df)/3))} days",
                    period=f"{min(90, int(len(df)/2))} days"
                )
                
                # Save accuracy metrics
                forecast_obj.accuracy = cv_metrics.get('mape')
                forecast_obj.save()
                
                # Include in response
                return Response({
                    'forecast': response_data,
                    'forecast_id': forecast_obj.id,
                    'accuracy_metrics': cv_metrics,
                    'success': True
                })
            
            return Response({
                'forecast': response_data,
                'forecast_id': forecast_obj.id,
                'success': True
            })
            
        except Exception as e:
            import traceback
            return Response({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)
    
    def perform_cross_validation(self, model, historical_df, horizon='30 days', period='90 days'):
        # Perform cross-validation
        from prophet.diagnostics import cross_validation, performance_metrics
        
        df_cv = cross_validation(model, horizon=horizon, period=period)
        df_p = performance_metrics(df_cv)
        
        # Return accuracy metrics
        return {
            'mape': float(df_p['mape'].mean()),  # Mean Absolute Percentage Error
            'rmse': float(df_p['rmse'].mean()),   # Root Mean Squared Error
            'coverage': float(df_p['coverage'].mean()),  # Coverage of prediction intervals
        }

class InventoryForecasting(APIView):
    def post(self, request):
        try:
            serializer = InventoryForecastSerializer(data=request.data)
            if not serializer.is_valid():
                return Response({
                    'errors': serializer.errors,
                    'success': False
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Extract validated data
            data = serializer.validated_data
            df = pd.DataFrame(data['historical_data'])
            df.columns = ['ds', 'y']  # Prophet requires these column names
            
            # Clean anomalies if requested
            if data.get('clean_anomalies', True):  # On by default
                df, anomaly_info = detect_and_clean_anomalies(df, sigma_threshold=data.get('anomaly_threshold', 3.0))
                # Optionally store anomaly info
                anomalies_detected = anomaly_info['anomaly_count'] > 0
            
            # Use ensemble forecast if requested
            if data.get('use_ensemble', False):
                periods = data.get('forecast_periods', 60)
                forecast_results = ensemble_forecast(
                    df, 
                    periods, 
                    methods=data.get('ensemble_methods', ['prophet', 'arima', 'ets'])
                )
                
                # Save forecast to database
                forecast_obj = serializer.save(forecast_data=forecast_results)
                
                return Response({
                    'forecast': forecast_results,
                    'forecast_id': forecast_obj.id,
                    'success': True
                })
            
            # Optimize parameters if requested
            if data.get('optimize_parameters', False):
                # This will be slower but more accurate
                optimal_params = optimize_hyperparameters(df, data.get('forecast_periods', 60))
                model = Prophet(**optimal_params)
            else:
                model = Prophet()
            
            # Add holidays if provided
            if 'holidays' in data and data['holidays']:
                holidays_df = pd.DataFrame(data['holidays'])
                model = Prophet(holidays=holidays_df)
            
            # Add regressors if provided
            if 'regressors' in data and data['regressors']:
                for regressor_name, regressor_values in data['regressors'].items():
                    # Add regressor to the model
                    model.add_regressor(regressor_name)
                    # Add regressor values to the dataframe
                    df[regressor_name] = regressor_values
            
            model.fit(df)
            
            # Make future dataframe for predictions
            periods = data.get('forecast_periods', 60)
            future = model.make_future_dataframe(periods=periods)
            
            # Add regressors to future dataframe if provided
            if 'future_regressors' in data and data['future_regressors']:
                # Calculate the length of the historical data
                hist_length = len(df)
                
                for regressor_name, regressor_values in data['future_regressors'].items():
                    # Check if the regressor values match the forecast periods
                    if len(regressor_values) != periods:
                        return Response({
                            'error': f"Length of future regressor '{regressor_name}' ({len(regressor_values)}) must match forecast_periods ({periods})",
                            'success': False
                        }, status=status.HTTP_400_BAD_REQUEST)
                    
                    # Check if we have historical values for this regressor
                    if regressor_name in data['regressors']:
                        # Use the historical values for historical dates and future values for future dates
                        future[regressor_name] = np.concatenate([
                            np.array(data['regressors'][regressor_name]),  # Historical values
                            np.array(regressor_values)                      # Future values
                        ])
                    else:
                        # Regressor wasn't used in training - can't use it for prediction
                        return Response({
                            'error': f"Future regressor '{regressor_name}' not found in historical regressors",
                            'success': False
                        }, status=status.HTTP_400_BAD_REQUEST)
            
            # Forecast
            forecast = model.predict(future)
            
            # Prepare response
            forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            # Convert timestamps to strings before converting to dict
            forecast_data['ds'] = forecast_data['ds'].dt.strftime('%Y-%m-%d')
            response_data = forecast_data.tail(periods).to_dict('records')
            
            # Save forecast to database
            forecast_obj = serializer.save(forecast_data=convert_to_serializable(response_data))
            
            # Perform cross-validation if requested
            if data.get('perform_validation', False):
                cv_metrics = self.perform_cross_validation(
                    model, 
                    df,
                    horizon=f"{min(30, int(len(df)/3))} days",
                    period=f"{min(90, int(len(df)/2))} days"
                )
                
                # Save accuracy metrics
                forecast_obj.accuracy = cv_metrics.get('mape')
                forecast_obj.save()
                
                # Include in response
                return Response({
                    'forecast': response_data,
                    'forecast_id': forecast_obj.id,
                    'accuracy_metrics': cv_metrics,
                    'success': True
                })
            
            return Response({
                'forecast': response_data,
                'forecast_id': forecast_obj.id,
                'success': True
            })
            
        except Exception as e:
            import traceback
            return Response({
                'error': str(e),
                'traceback': traceback.format_exc(),
                'success': False
            }, status=status.HTTP_400_BAD_REQUEST)
    
    def perform_cross_validation(self, model, historical_df, horizon='30 days', period='90 days'):
        # Perform cross-validation
        from prophet.diagnostics import cross_validation, performance_metrics
        
        df_cv = cross_validation(model, horizon=horizon, period=period)
        df_p = performance_metrics(df_cv)
        
        # Return accuracy metrics
        return {
            'mape': float(df_p['mape'].mean()),  # Mean Absolute Percentage Error
            'rmse': float(df_p['rmse'].mean()),   # Root Mean Squared Error
            'coverage': float(df_p['coverage'].mean()),  # Coverage of prediction intervals
        }

# ViewSets for retrieving saved forecasts
class DemandForecastViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = DemandForecast.objects.all().order_by('-created_at')
    serializer_class = DemandForecastSerializer

class CashFlowForecastViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = CashFlowForecast.objects.all().order_by('-created_at')
    serializer_class = CashFlowForecastSerializer

class InventoryForecastViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = InventoryForecast.objects.all().order_by('-created_at')
    serializer_class = InventoryForecastSerializer
