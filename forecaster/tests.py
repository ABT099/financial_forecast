from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ForecastApiTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        
        # Generate sample historical data
        dates = pd.date_range(start='2023-01-01', periods=100).strftime('%Y-%m-%d').tolist()
        values = (np.sin(np.arange(100)/10) * 50 + 100 + np.random.normal(0, 5, 100)).tolist()
        self.historical_data = [{"ds": date, "y": val} for date, val in zip(dates, values)]
        
    def test_demand_prediction(self):
        """Test demand prediction endpoint"""
        url = reverse('predict-demand')
        data = {
            'name': 'Test Demand Forecast',
            'historical_data': self.historical_data,
            'forecast_periods': 30
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertIn('forecast', response.data)
        self.assertEqual(len(response.data['forecast']), 30)
        
    def test_cashflow_prediction(self):
        """Test cash flow prediction endpoint"""
        url = reverse('predict-cashflow')
        data = {
            'name': 'Test Cash Flow Forecast',
            'historical_data': self.historical_data,
            'forecast_periods': 30,
            'seasonality': [
                {'name': 'quarterly', 'period': 91.25, 'fourier_order': 5}
            ]
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertIn('forecast', response.data)
        self.assertEqual(len(response.data['forecast']), 30)
        
    def test_inventory_prediction(self):
        """Test inventory prediction endpoint"""
        url = reverse('predict-inventory')
        
        # Generate regressor data
        regressor_values = np.random.normal(10, 2, 100).tolist()
        future_regressor_values = np.random.normal(10, 2, 60).tolist()
        
        data = {
            'name': 'Test Inventory Forecast',
            'historical_data': self.historical_data,
            'forecast_periods': 60,
            'regressors': {
                'price': regressor_values
            },
            'future_regressors': {
                'price': future_regressor_values
            }
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['success'])
        self.assertIn('forecast', response.data)
        self.assertEqual(len(response.data['forecast']), 60)
        
    def test_invalid_input(self):
        """Test API with invalid input"""
        url = reverse('predict-demand')
        # Missing required historical_data
        data = {
            'forecast_periods': 30
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(response.data['success'])
