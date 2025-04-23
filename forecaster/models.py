from django.db import models
from django.utils import timezone
import json

class ForecastBase(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    forecast_periods = models.IntegerField(default=30)
    historical_data = models.JSONField()
    forecast_data = models.JSONField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    
    class Meta:
        abstract = True
    
    def save_forecast_results(self, results):
        self.forecast_data = results
        self.save()

class DemandForecast(ForecastBase):
    name = models.CharField(max_length=100, default="Demand Forecast")
    
    def __str__(self):
        return f"Demand Forecast {self.id} - {self.created_at.strftime('%Y-%m-%d')}"

class CashFlowForecast(ForecastBase):
    name = models.CharField(max_length=100, default="Cash Flow Forecast")
    seasonality = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"Cash Flow Forecast {self.id} - {self.created_at.strftime('%Y-%m-%d')}"

class InventoryForecast(ForecastBase):
    name = models.CharField(max_length=100, default="Inventory Forecast")
    holidays = models.JSONField(null=True, blank=True)
    regressors = models.JSONField(null=True, blank=True)
    
    def __str__(self):
        return f"Inventory Forecast {self.id} - {self.created_at.strftime('%Y-%m-%d')}"
