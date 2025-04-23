from rest_framework import serializers
from .models import DemandForecast, CashFlowForecast, InventoryForecast

class DemandForecastSerializer(serializers.ModelSerializer):
    perform_validation = serializers.BooleanField(required=False, default=False)
    optimize_parameters = serializers.BooleanField(required=False, default=False)
    clean_anomalies = serializers.BooleanField(required=False, default=True)
    anomaly_threshold = serializers.FloatField(required=False, default=3.0)
    use_ensemble = serializers.BooleanField(required=False, default=False)
    ensemble_methods = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=['prophet']
    )
    
    class Meta:
        model = DemandForecast
        fields = ['id', 'name', 'historical_data', 'forecast_periods', 'forecast_data', 
                  'created_at', 'accuracy', 'perform_validation', 'optimize_parameters',
                  'clean_anomalies', 'anomaly_threshold', 'use_ensemble', 'ensemble_methods']
        read_only_fields = ['id', 'forecast_data', 'created_at', 'accuracy']

class SeasonalitySerializer(serializers.Serializer):
    name = serializers.CharField(required=True)
    period = serializers.FloatField(required=True)
    fourier_order = serializers.IntegerField(required=False, default=5)

class CashFlowForecastSerializer(serializers.ModelSerializer):
    perform_validation = serializers.BooleanField(required=False, default=False)
    optimize_parameters = serializers.BooleanField(required=False, default=False)
    clean_anomalies = serializers.BooleanField(required=False, default=True)
    anomaly_threshold = serializers.FloatField(required=False, default=3.0)
    use_ensemble = serializers.BooleanField(required=False, default=False)
    ensemble_methods = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=['prophet']
    )
    seasonality = serializers.ListField(
        child=SeasonalitySerializer(),
        required=False
    )
    
    class Meta:
        model = CashFlowForecast
        fields = ['id', 'name', 'historical_data', 'forecast_periods', 'forecast_data', 
                  'created_at', 'accuracy', 'perform_validation', 'optimize_parameters',
                  'clean_anomalies', 'anomaly_threshold', 'use_ensemble', 'ensemble_methods',
                  'seasonality']
        read_only_fields = ['id', 'forecast_data', 'created_at', 'accuracy']

class HolidaySerializer(serializers.Serializer):
    holiday = serializers.CharField(required=True)
    ds = serializers.DateField(required=True)
    lower_window = serializers.IntegerField(required=False, default=0)
    upper_window = serializers.IntegerField(required=False, default=0)

class InventoryForecastSerializer(serializers.ModelSerializer):
    perform_validation = serializers.BooleanField(required=False, default=False)
    optimize_parameters = serializers.BooleanField(required=False, default=False)
    clean_anomalies = serializers.BooleanField(required=False, default=True)
    anomaly_threshold = serializers.FloatField(required=False, default=3.0)
    use_ensemble = serializers.BooleanField(required=False, default=False)
    ensemble_methods = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=['prophet']
    )
    holidays = serializers.ListField(
        child=HolidaySerializer(),
        required=False
    )
    regressors = serializers.JSONField(required=False)
    future_regressors = serializers.JSONField(required=False)
    
    class Meta:
        model = InventoryForecast
        fields = ['id', 'name', 'historical_data', 'forecast_periods', 'forecast_data', 
                  'created_at', 'accuracy', 'perform_validation', 'optimize_parameters',
                  'clean_anomalies', 'anomaly_threshold', 'use_ensemble', 'ensemble_methods',
                  'holidays', 'regressors', 'future_regressors']
        read_only_fields = ['id', 'forecast_data', 'created_at', 'accuracy']