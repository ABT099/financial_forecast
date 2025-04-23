from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'demand-forecasts', views.DemandForecastViewSet)
router.register(r'cashflow-forecasts', views.CashFlowForecastViewSet)
router.register(r'inventory-forecasts', views.InventoryForecastViewSet)

urlpatterns = [
    path('predict/demand/', views.DemandPrediction.as_view(), name='predict-demand'),
    path('predict/cashflow/', views.CashFlowPrediction.as_view(), name='predict-cashflow'),
    path('predict/inventory/', views.InventoryForecasting.as_view(), name='predict-inventory'),
    path('', include(router.urls)),
]