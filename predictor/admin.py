from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['top_disease', 'top_confidence', 'model_used', 'user', 'created_at']
    list_filter  = ['model_used', 'created_at']
    search_fields = ['top_disease']
