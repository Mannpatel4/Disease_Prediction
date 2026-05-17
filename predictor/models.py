from django.db import models
from django.contrib.auth.models import User
import json


class PredictionHistory(models.Model):
    """Stores each disease prediction made by a user."""
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='predictions', null=True, blank=True)
    symptoms_json   = models.TextField()          # JSON list of selected symptoms
    result_json     = models.TextField()          # JSON of top_predictions
    model_used      = models.CharField(max_length=50)
    top_disease     = models.CharField(max_length=100)
    top_confidence  = models.FloatField()
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Prediction History'
        verbose_name_plural = 'Prediction Histories'

    def get_symptoms(self):
        return json.loads(self.symptoms_json)

    def get_results(self):
        return json.loads(self.result_json)

    def __str__(self):
        return f"{self.top_disease} ({self.top_confidence}%) — {self.created_at:%d %b %Y %H:%M}"
