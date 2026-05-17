"""
views.py — Healthcare Disease Prediction Django App
"""

import json
import logging
import os

from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_POST

from .models import PredictionHistory
from .ml.ml_engine import get_symptoms, get_diseases, predict, get_accuracy_summary

logger = logging.getLogger(__name__)

CHARTS_DIR = os.path.join(settings.MEDIA_ROOT, 'charts')


# ── Home / Predictor ─────────────────────────────────────────────────────────

def index(request):
    """Main prediction page."""
    symptoms  = get_symptoms()
    # Format for display: replace _ with space, title case
    symptom_display = [
        {'key': s, 'label': s.replace('_', ' ').title()}
        for s in symptoms
    ]

    accuracy = get_accuracy_summary()

    return render(request, 'predictor/index.html', {
        'symptoms': symptom_display,
        'diseases': get_diseases(),
        'accuracy': accuracy,
        'model_choices': list(accuracy.keys()),
    })


@require_POST
def predict_view(request):
    """Handle prediction form submission (AJAX or regular POST)."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = request.POST

    selected_symptoms = data.get('symptoms', [])
    model_name        = data.get('model', 'Random Forest')

    if not selected_symptoms:
        return JsonResponse({'error': 'Please select at least one symptom.'}, status=400)

    result = predict(selected_symptoms, model_name)

    if 'error' in result:
        return JsonResponse({'error': result['error']}, status=400)

    # Save to DB
    top = result['top_predictions'][0] if result['top_predictions'] else {}
    history_obj = PredictionHistory.objects.create(
        user           = request.user if request.user.is_authenticated else None,
        symptoms_json  = json.dumps(selected_symptoms),
        result_json    = json.dumps(result['top_predictions']),
        model_used     = model_name,
        top_disease    = top.get('disease', 'Unknown'),
        top_confidence = top.get('confidence', 0),
    )

    return JsonResponse({
        'success': True,
        'predictions': result['top_predictions'],
        'symptoms_used': result['input_symptoms'],
        'model_used': model_name,
        'history_id': history_obj.id,
    })


# ── Dashboard (stats + charts) ───────────────────────────────────────────────

def dashboard(request):
    """Analytics dashboard with ML performance charts."""
    accuracy  = get_accuracy_summary()
    recent    = PredictionHistory.objects.all()[:50]

    # Disease frequency from predictions
    disease_counts = {}
    for p in PredictionHistory.objects.all():
        d = p.top_disease.strip()
        disease_counts[d] = disease_counts.get(d, 0) + 1

    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Chart availability
    chart_files = {
        'model_comparison':   'charts/model_comparison.png',
        'confusion_matrix':   'charts/confusion_matrix.png',
        'disease_distribution': 'charts/disease_distribution.png',
        'performance_metrics': 'charts/performance_metrics.png',
    }

    charts_exist = {
        k: os.path.exists(os.path.join(settings.MEDIA_ROOT, v))
        for k, v in chart_files.items()
    }

    return render(request, 'predictor/dashboard.html', {
        'accuracy':     accuracy,
        'recent':       recent,
        'top_diseases': top_diseases,
        'chart_files':  chart_files,
        'charts_exist': charts_exist,
        'total_predictions': PredictionHistory.objects.count(),
    })


# ── History ──────────────────────────────────────────────────────────────────

@login_required
def history(request):
    """User's personal prediction history."""
    predictions = PredictionHistory.objects.filter(user=request.user)
    return render(request, 'predictor/history.html', {'predictions': predictions})


# ── Auth ─────────────────────────────────────────────────────────────────────

def register_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}!')
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'predictor/auth.html', {'form': form, 'mode': 'register'})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f'Welcome back, {user.username}!')
            return redirect('index')
    else:
        form = AuthenticationForm()
    return render(request, 'predictor/auth.html', {'form': form, 'mode': 'login'})


def logout_view(request):
    logout(request)
    return redirect('index')
