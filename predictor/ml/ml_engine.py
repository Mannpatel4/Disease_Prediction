"""
ml_engine.py
============
Loads trained models from disk (once, at startup) and provides
the predict() function used by Django views.
"""

import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Model directory (same folder as this file) ───────────────────────────────
ML_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_SYMPTOMS       = None
_LABEL_ENCODER  = None
_MODELS         = {}
_ACCURACY       = {}

# ── Doctor specialization map for each disease ───────────────────────────────
DOCTOR_MAP = {
    'Fungal infection':        'Dermatologist',
    'Allergy':                 'Allergist / Immunologist',
    'GERD':                    'Gastroenterologist',
    'Chronic cholestasis':     'Hepatologist',
    'Drug Reaction':           'Allergist',
    'Peptic ulcer diseae':     'Gastroenterologist',
    'AIDS':                    'Infectious Disease Specialist',
    'Diabetes ':               'Endocrinologist',
    'Gastroenteritis':         'Gastroenterologist',
    'Bronchial Asthma':        'Pulmonologist',
    'Hypertension ':           'Cardiologist',
    'Migraine':                'Neurologist',
    'Cervical spondylosis':    'Orthopedist / Neurologist',
    'Paralysis (brain hemorrhage)': 'Neurologist',
    'Jaundice':                'Hepatologist',
    'Malaria':                 'Infectious Disease Specialist',
    'Chicken pox':             'General Physician / Dermatologist',
    'Dengue':                  'Infectious Disease Specialist',
    'Typhoid':                 'General Physician',
    'hepatitis A':             'Hepatologist',
    'Hepatitis B':             'Hepatologist',
    'Hepatitis C':             'Hepatologist',
    'Hepatitis D':             'Hepatologist',
    'Hepatitis E':             'Hepatologist',
    'Alcoholic hepatitis':     'Hepatologist',
    'Tuberculosis':            'Pulmonologist',
    'Common Cold':             'General Physician',
    'Pneumonia':               'Pulmonologist',
    'Dimorphic hemmorhoids(piles)': 'Proctologist',
    'Heart attack':            'Cardiologist',
    'Varicose veins':          'Vascular Surgeon',
    'Hypothyroidism':          'Endocrinologist',
    'Hyperthyroidism':         'Endocrinologist',
    'Hypoglycemia':            'Endocrinologist',
    'Osteoarthritis':          'Orthopedist / Rheumatologist',
    'Arthritis':               'Rheumatologist',
    '(vertigo) Paroymsal  Positional Vertigo': 'Neurologist / ENT',
    'Acne':                    'Dermatologist',
    'Urinary tract infection': 'Urologist',
    'Psoriasis':               'Dermatologist',
    'Impetigo':                'Dermatologist',
}


def _load():
    """Load all pkl files once. Subsequent calls are no-ops."""
    global _SYMPTOMS, _LABEL_ENCODER, _MODELS, _ACCURACY

    if _SYMPTOMS is not None:
        return

    try:
        with open(os.path.join(ML_DIR, 'symptoms.pkl'), 'rb') as f:
            _SYMPTOMS = pickle.load(f)

        with open(os.path.join(ML_DIR, 'label_encoder.pkl'), 'rb') as f:
            _LABEL_ENCODER = pickle.load(f)

        with open(os.path.join(ML_DIR, 'accuracy_summary.pkl'), 'rb') as f:
            _ACCURACY = pickle.load(f)

        model_files = {
            'Naive Bayes':   'naive_bayes.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'Random Forest': 'random_forest.pkl',
        }
        for name, fname in model_files.items():
            path = os.path.join(ML_DIR, fname)
            with open(path, 'rb') as f:
                _MODELS[name] = pickle.load(f)

        logger.info(f"ML Engine loaded: {len(_SYMPTOMS)} symptoms, {len(_LABEL_ENCODER.classes_)} diseases")

    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        raise RuntimeError(f"ML models not found. Please run: python manage.py train_models\n{e}")


def get_symptoms():
    """Return list of all symptom column names."""
    _load()
    return _SYMPTOMS


def get_diseases():
    """Return sorted list of all disease names."""
    _load()
    return sorted(_LABEL_ENCODER.classes_.tolist())


def get_accuracy_summary():
    """Return {model_name: {accuracy, cv_mean}} dict."""
    _load()
    return _ACCURACY


def predict(selected_symptoms: list, model_name: str = 'Random Forest') -> dict:
    """
    Predict diseases from a list of selected symptom names.

    Args:
        selected_symptoms: e.g. ['itching', 'skin_rash', 'fever']
        model_name: 'Naive Bayes' | 'Decision Tree' | 'Random Forest'

    Returns:
        {
            'top_predictions': [{'disease': str, 'confidence': float, 'doctor': str}, ...],
            'input_symptoms': [str, ...],
            'model_used': str,
        }
    """
    _load()

    if not selected_symptoms:
        return {'error': 'No symptoms selected.'}

    # Build binary input vector
    input_vector = np.zeros(len(_SYMPTOMS))
    matched = []
    for sym in selected_symptoms:
        sym_clean = sym.strip()
        if sym_clean in _SYMPTOMS:
            idx = _SYMPTOMS.index(sym_clean)
            input_vector[idx] = 1
            matched.append(sym_clean)

    if not matched:
        return {'error': 'None of the selected symptoms matched known features.'}

    model = _MODELS.get(model_name, _MODELS['Random Forest'])

    # Get probability scores
    proba = model.predict_proba(input_vector.reshape(1, -1))[0]

    # Top 3 predictions
    top_indices = np.argsort(proba)[::-1][:3]
    top_predictions = []
    for idx in top_indices:
        disease = _LABEL_ENCODER.inverse_transform([idx])[0]
        confidence = round(float(proba[idx]) * 100, 2)
        if confidence > 0:
            top_predictions.append({
                'disease':    disease.strip(),
                'confidence': confidence,
                'doctor':     DOCTOR_MAP.get(disease, 'General Physician'),
            })

    return {
        'top_predictions': top_predictions,
        'input_symptoms':  matched,
        'model_used':      model_name,
    }
