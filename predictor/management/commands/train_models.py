from django.core.management.base import BaseCommand
import subprocess, sys, os

class Command(BaseCommand):
    help = 'Train all ML models (Naive Bayes, Decision Tree, Random Forest)'

    def handle(self, *args, **kwargs):
        script = os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'train_models.py')
        self.stdout.write('Training models...')
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode == 0:
            self.stdout.write(self.style.SUCCESS('✅ Models trained successfully!'))
        else:
            self.stdout.write(self.style.ERROR('❌ Training failed.'))
