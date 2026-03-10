from models import get_model, MODEL_NAMES
ar_names = [n for n in MODEL_NAMES if n.startswith('ar_')]
print('AR models:', ar_names)
import torch
m = get_model('ar_mean_flow', input_size=15)
print('ar_mean_flow OK:', type(m).__name__)
