import os
import sys

PROJECT_ROOT = os.path.abspath(0, os.path.os.path.join(
    os.path.dirname(__file__), '..', 'email_categorizer'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
for subdir in ['src', 'data']:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, subdir))
