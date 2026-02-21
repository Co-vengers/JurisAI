set -e

python - <<'PY'
from utils import Utils
Utils().preprocess_and_embed_data()
PY

exec gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 2 app:app
