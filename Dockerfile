# Resume Agent - small, non-root Python image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent.py eval.py ./
COPY evals/questions.json evals/

# Don't run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Interactive chat by default; override with `python eval.py` to run the evals
CMD ["python", "agent.py"]
