# Beam Lab web app - container image for a company server / Azure App Service / any Docker host.
#   docker build -t beamlab .
#   docker run -p 8501:8501 -v beamlab-data:/data beamlab
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BEAMLAB_DATA=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py beamcore.py store.py merge.py ./
COPY .streamlit/config.toml .streamlit/config.toml
# secrets.toml (sign-in settings) is NOT baked in - mount it or set it in the host's secret store:
#   -v /path/to/secrets.toml:/app/.streamlit/secrets.toml:ro

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
