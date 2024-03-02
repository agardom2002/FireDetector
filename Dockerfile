FROM python:3.8
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
RUN pip install streamlit ultralytics opencv-python-headless streamlit-webrtc
COPY src/app.py /app/
COPY src/.env /app/
COPY src/MandarCorreo.py /app/
COPY model/best.pt /app/model/best.pt
# COPY data/* /app/data/
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py"]
