FROM python:3.10

WORKDIR /Transcribe_api_v1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "core.py"]
