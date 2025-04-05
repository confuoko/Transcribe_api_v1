
import whisper
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import KMeans
from transformers import AutoTokenizer

import boto3
import os
from tempfile import NamedTemporaryFile


BERT_MODEL_NAME = 'bert-base-multilingual-cased'
TRANSCRIBE_MODEL_NAME = 'medium'

from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

load_dotenv()  # загрузит переменные из .env

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)
else:
    raise ValueError("Hugging Face token not found in environment variables.")


login(hf_token)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

def transcribe(file_path):
	print("TRANSCRIBING..")
	model = whisper.load_model(TRANSCRIBE_MODEL_NAME)
	return model.transcribe(file_path)


def diarize(file_path):
    print("DIARIZING..")
    vad_model = load_silero_vad()

    audio = read_audio(file_path)

    speech_timestamps = get_speech_timestamps(audio, vad_model, return_seconds=True)

    embedding_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models"
    )

    embeddings = []
    for segment in speech_timestamps:
        start = int(segment['start'] * 16000)
        end = int(segment['end'] * 16000)
        segment_wav = audio[start:end]
        segment_wav = segment_wav.unsqueeze(0)
        embedding = embedding_model.encode_batch(segment_wav)
        embeddings.append(embedding.squeeze().numpy())

    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(embeddings)
    return speech_timestamps, labels


def unite_results(transcribed_result, diarized_result, labels):
    print("UNITING RESULTS..")
    diarization_result = []
    base_string_res = []
    for i, segment in enumerate(diarized_result):
        speaker = f"Speaker_{labels[i] + 1}"
        diarization_result.append({
            "start": segment['start'],
            "end": segment['end'],
            "speaker": speaker
        })
    silero_vad_speakers = []
    for segment in transcribed_result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        max_overlap = 0
        best_speaker = None

        for diarization_segment in diarization_result:
            overlap_start = max(start, diarization_segment["start"])
            overlap_end = min(end, diarization_segment["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = diarization_segment["speaker"]

        if best_speaker is None:
            for diarization_segment in diarization_result:
                if diarization_segment["end"] >= start or diarization_segment["start"] <= end:
                    best_speaker = diarization_segment["speaker"]
                    break

        speaker = best_speaker if best_speaker else "Unknown"
        silero_vad_speakers.append((speaker, text))
        base_string_res.append(text)
    for view_res in silero_vad_speakers:
        print(f'Speaker: {view_res[0]} - {view_res[1]}')
    return base_string_res


def processFile(file_name):
    #file_path = os.path.join(os.path.expanduser("~"), file_name)
    # Получаем путь относительно текущей рабочей директории (где запускается скрипт)
    file_path = os.path.join(os.getcwd(), file_name)
    transcribed_result = transcribe(file_path)
    diarized_result, labels = diarize(file_path)
    processed_results = unite_results(transcribed_result, diarized_result, labels)
    return processed_results


#answer = processFile('Desktop\Transcribe_api_v1\interview1.wav')
answer = processFile('interview1.wav')
print(answer)

"""
# Настройки S3 (MinIO)
s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",  # Для MinIO
    aws_access_key_id="admin",
    aws_secret_access_key="adminpassword"
)

bucket_name = "audiotest"
file_key = "interview1.wav"

# Загружаем файл из S3 в временный файл
file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)

# Создаем временный файл
with NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(file_obj["Body"].read())
    temp_file_path = temp_file.name

# Передаем путь к файлу в processFile
answer = processFile(temp_file_path)
print(answer)
# Удаляем временный файл после обработки
os.remove(temp_file_path)
"""
