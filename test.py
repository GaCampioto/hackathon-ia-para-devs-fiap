import base64
import os
import requests
from ultralytics import YOLO
import cv2
import importlib
import config  # Primeiro, importe normalmente
importlib.reload(config)  # Força o reload do módulo

from config import EVOLUTION_API_URL, EVOLUTION_API_KEY, WHATSAPP_NUMBER  # Importa as configurações




def processar_video(video_input, modelo_yolo, intervalo=15):
    """Processa um vídeo detectando facas com YOLO e enviando alertas via WhatsApp."""
    model = YOLO(modelo_yolo)

    # Criar pasta para armazenar frames capturados
    frames_path = "./frames_alerta/"
    os.makedirs(frames_path, exist_ok=True)

    # Abrir o vídeo para leitura
    cap = cv2.VideoCapture(video_input)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Fim do vídeo

        frame_count += 1
        if frame_count % intervalo != 0:  # Capturar frame a cada 'intervalo' frames
            continue

        # Fazer inferência no frame atual
        results = model.predict(frame, conf=0.12)

        # Verificar se detectou uma faca (classe 'arma_branca' no YOLO)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                if class_id == 0:  # Classe 0 = Faca (segundo nosso treinamento)
                    frame_filename = os.path.join(frames_path, f"alerta_faca_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)  # Salvar frame do alerta
                    print(f"⚠️ Faca detectada no frame {frame_count}, imagem salva: {frame_filename}")

                    # Converter imagem para Base64 e enviar alerta
                    base64_image = convert_image_to_base64(frame_filename)
                    enviar_alerta_whatsapp(base64_image)

    cap.release()
    cv2.destroyAllWindows()
    print("Processamento do vídeo concluído!")

# Função para enviar um frame via Evolution API (WhatsApp)
def enviar_alerta_whatsapp(base64Image):
    url = EVOLUTION_API_URL
    headers = {
        "Content-Type": "application/json",
        "apikey": EVOLUTION_API_KEY
    }


    data = {
        "number": WHATSAPP_NUMBER,
        "options": {
            "delay": 1200,
            "presence": "composing"
        },
        "mediaMessage": {
            "mediatype": "image",
            "caption": f"⚠️ Alerta! Uma faca foi detectada! 🚨",
            "media": base64Image
        }
    }

    response = requests.post(url, json=data, headers=headers)
    print(f"✅ retorno : {response.text}")


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image

processar_video("./videos/video.mp4", "./armas_brancas_best.pt",30)

processar_video("./videos/video2.mp4", "./armas_brancas_best.pt",3)