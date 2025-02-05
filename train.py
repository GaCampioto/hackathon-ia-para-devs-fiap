import os
from ultralytics import YOLO

# 1️⃣ Carregar o modelo YOLO pré-treinado
os.environ['YOLO_RESULTS_DIR'] = "/content/runs"
model = YOLO("yolov5s.pt")  # Modelo inicial, caso vá rodar com o pré treinado é só trocar pelo nome do pré treinado.

# 2️⃣ Caminho atualizado do data.yaml
data_yaml_path = "/PATH_TO/data.yaml"

# 3️⃣ Iniciar o treinamento com os dados locais
model.train(
    data="/PATH_TO/data.yaml",
    epochs=100,  # Inicialmente treinamos com 50
    imgsz=640,
    batch=16,
    patience=10,  # Para early stopping automático
    lr0=0.001,  # Taxa de aprendizado inicial
    optimizer="SGD"
)

# 4️⃣ Salvar o modelo treinado
model_path = "/PATH_TO/armas_brancas.pt"
model.save(model_path)

print("Modelo treinado com sucesso!")
