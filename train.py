from ultralytics import YOLO

# Caminho para o arquivo de configuração do dataset
DATA_YAML = "datasets/data.yaml"  # Atualize com o caminho correto para o seu data.yaml

# Passo 1: Carregar o modelo pré-treinado YOLOv5
# model = YOLO("yolov5s.pt")  # Baixa o modelo pré-treinado pequeno
model = YOLO("armas_brancas_e_armas_de_fogo.pt")

# # Passo 2: Treinar o modelo com o dataset
# model.train(
#     data=DATA_YAML,  # Configuração do dataset
#     epochs=50,       # Número de épocas
#     imgsz=640,       # Tamanho das imagens (em pixels)
#     batch=16,        # Tamanho do lote
#     name="armas_brancas_e_armas_de_fogo"  # Nome do experimento
# )

# Passo 3: Salvar o modelo treinado
# model.save("armas_brancas_e_armas_de_fogo.pt")

# Passo 4: Testar o modelo treinado com novas imagens
results = model.predict(source="faca.jpeg", save=True, conf=0.5)
print(results)