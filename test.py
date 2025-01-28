from ultralytics import YOLO
import os

# Caminho para o modelo treinado
MODEL_PATH = "armas_brancas_e_armas_de_fogo.pt"

# Diretório com as imagens de teste
TEST_DIR = "./datasets/images/test/"  # Substitua pelo caminho da sua pasta

# Diretório de saída personalizado
OUTPUT_DIR = "./predictions/"  # Altere para o diretório desejado

# Certifique-se de que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carrega o modelo treinado
model = YOLO(MODEL_PATH)

# Realiza predições em todas as imagens da pasta de teste
results = model.predict(
    source=TEST_DIR,   # Diretório de entrada
    save=True,         # Salvar imagens com anotações
    save_txt=True,     # Salvar os resultados em formato texto (opcional)
    project=OUTPUT_DIR,  # Diretório raiz para salvar os resultados
    name="results",  # Subdiretório específico
    conf=0.5           # Confiança mínima para detecção
)

# Exibe os resultados
print("Resultados da Detecção:")
for result in results:
    print(result)

print(f"Imagens processadas salvas em: {os.path.join(OUTPUT_DIR, 'predictions')}")
