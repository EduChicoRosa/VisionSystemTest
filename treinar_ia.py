from ultralytics import YOLO
import torch

def treinar_modelo():
    # 1. Carrega um modelo pré-treinado (YOLOv11 Nano - o mais rápido e leve)
    # Se o computador for antigo, ele baixará o arquivo .pt na primeira vez
    model = YOLO("yolo11n.pt") 

    # 2. Inicia o treinamento
    # data: caminho para o seu arquivo .yaml
    # epochs: quantas vezes a IA verá as fotos (100 é um bom começo para 39 fotos)
    # imgsz: tamanho da imagem no treino (640 é o padrão equilibrado)
    # device: detecta automaticamente GPU (0) ou CPU ('cpu')
    
    print("Iniciando treinamento do Detector de Estatores...")
    
    results = model.train(
        data="dataset.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=16,          # Ajuste para 8 se der erro de memória (Out of Memory)
        name="wms_estatores",
        device=0 if torch.cuda.is_available() else "cpu"
    )

    print("\n--- TREINAMENTO CONCLUÍDO ---")
    print("O modelo treinado foi salvo em: runs/detect/wms_estatores/weights/best.pt")

if __name__ == "__main__":
    treinar_modelo()