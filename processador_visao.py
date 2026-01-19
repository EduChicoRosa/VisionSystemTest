import cv2
import json
import numpy as np

# --- CONFIGURAÇÕES TÉCNICAS ---
ARQUIVO_CONFIG = "config_bandeja.json"
CAMERA_ID = 1

# Ajuste conforme sua montagem (distância camera-peça)
ESC_PPMM = 3.8  # Exemplo: 3.8 pixels para cada 1mm
DIAMETRO_PECA_MM = 60
DIAMETRO_FURO_MM = 25 # Estimativa para o furo central

THRESHOLD_BRILHO = 140 
ALPHA = 0.4

class ProcessadorAnelar:
    def __init__(self):
        with open(ARQUIVO_CONFIG, "r") as f:
            self.config = json.load(f)
        
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Cálculos de Raio em Pixels
        self.raio_ext = int((DIAMETRO_PECA_MM * ESC_PPMM) / 2)
        self.raio_int = int((DIAMETRO_FURO_MM * ESC_PPMM) / 2)

    def criar_mascara_anelar(self, shape, centro):
        """Cria uma máscara preta com um anel branco (ROI)"""
        mask = np.zeros(shape[:2], dtype="uint8")
        # Desenha o círculo externo preenchido
        cv2.circle(mask, centro, self.raio_ext, 255, -1)
        # 'Subtrai' o furo interno (pintando de preto)
        cv2.circle(mask, centro, self.raio_int, 0, -1)
        return mask

    def processar(self):
        cv2.namedWindow("WMS Vision - Detector de Anéis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("WMS Vision - Detector de Anéis", 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            overlay = np.zeros_like(frame)
            contagem = 0

            for pos in self.config["posicoes"]:
                centro = tuple(pos["centro"])
                
                # 1. Lógica de Detecção no Anel
                mask = self.criar_mascara_anelar(frame.shape, centro)
                # cv2.mean retorna a média apenas onde a máscara é branca
                media_brilho = cv2.mean(frame, mask=mask)[0]

                ocupado = media_brilho > THRESHOLD_BRILHO
                cor = (0, 255, 0) if ocupado else (255, 191, 0)
                
                if ocupado: contagem += 1

                # 2. Visualização AR em formato de anel
                # Desenha o anel no overlay (espessura = diferença dos raios)
                espessura = self.raio_ext - self.raio_int
                raio_meio = self.raio_int + (espessura // 2)
                cv2.circle(overlay, centro, raio_meio, cor, espessura)
                
                # Texto de ID
                cv2.putText(frame, f"P{pos['id']}", (centro[0]-15, centro[1]+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # Fusão
            frame_final = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

            # Dashboard
            cv2.rectangle(frame_final, (0,0), (380, 60), (30,30,30), -1)
            cv2.putText(frame_final, f"ESTATORES: {contagem}/8", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

            cv2.imshow("WMS Vision - Detector de Anéis", frame_final)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ProcessadorAnelar()
    detector.processar()