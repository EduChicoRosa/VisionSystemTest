import cv2
import json
import numpy as np
import os

# --- CONFIGURAÇÕES ---
ARQUIVO_CONFIG = "config_bandeja.json"
CAMERA_ID = 1
THRESHOLD_BRILHO = 120  # Ajuste conforme sua iluminação (0 a 255)
ALPHA = 0.4             # Transparência dos círculos (0.1 a 0.9)

# CORES (Formato BGR)
COR_VAZIO = (255, 191, 0)   # Azul Claro (Berço disponível)
COR_OCUPADO = (0, 255, 0)   # Verde (Peça detectada)
COR_TEXTO = (255, 255, 255)

class DetectorBandeja:
    def __init__(self):
        self.config = self._carregar_config()
        self.cap = cv2.VideoCapture(CAMERA_ID)
        # Forçamos a mesma resolução usada na calibração
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def _carregar_config(self):
        if not os.path.exists(ARQUIVO_CONFIG):
            print(f"ERRO: Arquivo {ARQUIVO_CONFIG} não encontrado!")
            exit()
        with open(ARQUIVO_CONFIG, "r") as f:
            return json.load(f)

    def processar(self):
        cv2.namedWindow("Validacao WMS Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Validacao WMS Vision", 1280, 720)

        print("Processador iniciado. Pressione 'q' para sair.")

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # Criamos uma camada preta para desenhar os círculos translúcidos
            overlay = np.zeros_like(frame)
            contagem = 0
            raio = self.config["raio_padrao"]

            for pos in self.config["posicoes"]:
                cx, cy = pos["centro"]
                
                # Definimos a região circular para análise
                # Criamos uma máscara preta com um círculo branco na posição
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.circle(mask, (cx, cy), raio, 255, -1)
                
                # Calculamos o brilho médio apenas dentro do círculo
                media_brilho = cv2.mean(frame, mask=mask)[0]

                # Lógica de Ocupação
                ocupado = media_brilho > THRESHOLD_BRILHO
                cor = COR_OCUPADO if ocupado else COR_VAZIO
                
                if ocupado: contagem += 1

                # Desenha o círculo preenchido no overlay
                cv2.circle(overlay, (cx, cy), raio, cor, -1)
                # Opcional: ID da posição para debug
                cv2.putText(overlay, f"P{pos['id']}", (cx-15, cy+5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # Fusão da imagem real com o overlay translúcido
            # frame_final = frame * (1 - ALPHA) + overlay * ALPHA
            frame_final = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

            # Painel de Informações (Dashboard)
            cv2.rectangle(frame_final, (0, 0), (350, 80), (40, 40, 40), -1)
            cv2.putText(frame_final, f"ESTATORES: {contagem}/8", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COR_TEXTO, 3)

            cv2.imshow("Validacao WMS Vision", frame_final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DetectorBandeja()
    detector.processar()