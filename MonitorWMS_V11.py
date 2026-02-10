import cv2
import json
import numpy as np
from ultralytics import YOLO
import time
import sys

# --- CONFIGURAÇÕES DE PRECISÃO ---
ARQUIVO_CONFIG = "config_bandeja_real.json"
CAMINHO_MODELO = r"C:\VisionTest\runs\detect\wms_estatores\weights\best.pt"
CAMERA_ID = 1
CONF_MINIMA = 0.45    # Levemente menor para não perder peças entre colunas
TOLERANCIA_PX = 38    # Valor de equilíbrio para evitar falsos positivos
ALPHA_AR = 0.35

# Filtros Geométricos (1.0 = Quadrado Perfeito)
AR_MIN, AR_MAX = 0.85, 1.25 

# Rastreio das Barras Amarelas
AMARELO_LOW = np.array([15, 65, 65])
AMARELO_HIGH = np.array([40, 255, 255])

class SistemaWMS_V10_2:
    def __init__(self, gravar=True):
        print("--- INICIANDO HANDSHAKE ROBUSTO ---")
        
        # 1. CONEXÃO COM A GOPRO
        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        camera_ok = False
        for i in range(40):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"Vídeo OK na tentativa {i}!")
                camera_ok = True
                break
            time.sleep(0.1)
            
        if not camera_ok:
            print("ERRO: GoPro não respondeu. Tente reconectar o USB.")
            sys.exit()

        # 2. CARREGAMENTO DA INTELIGÊNCIA
        print("Carregando IA...")
        self.model = YOLO(CAMINHO_MODELO)
        
        with open(ARQUIVO_CONFIG, "r") as f:
            self.config = json.load(f)
        
        self.gravar = gravar
        self.ref_barras = None
        self.ultimo_offset = np.array([0, 0])
        self.raio = self.config["raio_padrao"]
        self.pts_bandeja = np.array(self.config["limite_bandeja"], np.int32)
        
        # Detector de Movimento
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=60, detectShadows=False)
        self.frames_pos_intervencao = 0

        if self.gravar:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(f'DEMO_SUCESSO_{int(time.time())}.mp4', fourcc, 15.0, (1920, 1080))

    def processar(self):
        cv2.namedWindow("Monitor WMS - Demo Final", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Monitor WMS - Demo Final", 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret: break

            # Detecção de Intervenção
            mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_roi, [self.pts_bandeja], 255)
            fg_mask = self.backSub.apply(frame)
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=mask_roi)
            massa = np.sum(fg_mask > 0)

            intervencao = massa > 25000 
            if intervencao: self.frames_pos_intervencao = 3
            elif self.frames_pos_intervencao > 0:
                self.frames_pos_intervencao -= 1
                intervencao = True

            overlay = np.zeros_like(frame)
            count_ok, num_erros = 0, 0

            if not intervencao:
                # Alinhamento Amarelo Lateral
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_am = cv2.inRange(hsv, AMARELO_LOW, AMARELO_HIGH)
                cnts, _ = cv2.findContours(mask_am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                barras = []
                for c in cnts:
                    x, y, w, h = cv2.boundingRect(c)
                    if (x < 450 or x > 1470) and cv2.contourArea(c) > 1000:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            barras.append(np.array([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])]))
                
                if self.ref_barras is None and len(barras) >= 2:
                    self.ref_barras = barras
                elif self.ref_barras is not None and len(barras) > 0:
                    num = min(len(barras), len(self.ref_barras))
                    self.ultimo_offset = np.mean([barras[i] - self.ref_barras[i] for i in range(num)], axis=0).astype(int)

                # IA + Filtro de ROI da Bandeja
                results = self.model(frame, conf=CONF_MINIMA, verbose=False, imgsz=640)[0]
                bercos_ocupados = set()
                pecas_erro = []

                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    centro_ia = np.array([int((x1+x2)/2), int((y1+y2)/2)])
                    
                    # --- NOVO FILTRO: Ignora o que está fora da bandeja ---
                    dentro_da_bandeja = cv2.pointPolygonTest(self.pts_bandeja, (float(centro_ia[0]), float(centro_ia[1])), False) >= 0
                    if not dentro_da_bandeja:
                        continue # Pula para a próxima detecção
                    
                    w, h = x2 - x1, y2 - y1
                    geometria_ok = AR_MIN <= (w/h) <= AR_MAX
                    
                    encontrou_berco = False
                    for i, pos in enumerate(self.config["posicoes"]):
                        c_berco = np.array(pos["centro"]) + self.ultimo_offset
                        if np.linalg.norm(centro_ia - c_berco) < TOLERANCIA_PX:
                            if geometria_ok:
                                bercos_ocupados.add(i)
                                encontrou_berco = True
                                break
                    
                    if not encontrou_berco:
                        pecas_erro.append(centro_ia)

                # Desenho
                for i, pos in enumerate(self.config["posicoes"]):
                    c = tuple(np.array(pos["centro"]) + self.ultimo_offset)
                    cor = (0, 255, 0) if i in bercos_ocupados else (255, 120, 0)
                    cv2.circle(overlay, c, self.raio, cor, -1)
                
                for p_e in pecas_erro:
                    cv2.circle(overlay, tuple(p_e), self.raio + 10, (0, 0, 255), -1)
                
                count_ok, num_erros = len(bercos_ocupados), len(pecas_erro)

            # UI e Dashboard
            frame_final = cv2.addWeighted(overlay, ALPHA_AR, frame, 1-ALPHA_AR, 0)
            status_cor = (0, 165, 255) if intervencao else (40, 40, 40)
            if not intervencao and num_erros > 0: status_cor = (0, 0, 160)

            cv2.rectangle(frame_final, (0,0), (850, 110), status_cor, -1)
            msg = "INTERVENCAO EM CURSO..." if intervencao else "MONITORAMENTO ATIVO"
            if not intervencao and num_erros > 0: msg = f"ALERTA: {num_erros} IRREGULARIDADE(S)!"
            
            cv2.putText(frame_final, f"WMS MONITOR: {count_ok if not intervencao else '--'}/60", (20, 50), 2, 1.4, (255,255,255), 4)
            cv2.putText(frame_final, msg, (20, 95), 2, 0.8, (255,255,255), 2)

            frame_disp = cv2.resize(frame_final, (1280, 720))
            cv2.imshow("Monitor WMS - Demo Final", frame_disp)
            
            if self.gravar: self.out.write(frame_final)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        if self.gravar: self.out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SistemaWMS_V10_2(gravar=True)
    app.processar()