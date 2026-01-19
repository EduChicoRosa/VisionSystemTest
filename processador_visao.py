import cv2
import json
import numpy as np
import os

# --- PARÂMETROS DE ENGENHARIA ---
ARQUIVO_CONFIG = "config_bandeja.json"
CAMERA_ID = 1
ESC_PPMM = 3.8          # Pixels por milímetro (Ajuste conforme sua régua)
DIAMETRO_PECA_MM = 60
DIAMETRO_FURO_MM = 25 
THRESHOLD_BRILHO = 160  # Sensibilidade para os berços (Verde/Azul)
THRESHOLD_GLOBAL = 180  # Sensibilidade para detectar intrusos (Vermelho)
TOLERANCIA_PX = 60      # Distância máxima do centro para não ser erro
ALPHA_AR = 0.4          # Transparência da Realidade Aumentada

class ProcessadorWMS_Final:
    def __init__(self, gravar=True):
        self._carregar_config()
        
        # Inicialização da GoPro
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Cálculos dimensionais
        self.raio_ext = int((DIAMETRO_PECA_MM * ESC_PPMM) / 2)
        self.raio_int = int((DIAMETRO_FURO_MM * ESC_PPMM) / 2)
        
        # Máscara Poligonal da Bandeja (Filtro de Falsos Alarmes)
        self.mask_bandeja = np.zeros((1080, 1920), dtype=np.uint8)
        pts = np.array(self.config["limite_bandeja"], np.int32)
        cv2.fillPoly(self.mask_bandeja, [pts], 255)

        # Configuração de Gravação
        self.gravar = gravar
        if self.gravar:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter('demo_final_wms.mp4', fourcc, 20.0, (1920, 1080))
            print("Gravando em: demo_final_wms.mp4")

    def _carregar_config(self):
        if not os.path.exists(ARQUIVO_CONFIG):
            print("ERRO: Execute o calibrador primeiro!")
            exit()
        with open(ARQUIVO_CONFIG, "r") as f:
            self.config = json.load(f)

    def criar_mascara_anelar(self, centro):
        mask = np.zeros((1080, 1920), dtype="uint8")
        cv2.circle(mask, centro, self.raio_ext, 255, -1)
        cv2.circle(mask, centro, self.raio_int, 0, -1)
        return mask

    def calcular_distancia(self, p1, p2):
        """Calcula a distância euclidiana entre dois pontos."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def processar(self):
        cv2.namedWindow("Sistema de Visao WMS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sistema de Visao WMS", 1280, 720)

        print("Monitoramento ativo. 'q' para sair.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break

                overlay = np.zeros_like(frame)
                contagem_ok = 0
                alerta_erro = False

                # --- 1. DETECÇÃO NOS BERÇOS OFICIAIS ---
                for pos in self.config["posicoes"]:
                    centro = tuple(pos["centro"])
                    mask_anelar = self.criar_mascara_anelar(centro)
                    brilho = cv2.mean(frame, mask=mask_anelar)[0]

                    ocupado = brilho > THRESHOLD_BRILHO
                    cor = (0, 255, 0) if ocupado else (255, 191, 0) # Verde / Azul
                    
                    if ocupado: contagem_ok += 1

                    # Desenho do Anel AR
                    espessura = self.raio_ext - self.raio_int
                    r_meio = self.raio_int + (espessura // 2)
                    cv2.circle(overlay, centro, r_meio, cor, espessura)
                    cv2.putText(frame, f"P{pos['id']}", (centro[0]-15, centro[1]+5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # --- 2. DETECÇÃO DE INTRUSOS (CÍRCULO VERMELHO) ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, THRESHOLD_GLOBAL, 255, cv2.THRESH_BINARY)
                
                # Aplica a Máscara da Bandeja (ignora tudo que está fora do polígono)
                thresh_focada = cv2.bitwise_and(thresh, thresh, mask=self.mask_bandeja)
                
                contours, _ = cv2.findContours(thresh_focada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) > 2500: # Filtro de tamanho mínimo
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            
                            # Verifica se o objeto está "perdido" (longe dos berços)
                            esta_no_berco = False
                            for pos in self.config["posicoes"]:
                                if self.calcular_distancia((cX, cY), pos["centro"]) < TOLERANCIA_PX:
                                    esta_no_berco = True
                                    break
                            
                            if not esta_no_berco:
                                alerta_erro = True
                                # Desenha o Círculo Vermelho AR
                                cv2.circle(overlay, (cX, cY), self.raio_ext + 10, (0, 0, 255), -1)

                # --- 3. COMPOSIÇÃO FINAL E UI ---
                frame_final = cv2.addWeighted(overlay, ALPHA_AR, frame, 1 - ALPHA_AR, 0)

                # Dashboard superior dinâmico
                bg_cor = (0, 0, 160) if alerta_erro else (40, 40, 40)
                cv2.rectangle(frame_final, (0,0), (600, 75), bg_cor, -1)
                cv2.putText(frame_final, f"ESTATORES: {contagem_ok}/8", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                
                if alerta_erro:
                    cv2.putText(frame_final, "ALERTA: POSICIONAMENTO INCORRETO", (20, 110), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                # Gravação e Exibição
                if self.gravar: self.out.write(frame_final)
                cv2.imshow("Sistema de Visao WMS", frame_final)
                
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        finally:
            self.cap.release()
            if self.gravar: self.out.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ProcessadorWMS_Final(gravar=True)
    app.processar()