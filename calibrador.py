import cv2
import json
import numpy as np

# Configurações
ARQUIVO_CONFIG = "config_bandeja.json"
RAIO_PADRAO = 60
CAMERA_ID = 1 

pontos_clicados = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pontos_clicados) < 8:
            # Importante: O clique é registrado na imagem original (1920x1080)
            # mesmo que a janela de exibição seja menor.
            pontos_clicados.append((x, y))
            print(f"Ponto {len(pontos_clicados)} registrado: ({x}, {y})")

def calibrar():
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # Forçamos a GoPro a entregar 1080p para evitar o "zoom" de baixa resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Criamos uma janela redimensionável
    cv2.namedWindow("Calibrador de Bandeja", cv2.WINDOW_NORMAL)
    # Redimensionamos a janela para caber no seu monitor (ex: 1280x720)
    cv2.resizeWindow("Calibrador de Bandeja", 1280, 720)
    cv2.setMouseCallback("Calibrador de Bandeja", mouse_callback)

    print("Iniciando calibração...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_draw = frame.copy()

        # Desenha feedbacks
        for i, p in enumerate(pontos_clicados):
            cv2.circle(img_draw, p, 5, (0, 255, 255), -1)
            cv2.circle(img_draw, p, RAIO_PADRAO, (255, 0, 0), 2)
            cv2.putText(img_draw, f"P{i+1}", (p[0]-10, p[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Exibe a imagem. O OpenCV vai ajustar o frame de 1080p 
        # para o tamanho da janela de 720p que definimos acima.
        cv2.imshow("Calibrador de Bandeja", img_draw)

        if len(pontos_clicados) == 8 or (cv2.waitKey(1) & 0xFF == 27):
            break

    # Salva o JSON
    if len(pontos_clicados) == 8:
        config_data = {"raio_padrao": RAIO_PADRAO, "posicoes": []}
        for i, (cx, cy) in enumerate(pontos_clicados):
            row, col = divmod(i, 4)
            config_data["posicoes"].append({
                "id": i + 1,
                "matriz_idx": [row, col],
                "centro": [cx, cy]
            })

        with open(ARQUIVO_CONFIG, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"\nCalibração salva com sucesso!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrar()