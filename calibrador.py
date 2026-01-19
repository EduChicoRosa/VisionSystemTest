import cv2
import json
import numpy as np

# --- CONFIGURAÇÕES ---
ARQUIVO_CONFIG = "config_bandeja.json"
RAIO_PADRAO = 60
CAMERA_ID = 1 

pontos_bandeja = [] # 4 pontos dos cantos
pontos_bercos = []  # 8 pontos dos centros

def mouse_callback(event, x, y, flags, param):
    global pontos_bandeja, pontos_bercos
    if event == cv2.EVENT_LBUTTONDOWN:
        # Primeiro coleta os 4 cantos da bandeja
        if len(pontos_bandeja) < 4:
            pontos_bandeja.append((x, y))
            print(f"Canto da Bandeja {len(pontos_bandeja)}: ({x}, {y})")
        # Depois os 8 berços
        elif len(pontos_bercos) < 8:
            pontos_bercos.append((x, y))
            print(f"Berço {len(pontos_bercos)}: ({x}, {y})")

def calibrar():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Calibrador Profissional", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrador Profissional", 1280, 720)
    cv2.setMouseCallback("Calibrador Profissional", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret: break

        img_draw = frame.copy()

        # 1. Desenha o polígono da bandeja
        if len(pontos_bandeja) > 0:
            for p in pontos_bandeja: cv2.circle(img_draw, p, 5, (0, 0, 255), -1)
            if len(pontos_bandeja) > 1:
                pts = np.array(pontos_bandeja, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_draw, [pts], len(pontos_bandeja) == 4, (0, 255, 255), 2)

        # 2. Desenha os berços
        for i, p in enumerate(pontos_bercos):
            cv2.circle(img_draw, p, 5, (0, 255, 0), -1)
            cv2.circle(img_draw, p, RAIO_PADRAO, (255, 0, 0), 1)

        # Instruções na tela
        msg = "Clique nos 4 CANTOS da BANDEJA" if len(pontos_bandeja) < 4 else "Clique nos 8 CENTROS dos BERCOS"
        if len(pontos_bercos) == 8: msg = "Calibracao Concluida! Pressione qualquer tecla."
        cv2.putText(img_draw, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Calibrador Profissional", img_draw)
        
        if len(pontos_bercos) == 8:
            cv2.waitKey(2000)
            break
        if cv2.waitKey(1) & 0xFF == 27: return

    # Salva JSON com o polígono da bandeja e os berços
    config_data = {
        "limite_bandeja": pontos_bandeja,
        "raio_padrao": RAIO_PADRAO,
        "posicoes": []
    }
    for i, p in enumerate(pontos_bercos):
        row, col = divmod(i, 4)
        config_data["posicoes"].append({"id": i+1, "matriz_idx": [row, col], "centro": p})

    with open(ARQUIVO_CONFIG, "w") as f:
        json.dump(config_data, f, indent=4)
    
    print("Configuração salva!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrar()