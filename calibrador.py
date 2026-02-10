import cv2
import json
import numpy as np
import os

# --- CONFIGURAÇÕES DA BANDEJA ---
LINHAS = 6   # Horizontal (Eixo Y na imagem)
COLUNAS = 10 # Vertical (Eixo X na imagem)
ARQUIVO_CONFIG = "config_bandeja_real.json"
RAIO_PADRAO = 40 
CAMERA_ID = 1 # Ajuste para 0 ou 1 dependendo da sua GoPro

pontos_clicados = []

def ordenar_pontos(pts):
    """
    Ordena os 4 pontos clicados para garantir a matriz de perspectiva correta:
    [Superior Esquerdo, Superior Direito, Inferior Direito, Inferior Esquerdo]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Soma mínima = Top-Left
    rect[2] = pts[np.argmax(s)] # Soma máxima = Bottom-Right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Diferença mínima = Top-Right
    rect[3] = pts[np.argmax(diff)] # Diferença máxima = Bottom-Left
    return rect

def gerar_grade_perspectiva(pts_clicados):
    """
    Mapeia a bandeja real para um plano retangular perfeito e gera os 60 centros.
    """
    pts_origem = ordenar_pontos(np.array(pts_clicados, dtype="float32"))
    
    # Definimos um retângulo virtual com proporção 10:6
    largura_v, altura_v = 1000, 600
    pts_destino = np.array([
        [0, 0], [largura_v, 0], 
        [largura_v, altura_v], [0, altura_v]
    ], dtype="float32")

    # Matriz de Transformação de Perspectiva
    M = cv2.getPerspectiveTransform(pts_origem, pts_destino)
    M_inv = np.linalg.inv(M)

    grade = []
    for r in range(LINHAS):
        for c in range(COLUNAS):
            # Coordenadas no espaço virtual 'desentortado'
            tx = (c + 0.5) * (largura_v / COLUNAS)
            ty = (r + 0.5) * (altura_v / LINHAS)
            
            # Projeção inversa: traz o ponto virtual de volta para a lente da GoPro
            p = np.array([tx, ty, 1], dtype="float32")
            p_real = M_inv @ p
            p_real /= p_real[2] # Normalização Z

            grade.append({
                "id": len(grade) + 1,
                "matriz_idx": [r, c],
                "centro": [int(p_real[0]), int(p_real[1])]
            })
    return grade

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pontos_clicados) < 4:
            pontos_clicados.append((x, y))
            print(f"Ponto {len(pontos_clicados)} registrado: ({x}, {y})")

def calibrar():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    cv2.namedWindow("Calibrador de Precisao 6x10", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrador de Precisao 6x10", 1280, 720)
    cv2.setMouseCallback("Calibrador de Precisao 6x10", mouse_callback)

    print("\n--- INSTRUÇÕES ---")
    print("Clique nos 4 cantos da área de depósito dos estatores.")
    print("A ordem não importa (o script ordena automaticamente).")
    print("Pressione 'ESC' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        img_draw = frame.copy()
        
        # Desenha os pontos e as linhas de conexão
        for p in pontos_clicados:
            cv2.circle(img_draw, p, 8, (0, 0, 255), -1)
        
        if len(pontos_clicados) == 4:
            pts_ordenados = ordenar_pontos(np.array(pontos_clicados))
            pts_view = pts_ordenados.astype(int).reshape((-1, 1, 2))
            cv2.polylines(img_draw, [pts_view], True, (0, 255, 255), 3)
            cv2.putText(img_draw, "4 pontos OK! Pressione 'S' para salvar ou 'C' para limpar.", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Calibrador de Precisao 6x10", img_draw)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and len(pontos_clicados) == 4:
            grade = gerar_grade_perspectiva(pontos_clicados)
            config = {
                "limite_bandeja": pontos_clicados,
                "posicoes": grade,
                "raio_padrao": RAIO_PADRAO
            }
            with open(ARQUIVO_CONFIG, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Configuração salva em {ARQUIVO_CONFIG}!")
            break
        elif key == ord('c'):
            pontos_clicados.clear()
        elif key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrar()