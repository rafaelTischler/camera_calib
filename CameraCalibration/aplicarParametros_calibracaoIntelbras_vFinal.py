#!/usr/bin/env python
"""
Validação da calibração fisheye em tempo real via RTSP.
Mostra original (esquerda) e desdistorcido (direita) lado a lado.
"""

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ============================================================
# CONFIGURAÇÕES
# ============================================================
RTSP_URL = "rtsp://admin:nupedee7@192.168.1.4:554/cam/realmonitor?channel=1&subtype=1&proto=Onvif"
NPZ_PATH = "./camera_calib_fisheye.npz"
BALANCE  = 0.5  # 0.0 = corta mais, 1.0 = mantém tudo


# ============================================================
# CARREGA PARÂMETROS E PRÉ-CALCULA MAPAS
# ============================================================
def carregar_e_mapear(balance):
    data     = np.load(NPZ_PATH)
    K        = data["K"]
    D        = data["D"]
    img_shape = tuple(data["img_shape"])

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_shape, np.eye(3), balance=balance
    )
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, img_shape, cv2.CV_16SC2
    )
    return K, D, img_shape, mapx, mapy


# ============================================================
# STREAM AO VIVO - ANTES/DEPOIS LADO A LADO
# ============================================================
def stream_tempo_real():
    if not os.path.exists(NPZ_PATH):
        print(f"ERRO: {NPZ_PATH} não encontrado. Rode primeiro a calibração.")
        return

    balance = BALANCE
    print(f"Carregando parâmetros de: {NPZ_PATH}")
    K, D, img_shape, mapx, mapy = carregar_e_mapear(balance)
    print(f"Resolução: {img_shape} | Balance: {balance}")
    print(f"K:\n{K}")
    print(f"D:\n{D}\n")

    print(f"Conectando: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERRO: não foi possível conectar ao stream RTSP.")
        return

    print("Conectado!\n")
    print("Controles:")
    print("  [+] aumenta balance (preserva mais campo de visão)")
    print("  [-] diminui balance (corta mais as bordas pretas)")
    print("  [S] salvar frame atual como comparacao_fisheye.jpg")
    print("  [Q] ou [ESC] sair\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Aplica desdistorção
        dst = cv2.remap(frame, mapx, mapy,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

        # Monta lado a lado
        combined = np.hstack((frame, dst))
        scale = min(1.0, 1600 / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)

        meio = int(frame.shape[1] * scale)
        h_c  = combined.shape[0]

        # HUD
        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0),       (meio, 75),          (0, 0, 0), -1)
        cv2.rectangle(overlay, (meio, 0),    (combined.shape[1], 75), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, combined, 0.5, 0, combined)

        cv2.putText(combined, "ORIGINAL",      (10, 35),        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(combined, "DESDISTORCIDO", (meio + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(combined, f"balance={balance:.1f}", (meio + 10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(combined, "[+][-] balance  [S] salvar  [Q] sair",
                    (10, h_c - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Linha divisória
        cv2.line(combined, (meio, 0), (meio, h_c), (255, 255, 255), 1)

        cv2.imshow("Calibracao Fisheye - Antes vs Depois (tempo real)", combined)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break

        elif key == ord('+'):
            balance = round(min(1.0, balance + 0.1), 1)
            _, _, _, mapx, mapy = carregar_e_mapear(balance)
            print(f"Balance: {balance}")

        elif key == ord('-'):
            balance = round(max(0.0, balance - 0.1), 1)
            _, _, _, mapx, mapy = carregar_e_mapear(balance)
            print(f"Balance: {balance}")

        elif key == ord('s'):
            path = "./comparacao_fisheye.jpg"
            cv2.imwrite(path, combined)
            print(f"[SALVO] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado.")


if __name__ == "__main__":
    stream_tempo_real()