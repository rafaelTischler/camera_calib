#!/usr/bin/env python
"""
Calibração fisheye da câmera Intelbras (teto do laboratório).
Captura com detecção em tempo real do tabuleiro via RTSP.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))

# Redireciona erros internos do OpenCV para arquivo (limpa o console)
_stderr_log = open(os.path.join(_script_dir, "opencv_errors.log"), "w")
os.dup2(_stderr_log.fileno(), 2)

import cv2
import numpy as np
import glob

cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ============================================================
# CONFIGURAÇÕES
# ============================================================
RTSP_URL = "rtsp://admin:nupedee7@192.168.1.4:554/cam/realmonitor?channel=1&subtype=1&proto=Onvif"

CHECKERBOARD = (8, 6)
SQUARE_SIZE = 5.0
IMAGES_DIR       = os.path.join(_script_dir, "images_intelbras")
TARGET_IMAGES    = 25
DETECCAO_A_CADA  = 144   # roda detecção a cada N frames (aumente se travar)

os.makedirs(IMAGES_DIR, exist_ok=True)

subpix_criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
fisheye_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

FISHEYE_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
    cv2.fisheye.CALIB_FIX_SKEW
)


# ============================================================
# ETAPA 1: CAPTURA COM DETECÇÃO EM TEMPO REAL
# ============================================================
def tentar_detectar(gray):
    """Tenta detectar o tabuleiro com 3 estratégias em sequência."""

    # Tentativa 1: findChessboardCornersSB (mais robusto)
    try:
        ret, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        )
        if ret:
            return True, corners.astype(np.float32), "SB"
    except cv2.error:
        pass

    # Tentativa 2: CLAHE + método clássico
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        ret2, corners2 = cv2.findChessboardCorners(
            gray_clahe, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret2:
            corners2 = cv2.cornerSubPix(gray_clahe, corners2, (11, 11), (-1, -1), subpix_criteria)
            return True, corners2.astype(np.float32), "classico+clahe"
    except cv2.error:
        pass

    # Tentativa 3: método clássico puro
    try:
        ret3, corners3 = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret3:
            corners3 = cv2.cornerSubPix(gray, corners3, (11, 11), (-1, -1), subpix_criteria)
            return True, corners3.astype(np.float32), "classico"
    except cv2.error:
        pass

    return False, None, ""


def capturar_imagens():
    count = len(glob.glob(os.path.join(IMAGES_DIR, "*.png")))

    print(f"Conectando: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERRO: não foi possível conectar ao stream RTSP.")
        return 0

    print("Conectado!\n")
    print("=== CAPTURA COM DETECÇÃO EM TEMPO REAL ===")
    print(f"Salvando em: {IMAGES_DIR}")
    print(f"Meta: {TARGET_IMAGES} imagens COM tabuleiro detectado.")
    print("  [ESPAÇO] = salvar (só quando tabuleiro estiver VERDE)")
    print("  [S]      = forçar salvar sem detecção")
    print("  [ESC]    = encerrar e ir para calibração\n")

    ultimo_ret     = False
    ultimos_corners = None
    ultimo_metodo  = ""
    frame_count    = 0

    while True:
        ret_cap, frame = cap.read()
        if not ret_cap:
            print("AVISO: falha ao ler frame, tentando novamente...")
            continue

        frame_count += 1

        if frame_count % DETECCAO_A_CADA == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ultimo_ret, ultimos_corners, ultimo_metodo = tentar_detectar(gray)

        display = frame.copy()
        h, w = display.shape[:2]

        if ultimo_ret and ultimos_corners is not None:
            cv2.drawChessboardCorners(display, CHECKERBOARD, ultimos_corners, True)
            status_texto = f"DETECTADO ({ultimo_metodo})"
            status_cor   = (0, 255, 0)
            instrucao    = "OK - pressione ESPACO para salvar"
        else:
            status_texto = "Tabuleiro NAO detectado - reposicione"
            status_cor   = (0, 0, 255)
            instrucao    = "Aguardando deteccao..."

        # HUD semitransparente
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

        cv2.putText(display, status_texto,            (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_cor,       2)
        cv2.putText(display, f"Salvas: {count}/{TARGET_IMAGES}", (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 0),   2)
        cv2.putText(display, instrucao,               (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 2)
        cv2.putText(display, "[ESC]=Calibrar  [S]=Forcar salvar", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)

        cv2.imshow("Captura RTSP - Intelbras", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # ESPAÇO
            if ultimo_ret:
                filename = os.path.join(IMAGES_DIR, f"img_{count:02d}.png")
                cv2.imwrite(filename, frame)
                print(f"[SALVO] {filename}  (metodo: {ultimo_metodo})")
                count += 1

                # Flash de confirmação
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
                cv2.addWeighted(flash, 0.4, frame, 0.6, 0, flash)
                cv2.imshow("Captura RTSP - Intelbras", flash)
                cv2.waitKey(100)

                if count >= TARGET_IMAGES:
                    print(f"\nMeta de {TARGET_IMAGES} imagens atingida!")
                    break
            else:
                print("[AVISO] Tabuleiro nao detectado - nao salvo.")
                print("        Reposicione ate aparecer VERDE na tela.")

        elif key == ord('s'):  # forçar salvar
            filename = os.path.join(IMAGES_DIR, f"img_{count:02d}.png")
            cv2.imwrite(filename, frame)
            print(f"[SALVO FORCADO] {filename}")
            count += 1
            if count >= TARGET_IMAGES:
                break

        elif key == 27:  # ESC
            print(f"\nCaptura encerrada. Total salvo: {count} imagens.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nImagens salvas em: {IMAGES_DIR}")
    return count


# ============================================================
# ETAPA 2: CALIBRAÇÃO FISHEYE
# ============================================================
def detectar_cantos_fisheye(images, mostrar=True):
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    image_names_validas = []
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
            objpoints.append(objp)
            imgpoints.append(corners2.reshape(1, -1, 2))
            image_names_validas.append(fname)

            if mostrar:
                img_drawn = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
                cv2.imshow('Cantos detectados', img_drawn)
                cv2.waitKey(300)
            print(f"[OK] Cantos encontrados em: {fname}")
        else:
            if mostrar:
                cv2.imshow('Cantos detectados', img)
                cv2.waitKey(300)
            print(f"[FALHOU] Cantos NAO encontrados em: {fname}")

    if mostrar:
        cv2.destroyAllWindows()

    return objpoints, imgpoints, image_names_validas, img_shape


def calibrar_fisheye():
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))

    if len(images) == 0:
        print(f"ERRO: nenhuma imagem encontrada em {IMAGES_DIR}")
        return None

    print(f"\n=== CALIBRAÇÃO FISHEYE ===")
    print(f"Processando {len(images)} imagens...\n")

    objpoints, imgpoints, image_names_validas, img_shape = detectar_cantos_fisheye(images)

    if len(objpoints) < 5:
        print("ERRO: imagens insuficientes (menos de 5 com cantos detectados). Abortando.")
        return None

    print(f"\n{len(objpoints)} de {len(images)} imagens usadas na calibração.")

    N = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_shape, K, D, rvecs, tvecs,
            FISHEYE_FLAGS, fisheye_criteria
        )
    except cv2.error as e:
        print(f"\nERRO durante a calibração fisheye:\n{e}")
        print("\nDica: remova as imagens com tabuleiro muito inclinado/cortado e tente novamente.")
        return None

    print(f"\nErro RMS de reprojeção (fisheye): {rms:.4f}")
    print("\nMatriz da câmera (K):\n", K)
    print("\nCoeficientes de distorção fisheye (D):\n", D)

    print("\n=== ERRO DE REPROJEÇÃO POR IMAGEM ===")
    erros = []
    for i in range(N):
        imgpoints_proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        erro = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj[0])
        erros.append((image_names_validas[i], erro))

    erros.sort(key=lambda x: x[1], reverse=True)
    for nome, err in erros:
        print(f"{nome}: {err:.4f} px")

    out_npz = os.path.join(_script_dir, "camera_calib_fisheye.npz")
    np.savez(out_npz, K=K, D=D, img_shape=img_shape)
    print(f"\nParâmetros salvos em: {out_npz}")

    return K, D, img_shape


# ============================================================
# ETAPA 3: COMPARAÇÃO ANTES/DEPOIS
# ============================================================
def comparar_antes_depois_fisheye(K, D, img_shape):
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if not images:
        print("Nenhuma imagem disponível para comparação.")
        return

    print("\n=== COMPARAÇÃO ANTES/DEPOIS (FISHEYE) ===")
    for i, fname in enumerate(images):
        print(f"  [{i}] {fname}")

    while True:
        escolha = input(f"\nDigite o número da imagem (0-{len(images)-1}): ").strip()
        if escolha.isdigit() and 0 <= int(escolha) < len(images):
            idx = int(escolha)
            break
        print("Entrada inválida.")

    img = cv2.imread(images[idx])
    h, w = img.shape[:2]

    balance_input = input("Valor de balance entre 0.0 e 1.0 (padrão 0.5): ").strip()
    try:
        balance = max(0.0, min(1.0, float(balance_input)))
    except ValueError:
        balance = 0.5

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, img_shape, np.eye(3), balance=balance
    )
    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, img_shape, cv2.CV_16SC2
    )
    dst = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)

    combined = np.hstack((img, dst))
    max_width = 1600
    scale = min(1.0, max_width / combined.shape[1])
    if scale < 1.0:
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    largura_metade = int(img.shape[1] * scale)
    cv2.putText(combined, "ANTES",  (10, 30),                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined, "DEPOIS", (largura_metade + 10, 30),   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Antes (esquerda) vs Depois (direita)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    out_jpg = os.path.join(_script_dir, "comparacao_fisheye.jpg")
    out_npz = os.path.join(_script_dir, "camera_calib_fisheye.npz")
    cv2.imwrite(out_jpg, combined)
    np.savez(out_npz, K=K, D=D, img_shape=img_shape, new_K=new_K, balance=balance)
    print(f"Comparação salva em: {out_jpg}  (balance={balance})")


# ============================================================
# FLUXO PRINCIPAL
# ============================================================
def main():
    print("=" * 60)
    print(" CALIBRAÇÃO FISHEYE - CÂMERA INTELBRAS (TETO DO LABORATÓRIO)")
    print("=" * 60)

    total = capturar_imagens()
    if total < 5:
        print("Imagens insuficientes para calibrar. Encerrando.")
        return

    resultado = calibrar_fisheye()
    if resultado is None:
        print("Encerrando: calibração não pôde ser concluída.")
        return

    K, D, img_shape = resultado
    comparar_antes_depois_fisheye(K, D, img_shape)

    print("\nProcesso concluído!")
    _stderr_log.close()


if __name__ == "__main__":
    main()