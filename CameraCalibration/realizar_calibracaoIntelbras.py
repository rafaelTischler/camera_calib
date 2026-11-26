#!/usr/bin/env python
"""
Detecção otimizada de tabuleiro de xadrez pequeno/distante (câmera fisheye no teto).
Usa upscaling, recorte de região de interesse (ROI) e múltiplas estratégias de
pré-processamento para maximizar a chance de detecção.
"""

import cv2
import numpy as np
import os
import glob

# Desabilita OpenCL para evitar os erros de cache (não afeta o resultado, só limpa o log)
cv2.ocl.setUseOpenCL(False)

# ============================================================
# CONFIGURAÇÕES
# ============================================================
CHECKERBOARD = (8, 6)        # cantos internos do seu tabuleiro A4 atual
SQUARE_SIZE = 20.0            # mm - ajuste para o tamanho real do quadrado impresso
IMAGES_DIR = "./images_intelbras"
OUTPUT_DEBUG_DIR = "./debug_deteccao"
ESCALAS_TESTE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

UPSCALE_FACTOR = 3.0          # fator de ampliação antes de detectar (teste 2.0 a 4.0)

# Se True, mostra/salva a imagem com grade para você ajustar a ROI manualmente
MODO_AJUSTE_ROI = True

# Região de interesse aproximada (x1, y1, x2, y2) em pixels da imagem ORIGINAL,
# cobrindo a área do chão onde o tabuleiro normalmente aparece.
# Ajuste esses valores olhando a grade de debug (ou deixe None para processar a imagem toda).
ROI = None  # exemplo depois de calibrar visualmente: (50, 100, 500, 400)


os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)


# ============================================================
# FUNÇÃO: desenhar grade de referência (para ajudar a definir ROI)
# ============================================================
def salvar_grade_debug(img, fname, passo=50):
    debug_img = img.copy()
    h, w = debug_img.shape[:2]

    for x in range(0, w, passo):
        cv2.line(debug_img, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.putText(debug_img, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    for y in range(0, h, passo):
        cv2.line(debug_img, (0, y), (w, y), (0, 255, 0), 1)
        cv2.putText(debug_img, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    out_path = os.path.join(OUTPUT_DEBUG_DIR, f"grade_{os.path.basename(fname)}")
    cv2.imwrite(out_path, debug_img)
    return out_path


# ============================================================
# FUNÇÃO: pré-processamentos a tentar, em ordem
# ============================================================
def gerar_variantes_preprocessamento(gray):
    variantes = []

    # 1. Original (apenas upscale, sem alteração extra)
    variantes.append(("original", gray))

    # 2. CLAHE (equalização adaptativa de contraste) - ajuda com sombra/iluminação irregular
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variantes.append(("clahe", clahe.apply(gray)))

    # 3. Nitidez (sharpening) via unsharp mask
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    variantes.append(("sharpen", sharpened))

    # 4. CLAHE + nitidez combinados
    clahe_img = clahe.apply(gray)
    blur2 = cv2.GaussianBlur(clahe_img, (0, 0), 3)
    sharp2 = cv2.addWeighted(clahe_img, 1.5, blur2, -0.5, 0)
    variantes.append(("clahe_sharpen", sharp2))

    # 5. Binarização adaptativa (útil quando contraste preto/branco está degradado)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    variantes.append(("adaptive_thresh", thresh))

    return variantes


# ============================================================
# FUNÇÃO: tentar detectar cantos com múltiplas estratégias
# ============================================================
def detectar_com_fallback(img, checkerboard, fname, salvar_debug=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    flags_sb = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    flags_classico = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for escala in ESCALAS_TESTE:
        if escala == 1.0:
            gray_scaled = gray
        else:
            new_w, new_h = int(w * escala), int(h * escala)
            gray_scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        variantes = gerar_variantes_preprocessamento(gray_scaled)

        for nome_metodo, img_variante in variantes:
            # Tenta findChessboardCornersSB primeiro
            try:
                ret, corners = cv2.findChessboardCornersSB(img_variante, checkerboard, flags_sb)
            except cv2.error:
                ret, corners = False, None

            if ret:
                corners_original = corners / escala
                if salvar_debug:
                    debug_img = img.copy()
                    cv2.drawChessboardCorners(debug_img, checkerboard, corners_original, ret)
                    out_path = os.path.join(OUTPUT_DEBUG_DIR,
                        f"OK_SB_{nome_metodo}_escala{escala}_{os.path.basename(fname)}")
                    cv2.imwrite(out_path, debug_img)
                return True, corners_original.astype(np.float32), f"SB_{nome_metodo}_escala{escala}"

            # Tenta método clássico
            try:
                ret2, corners2 = cv2.findChessboardCorners(img_variante, checkerboard, flags_classico)
            except cv2.error:
                ret2, corners2 = False, None

            if ret2:
                corners2 = cv2.cornerSubPix(img_variante, corners2, (11, 11), (-1, -1), criteria)
                corners_original = corners2 / escala
                if salvar_debug:
                    debug_img = img.copy()
                    cv2.drawChessboardCorners(debug_img, checkerboard, corners_original, ret2)
                    out_path = os.path.join(OUTPUT_DEBUG_DIR,
                        f"OK_classico_{nome_metodo}_escala{escala}_{os.path.basename(fname)}")
                    cv2.imwrite(out_path, debug_img)
                return True, corners_original.astype(np.float32), f"classico_{nome_metodo}_escala{escala}"

    return False, None, None


# ============================================================
# PROCESSAMENTO PRINCIPAL
# ============================================================
def processar_imagens():
    images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
    if not images:
        print(f"ERRO: nenhuma imagem encontrada em {IMAGES_DIR}")
        return [], [], None

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []
    image_names_validas = []
    img_shape = None

    print(f"Processando {len(images)} imagens (upscale={UPSCALE_FACTOR}x)...\n")

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[ERRO LEITURA] {fname}")
            continue

        if ROI is not None:
            x1, y1, x2, y2 = ROI
            img_proc = img[y1:y2, x1:x2]
        else:
            img_proc = img

        if img_shape is None:
            img_shape = (img.shape[1], img.shape[0])  # shape da imagem ORIGINAL, sem crop

        if MODO_AJUSTE_ROI:
            grade_path = salvar_grade_debug(img, fname)

        ret, corners, metodo = detectar_com_fallback(
            img_proc, CHECKERBOARD, fname, salvar_debug=True
        )

        if ret:
            # Se usou ROI (crop), precisa somar o deslocamento de volta às coordenadas originais
            if ROI is not None:
                corners[:, 0, 0] += ROI[0]
                corners[:, 0, 1] += ROI[1]

            objpoints.append(objp)
            imgpoints.append(corners.reshape(1, -1, 2))
            image_names_validas.append(fname)
            print(f"[OK - {metodo}] {fname}")
        else:
            print(f"[FALHOU] {fname}")

    return objpoints, imgpoints, image_names_validas, img_shape


# ============================================================
# CALIBRAÇÃO FISHEYE (reaproveitada)
# ============================================================
def calibrar_fisheye(objpoints, imgpoints, img_shape):
    if len(objpoints) < 5:
        print(f"\nERRO: apenas {len(objpoints)} imagens detectadas. "
              f"Mínimo de 5 necessário, idealmente 10+.")
        return None

    print(f"\n{len(objpoints)} imagens válidas para calibração.")

    N = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N)]

    fisheye_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints, imgpoints, img_shape, K, D, rvecs, tvecs, flags, fisheye_criteria
        )
    except cv2.error as e:
        print(f"\nERRO na calibração: {e}")
        return None

    print(f"\nErro RMS de reprojeção: {rms:.4f}")
    print("\nMatriz K:\n", K)
    print("\nCoeficientes D:\n", D)

    np.savez('camera_calib_fisheye.npz', K=K, D=D, img_shape=img_shape)
    print("\nParâmetros salvos em 'camera_calib_fisheye.npz'")

    return K, D


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print(" DETECÇÃO OTIMIZADA - TABULEIRO PEQUENO/DISTANTE")
    print("=" * 60)
    print(f"\nImagens de debug serão salvas em: {os.path.abspath(OUTPUT_DEBUG_DIR)}")
    print("Verifique lá os arquivos 'grade_*.png' para ajustar a ROI, se necessário.\n")

    objpoints, imgpoints, image_names_validas, img_shape = processar_imagens()

    print(f"\n{len(objpoints)} de {len(glob.glob(os.path.join(IMAGES_DIR, '*.png')))} "
          f"imagens tiveram o tabuleiro detectado com sucesso.")

    if len(objpoints) >= 5:
        calibrar_fisheye(objpoints, imgpoints, img_shape)
    else:
        print("\nImagens insuficientes para calibrar com esse conjunto.")
        print("Veja a pasta de debug para entender o que está acontecendo,")
        print("e considere usar um tabuleiro maior na próxima captura.")


if __name__ == "__main__":
    main()