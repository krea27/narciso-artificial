"""
Motor Generativo Local (OpenCV + MediaPipe + Pillow) para Narciso Artificial.
Diseñado para replicar fielmente la estética de la imagen conceptual del proyecto (image6.png / image8.png):
1. Embellecimiento e idealización escultórica de la piel: elimina por completo enrojecimiento, manchas
   de frecuencia media, hiperpigmentación y poros mediante filtrado guiado de baja frecuencia sobre el canal
   infrarrojo/rojo y reconstrucción de iluminación 3D (esculpido de pómulos, puente nasal, labios y mirada).
2. Integración orgánica 3D de filamentos sinápticos y micro-retícula curvada sobre el volumen real del rostro
   (nunca líneas planas que tapen los ojos ni la expresión).
3. Encuadre de retrato de deidad con margen oscuro alrededor del rostro, halo y matriz digital de cuadrados
   luminosos en el fondo negro y contorno craneal.
"""

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from prompt_builder import BiometricSimulation


def _detect_face_box_and_landmarks(rgb_arr: np.ndarray) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Detecta el rostro y sus puntos clave. Si MediaPipe no está disponible,
    estima la elipse anatómica del rostro a partir del mapa de piel central.
    """
    h, w, _ = rgb_arr.shape

    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.35,
            ) as face_mesh:
                results = face_mesh.process(rgb_arr)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm], dtype=np.int32)
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)
                    return pts, (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    except Exception:
        pass

    ycrcb = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = ((cr >= 133) & (cr <= 180) & (cb >= 75) & (cb <= 132)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > (w * h * 0.06):
            fx, fy, fw, fh = cv2.boundingRect(largest)
            fh = min(fh, int(fw * 1.35))
            return None, (int(fx), int(fy), int(fw), int(fh))

    fw = int(w * 0.58)
    fh = int(fw * 1.25)
    fx = (w - fw) // 2
    fy = max(int(h * 0.12), (h - fh) // 2)
    return None, (fx, fy, fw, fh)


def _prepare_framed_portrait(source_image: Image.Image, target_w: int = 768, target_h: int = 1024) -> np.ndarray:
    """
    Prepara la fotografía en formato vertical 3:4 (768x1024).
    Si la foto original es un primerísimo plano muy cerrado (donde la cara ocupa casi el 100% del ancho),
    agrega automáticamente un margen negro ritual alrededor para que la cabeza tenga aire,
    halo sagrado y columnas de matriz digital como en image6.png e image8.png.
    """
    src_rgb = np.array(source_image.convert("RGB"))
    sh, sw, _ = src_rgb.shape

    _, (fx, fy, fw, fh) = _detect_face_box_and_landmarks(src_rgb)
    face_width_ratio = fw / float(sw)

    # Si el rostro ocupa más del 78% del ancho de la foto, alejar muy levemente (0.72) para dar margen al halo
    if face_width_ratio > 0.78:
        scale = 0.72 / face_width_ratio
        new_sw = int(sw * scale)
        new_sh = int(sh * scale)
        resized_src = cv2.resize(src_rgb, (new_sw, new_sh), interpolation=cv2.INTER_LANCZOS4)

        canvas_w = max(sw, int(new_sw / 0.75))
        canvas_h = int(canvas_w * (target_h / target_w))
        if canvas_h < new_sh:
            canvas_h = int(new_sh * 1.15)
            canvas_w = int(canvas_h * (target_w / target_h))

        framed = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        x_off = (canvas_w - new_sw) // 2
        y_off = (canvas_h - new_sh) // 2

        yy, xx = np.mgrid[0:new_sh, 0:new_sw].astype(np.float32)
        edge_fade_x = np.minimum(xx / (new_sw * 0.10 + 1), (new_sw - xx) / (new_sw * 0.10 + 1))
        edge_fade_y = np.minimum(yy / (new_sh * 0.10 + 1), (new_sh - yy) / (new_sh * 0.08 + 1))
        edge_fade = np.clip(np.minimum(edge_fade_x, edge_fade_y), 0.0, 1.0) ** 1.3
        edge_fade = edge_fade[:, :, np.newaxis]

        framed[y_off : y_off + new_sh, x_off : x_off + new_sw] = (resized_src.astype(np.float32) * edge_fade).astype(np.uint8)
        src_rgb = framed
        sh, sw, _ = src_rgb.shape

    target_aspect = target_w / target_h
    src_aspect = sw / sh
    if src_aspect > target_aspect:
        new_w = int(sh * target_aspect)
        x0 = (sw - new_w) // 2
        cropped = src_rgb[:, x0 : x0 + new_w]
    else:
        new_h = int(sw / target_aspect)
        y0 = (sh - new_h) // 2
        cropped = src_rgb[y0 : y0 + new_h, :]

    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _create_curved_3d_face_grid(
    h: int,
    w: int,
    cx: float,
    cy: float,
    fw: float,
    fh: float,
    depth_map: np.ndarray,
    spacing: int = 34,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una retícula sagrada muy fina que SE CURVA tridimensionalmente siguiendo
    el relieve real del rostro (depth_map), igual que el wireframe de la frente, pómulos y cuello
    en image8.png.
    """
    depth_f32 = cv2.GaussianBlur(depth_map.astype(np.float32), (0, 0), sigmaX=6.0)
    grad_x = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=5) * 16.0
    grad_y = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=5) * 16.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    rel_x = (xx - cx) / (fw * 0.55 + 1e-5)
    rel_y = (yy - cy) / (fh * 0.65 + 1e-5)
    r2 = np.clip(rel_x**2 + rel_y**2, 0.0, 2.5)
    spherical_warp = 1.0 + 0.15 * r2

    warped_x = (xx - cx) * spherical_warp + cx + grad_x
    warped_y = (yy - cy) * spherical_warp + cy + grad_y

    phase_x = np.abs(((warped_x % spacing) / spacing) - 0.5) * 2.0
    phase_y = np.abs(((warped_y % spacing) / spacing) - 0.5) * 2.0

    line_x = np.clip((0.075 - phase_x) / 0.075, 0.0, 1.0)
    line_y = np.clip((0.075 - phase_y) / 0.075, 0.0, 1.0)

    grid_lines = np.maximum(line_x, line_y)
    grid_nodes = np.clip(line_x * line_y * 2.6, 0.0, 1.0)

    return grid_lines, grid_nodes


def _generate_synaptic_branches(
    h: int,
    w: int,
    cx: int,
    cy: int,
    fw: int,
    fh: int,
    sim: BiometricSimulation,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Traza filamentos orgánicos ramificados tipo red neuronal / fibras ópticas sinápticas
    que recorren el contorno de los pómulos, mandíbula, sienes, frente y cuello,
    respondiendo con precisión a los 4 sensores simulados.
    """
    sharp_layer = np.zeros((h, w), dtype=np.uint8)
    node_layer = np.zeros((h, w), dtype=np.uint8)

    avg_p = sim.promedio_total / 100.0
    rng = random.Random(
        sim.mano_izquierda * 1000
        + sim.mano_derecha * 100
        + sim.rodilla_izquierda * 10
        + sim.rodilla_derecha
    )

    # 4 Cuadrantes controlados directamente por los 4 sensores individuales:
    # - Mano Izquierda -> Sien, frente y pómulo superior izquierdo
    # - Mano Derecha   -> Sien, frente y pómulo superior derecho
    # - Rodilla Izquierda -> Mandíbula, mejilla baja y cuello izquierdo
    # - Rodilla Derecha   -> Mandíbula, mejilla baja y cuello derecho
    quadrant_anchors = [
        # (sensor_val, lista de anclas en ese cuadrante)
        (
            sim.mano_izquierda / 100.0,
            [
                (int(cx - fw * 0.18), int(cy - fh * 0.36)),
                (int(cx - fw * 0.36), int(cy - fh * 0.24)),
                (int(cx - fw * 0.42), int(cy - fh * 0.04)),
                (int(cx - fw * 0.30), int(cy + fh * 0.08)),
            ],
        ),
        (
            sim.mano_derecha / 100.0,
            [
                (int(cx + fw * 0.18), int(cy - fh * 0.36)),
                (int(cx + fw * 0.36), int(cy - fh * 0.24)),
                (int(cx + fw * 0.42), int(cy - fh * 0.04)),
                (int(cx + fw * 0.30), int(cy + fh * 0.08)),
            ],
        ),
        (
            sim.rodilla_izquierda / 100.0,
            [
                (int(cx - fw * 0.38), int(cy + fh * 0.18)),
                (int(cx - fw * 0.32), int(cy + fh * 0.34)),
                (int(cx - fw * 0.16), int(cy + fh * 0.46)),
                (int(cx - fw * 0.24), int(cy + fh * 0.68)),
            ],
        ),
        (
            sim.rodilla_derecha / 100.0,
            [
                (int(cx + fw * 0.38), int(cy + fh * 0.18)),
                (int(cx + fw * 0.32), int(cy + fh * 0.34)),
                (int(cx + fw * 0.16), int(cy + fh * 0.46)),
                (int(cx + fw * 0.24), int(cy + fh * 0.68)),
            ],
        ),
    ]

    for sensor_norm, anchors in quadrant_anchors:
        # Cantidad de filamentos en este cuadrante proporcional a su sensor (0 a 100)
        n_q_branches = int(4 + 20 * sensor_norm)
        for _ in range(n_q_branches):
            ax, ay = rng.choice(anchors)
            curr_x = float(ax + rng.randint(-20, 20))
            curr_y = float(ay + rng.randint(-20, 20))

            angle = math.atan2(curr_y - cy, curr_x - cx) + rng.uniform(-0.75, 0.75)
            steps = rng.randint(8, int(14 + 14 * sensor_norm))
            step_len = rng.uniform(7.0, 14.0)

            for s in range(steps):
                angle += rng.uniform(-0.38, 0.38)
                nx = curr_x + math.cos(angle) * step_len
                ny = curr_y + math.sin(angle) * step_len

                ix1, iy1 = int(np.clip(curr_x, 0, w - 1)), int(np.clip(curr_y, 0, h - 1))
                ix2, iy2 = int(np.clip(nx, 0, w - 1)), int(np.clip(ny, 0, h - 1))

                val = int(255 * (0.45 + 0.55 * sensor_norm) * (1.0 - (s / (steps + 3)) * 0.5))
                cv2.line(sharp_layer, (ix1, iy1), (ix2, iy2), val, 1, cv2.LINE_AA)

                if s % 3 == 0 and rng.random() < (0.35 + 0.5 * sensor_norm):
                    cv2.circle(node_layer, (ix2, iy2), rng.choice([2, 3]), 255, -1, cv2.LINE_AA)
                    if rng.random() < 0.35:
                        sq = rng.choice([4, 7, 10])
                        cv2.rectangle(
                            sharp_layer,
                            (ix2 - sq, iy2 - sq),
                            (ix2 + sq, iy2 + sq),
                            int(val * 0.85),
                            1,
                            cv2.LINE_AA,
                        )
                curr_x, curr_y = nx, ny

    return sharp_layer.astype(np.float32) / 255.0, node_layer.astype(np.float32) / 255.0


def render_sacred_deity_portrait(
    source_image: Image.Image,
    sim: BiometricSimulation,
) -> Image.Image:
    """
    Transforma la fotografía en una Deidad Artificial embellecida, escultórica e integrada,
    fiel a las imágenes conceptuales de Narciso Artificial (image6.png e image8.png).
    """
    target_w, target_h = 768, 1024
    img_rgb = _prepare_framed_portrait(source_image, target_w, target_h)
    h, w, _ = img_rgb.shape

    _, (fx, fy, fw, fh) = _detect_face_box_and_landmarks(img_rgb)
    cx = int(fx + fw * 0.50)
    cy = int(fy + fh * 0.46)

    avg_p = sim.promedio_total / 100.0
    lat_diff = sim.diferencia_lateral / 100.0
    vert_diff = sim.diferencia_eje_vertical / 100.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # =========================================================================
    # 1. EMBELLECIMIENTO PROFUNDO DE LA PIEL ("PORCELANA / MÁRMOL DIVINO")
    # =========================================================================
    # En fotos clínicas o con manchas/rojeces, las manchas son variaciones de frecuencia media.
    # Solución:
    # A) Usar el canal Rojo puro (donde las manchas rojizas desaparecen casi por completo).
    # B) Extraer una base de iluminación ultra-suave (low-pass) de la dermis para que las mejillas
    #    y la frente tengan una superficie de porcelana lisa e inmaculada.
    # C) Preservar con nitidez cristalina los ojos (iris, pestañas), cejas, fosas nasales y labios.
    r_f = img_rgb[:, :, 0].astype(np.float32)
    g_f = img_rgb[:, :, 1].astype(np.float32)
    b_f = img_rgb[:, :, 2].astype(np.float32)

    # Canal base limpio de manchas vasculares (88% Rojo + 12% Verde)
    clean_channel = np.clip(r_f * 0.88 + g_f * 0.12, 0, 255).astype(np.uint8)

    # Máscara de rasgos clave (ojos, cejas, labios, contorno nasal) donde QUEREMOS máximo detalle y contraste
    eye_left_x, eye_right_x = cx - fw * 0.19, cx + fw * 0.19
    eye_y = cy - fh * 0.07
    nose_tip_y = cy + fh * 0.09
    mouth_y = cy + fh * 0.25

    d_eye_l = np.exp(-(((xx - eye_left_x) / (fw * 0.16)) ** 2 + ((yy - eye_y) / (fh * 0.11)) ** 2))
    d_eye_r = np.exp(-(((xx - eye_right_x) / (fw * 0.16)) ** 2 + ((yy - eye_y) / (fh * 0.11)) ** 2))
    d_mouth = np.exp(-(((xx - cx) / (fw * 0.20)) ** 2 + ((yy - mouth_y) / (fh * 0.10)) ** 2))
    d_nose = np.exp(-(((xx - cx) / (fw * 0.11)) ** 2 + ((yy - nose_tip_y) / (fh * 0.10)) ** 2))

    feature_sharp_mask = np.clip(d_eye_l + d_eye_r + d_mouth * 0.9 + d_nose * 0.65, 0.0, 1.0)

    # Piel de porcelana ultra-lisa en mejillas y frente (elimina manchas medianas y poros)
    bilateral_1 = cv2.bilateralFilter(clean_channel, d=21, sigmaColor=75, sigmaSpace=75).astype(np.float32)
    porcelain_dome = cv2.GaussianBlur(bilateral_1, (0, 0), sigmaX=14.0)
    # Mezclar 72% domo de porcelana + 28% bilateral en las mejillas, y 100% nitidez en ojos/cejas/labios
    skin_smooth = bilateral_1 * 0.32 + porcelain_dome * 0.68
    sharp_features = cv2.bilateralFilter(clean_channel, d=5, sigmaColor=20, sigmaSpace=20).astype(np.float32)

    # Realzar pestañas, iris y línea de los labios
    fine_detail = sharp_features - cv2.GaussianBlur(sharp_features, (0, 0), sigmaX=1.8)
    enhanced_features = np.clip(sharp_features + fine_detail * 1.4, 0, 255)

    idealized_face = skin_smooth * (1.0 - feature_sharp_mask) + enhanced_features * feature_sharp_mask

    # =========================================================================
    # 2. ESCULPIDO 3D ("CONTOURING & HIGHLIGHTING" DE DEIDAD COMO EN IMAGE8)
    # =========================================================================
    norm_face = idealized_face / 255.0
    p_low, p_high = np.percentile(norm_face, (4, 98))
    norm_face = np.clip((norm_face - p_low) / (p_high - p_low + 1e-5), 0.0, 1.0)

    # Curva gamma cinematográfica (evita sobreexponer el centro de la nariz y mejillas, manteniendo volumen 3D)
    norm_face = np.power(norm_face, 1.18)

    # Luces de contorno sutiles (Highlight) en pómulos, puente nasal, mentón y arco de Cupido
    hl_nose_bridge = np.exp(-(((xx - cx) / (fw * 0.032)) ** 2 + ((yy - (cy - fh * 0.02)) / (fh * 0.13)) ** 2))
    hl_cheek_l = np.exp(-(((xx - (cx - fw * 0.22)) / (fw * 0.12)) ** 2 + ((yy - (cy + fh * 0.04)) / (fh * 0.09)) ** 2))
    hl_cheek_r = np.exp(-(((xx - (cx + fw * 0.22)) / (fw * 0.12)) ** 2 + ((yy - (cy + fh * 0.04)) / (fh * 0.09)) ** 2))
    hl_chin = np.exp(-(((xx - cx) / (fw * 0.07)) ** 2 + ((yy - (cy + fh * 0.36)) / (fh * 0.05)) ** 2))
    hl_cupid = np.exp(-(((xx - cx) / (fw * 0.06)) ** 2 + ((yy - (mouth_y - fh * 0.03)) / (fh * 0.03)) ** 2))

    sculpt_boost = (
        hl_nose_bridge * 0.12
        + (hl_cheek_l + hl_cheek_r) * 0.10
        + hl_chin * 0.10
        + hl_cupid * 0.12
    )
    norm_face = np.clip(norm_face * 0.88 + sculpt_boost * norm_face, 0.0, 1.0)

    # Brillo en el iris de ambos ojos (mirada sobrenatural cian-blanca)
    iris_l = np.exp(-(((xx - eye_left_x) / (fw * 0.032)) ** 2 + ((yy - eye_y) / (fh * 0.022)) ** 2))
    iris_r = np.exp(-(((xx - eye_right_x) / (fw * 0.032)) ** 2 + ((yy - eye_y) / (fh * 0.022)) ** 2))
    iris_glow = (iris_l + iris_r) * 0.28

    sculpt_curve = np.power(norm_face, 1.32)
    specular = np.power(np.clip((norm_face - 0.65) / 0.35, 0.0, 1.0), 2.1) * 0.65 + iris_glow

    # =========================================================================
    # 3. PALETA CIAN / AZUL ELÉCTRICO / NACARADO (IDÉNTICA A IMAGE6 & IMAGE8)
    # =========================================================================
    deity_r = np.clip(np.power(sculpt_curve, 1.95) * 145.0 + specular * 75.0, 0, 255)
    deity_g = np.clip(np.power(sculpt_curve, 1.35) * 195.0 + specular * 58.0, 0, 255)
    deity_b = np.clip(np.power(sculpt_curve, 0.90) * 242.0 + 14.0 + specular * 32.0, 0, 255)
    deity_skin_rgb = np.stack([deity_r, deity_g, deity_b], axis=-1)

    # Sutil matiz satinado rosado/malva en los labios con relieve oscuro en la comisura (como en image8.png)
    lip_tint = np.stack([d_mouth * 48.0, d_mouth * 6.0, d_mouth * 30.0], axis=-1)
    deity_skin_rgb = np.clip(deity_skin_rgb * (1.0 - d_mouth[:, :, np.newaxis] * 0.15) + lip_tint, 0, 255)

    # =========================================================================
    # 4. AISLAMIENTO EN EL ESPEJO NEGRO Y MÁSCARA ESPACIAL DE LOS 4 SENSORES
    # =========================================================================
    ellipse_x = (xx - cx) / (fw * 0.76 + 1e-5)
    ellipse_y = (yy - (cy + fh * 0.10)) / (fh * 1.02 + 1e-5)
    dist_center = np.sqrt(ellipse_x**2 + ellipse_y**2)

    subject_mask = np.clip(1.36 - dist_center, 0.0, 1.0)
    subject_mask = np.power(subject_mask, 1.65).astype(np.float32)
    subject_mask = cv2.GaussianBlur(subject_mask, (0, 0), sigmaX=18.0)[:, :, np.newaxis]

    portrait_canvas = deity_skin_rgb * subject_mask

    # Mapa de intensidad espacial de los 4 sensores (Mano Izq, Mano Der, Rodilla Izq, Rodilla Der)
    # interpolado suavemente en los 4 cuadrantes de la imagen
    norm_x = np.clip((xx - (cx - fw * 0.7)) / (fw * 1.4 + 1e-5), 0.0, 1.0)  # 0 izq -> 1 der
    norm_y = np.clip((yy - (cy - fh * 0.7)) / (fh * 1.4 + 1e-5), 0.0, 1.0)  # 0 arriba (manos) -> 1 abajo (rodillas)

    p_tl = sim.mano_izquierda / 100.0
    p_tr = sim.mano_derecha / 100.0
    p_bl = sim.rodilla_izquierda / 100.0
    p_br = sim.rodilla_derecha / 100.0

    top_interp = p_tl * (1.0 - norm_x) + p_tr * norm_x
    bot_interp = p_bl * (1.0 - norm_x) + p_br * norm_x
    quadrant_field = top_interp * (1.0 - norm_y) + bot_interp * norm_y
    sensor_spatial_mod = np.clip(0.30 + 1.15 * quadrant_field, 0.25, 1.65)

    # Protección para que los ojos y el centro de los labios permanezcan limpios
    protect_eyes_mouth = np.clip(1.0 - (d_eye_l * 0.88 + d_eye_r * 0.88 + d_mouth * 0.75), 0.10, 1.0).astype(np.float32)

    # =========================================================================
    # 5. RETÍCULA 3D CURVADA + FILAMENTOS SINÁPTICOS INTEGRADOS EN LA PIEL
    # =========================================================================
    grid_spacing = int(38 - 12 * avg_p)
    grid_lines, grid_nodes = _create_curved_3d_face_grid(
        h, w, cx, cy, fw, fh, norm_face, spacing=grid_spacing
    )

    skin_presence = np.clip(1.18 - dist_center * 0.78, 0.0, 1.0) * protect_eyes_mouth
    integrated_grid = (
        grid_lines * (0.20 + 0.35 * avg_p) * (0.30 + 0.70 * sculpt_curve)
        + grid_nodes * (0.40 + 0.55 * avg_p)
    ) * skin_presence * sensor_spatial_mod

    syn_branches, syn_nodes = _generate_synaptic_branches(h, w, cx, cy, fw, fh, sim)
    syn_branches_glow = cv2.GaussianBlur(syn_branches, (0, 0), sigmaX=2.4)

    # =========================================================================
    # 6. HALO GEOMÉTRICO SAGRADO Y MATRIZ DE CUADRADOS DIGITALES EN EL CONTORNO
    # =========================================================================
    bg_matrix_sharp = np.zeros((h, w), dtype=np.float32)
    bg_matrix_glow = np.zeros((h, w), dtype=np.float32)

    rng = random.Random(
        sim.mano_izquierda * 1000
        + sim.mano_derecha * 100
        + sim.rodilla_izquierda * 10
        + sim.rodilla_derecha + 99
    )

    num_cols = int(16 + 22 * avg_p)
    for _ in range(num_cols):
        col_x = rng.randint(14, w - 14)
        dist_from_face_x = abs(col_x - cx) / (fw * 0.48 + 1e-5)
        if dist_from_face_x < 0.60 and rng.random() < 0.78:
            continue

        cv2.line(bg_matrix_glow, (col_x, 0), (col_x, h), 0.16, 1, cv2.LINE_AA)
        n_tiles = int(10 + 18 * avg_p)
        for _ in range(n_tiles):
            tile_y = rng.randint(14, h - 14)
            sz = rng.choice([6, 10, 14, 18])
            val = rng.uniform(0.25, 0.85)
            cv2.rectangle(
                bg_matrix_sharp,
                (col_x - sz // 2, tile_y - sz // 2),
                (col_x + sz // 2, tile_y + sz // 2),
                val,
                1,
                cv2.LINE_AA,
            )
            if sz >= 10 and rng.random() < 0.55:
                cv2.circle(bg_matrix_glow, (col_x, tile_y), 2, val * 1.15, -1, cv2.LINE_AA)

    # Anillos de geometría sagrada que enmarcan la cabeza por detrás
    halo_cy = int(cy - fh * (0.10 + 0.10 * vert_diff))
    halo_cx = int(cx + lat_diff * fw * 0.16)
    for r_scale in (0.58, 0.74, 0.90):
        rad = int(max(fw, fh) * r_scale * (0.85 + 0.20 * avg_p))
        cv2.circle(bg_matrix_sharp, (halo_cx, halo_cy), rad, 0.45 + 0.35 * avg_p, 1, cv2.LINE_AA)
        cv2.circle(bg_matrix_glow, (halo_cx, halo_cy), rad, 0.35 + 0.35 * avg_p, 2, cv2.LINE_AA)

    halo_behind_mask = np.clip((dist_center - 0.44) / 0.48, 0.06, 1.0) * sensor_spatial_mod
    bg_matrix_sharp *= halo_behind_mask
    bg_matrix_glow *= halo_behind_mask

    # =========================================================================
    # 7. EJE VERTICAL DE LUZ SUTIL Y CORONA EN LA FRENTE (SIN QUEMAR LA PIEL)
    # =========================================================================
    # Haz fino y elegante que recorre el eje de simetría, igual que en image8.png
    center_filament = np.exp(-((xx - cx) ** 2) / (2.0 * (fw * 0.008) ** 2))
    center_aura = np.exp(-((xx - cx) ** 2) / (2.0 * (fw * 0.055) ** 2))
    crown_star = np.exp(-(((xx - cx) / (fw * 0.06)) ** 2 + ((yy - (cy - fh * 0.36)) / (fh * 0.06)) ** 2))
    neck_star = np.exp(-(((xx - cx) / (fw * 0.08)) ** 2 + ((yy - (cy + fh * 0.65)) / (fh * 0.09)) ** 2))

    divine_axis_light = (
        center_filament * (0.32 + 0.35 * avg_p) * protect_eyes_mouth
        + center_aura * (0.14 + 0.18 * avg_p) * norm_face * protect_eyes_mouth
        + crown_star * (0.35 + 0.40 * ((p_tl + p_tr) * 0.5))
        + neck_star * (0.30 + 0.40 * ((p_bl + p_br) * 0.5))
    )

    # =========================================================================
    # 8. FUSIÓN ÓPTICA FINAL (SCREEN + SPECULAR GLOW)
    # =========================================================================
    energy_fine = (
        integrated_grid * 0.68
        + (syn_branches * 0.80 + syn_nodes * 1.15) * protect_eyes_mouth * sensor_spatial_mod
        + bg_matrix_sharp * 0.75
    ).astype(np.float32)

    energy_bloom = (
        cv2.GaussianBlur(energy_fine, (0, 0), sigmaX=3.0) * 0.80
        + cv2.GaussianBlur((syn_branches_glow + bg_matrix_glow).astype(np.float32), (0, 0), sigmaX=11.0) * 0.70
        + divine_axis_light
    )

    total_energy = np.clip(energy_fine * 0.75 + energy_bloom * 0.80, 0.0, 1.55)

    energy_rgb = np.zeros((h, w, 3), dtype=np.float32)
    energy_rgb[:, :, 0] = np.clip(np.power(np.clip(total_energy * 0.72, 0, 1.4), 1.85) * 230.0, 0, 255)
    energy_rgb[:, :, 1] = np.clip(np.power(np.clip(total_energy * 0.92, 0, 1.4), 1.15) * 235.0, 0, 255)
    energy_rgb[:, :, 2] = np.clip(np.power(np.clip(total_energy * 1.15, 0, 1.4), 0.75) * 255.0, 0, 255)

    base_norm = np.clip(portrait_canvas / 255.0, 0.0, 1.0)
    energy_norm = np.clip(energy_rgb / 255.0, 0.0, 1.0)

    screen_blended = 1.0 - (1.0 - base_norm) * (1.0 - energy_norm * 0.88)
    high_peaks = np.power(energy_norm, 2.3) * 0.38
    final_norm = np.clip(screen_blended + high_peaks, 0.0, 1.0)

    final_arr = (final_norm * 255.0).astype(np.uint8)
    return Image.fromarray(final_arr, mode="RGB")
