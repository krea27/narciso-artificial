"""
NARCISO ARTIFICIAL — Prototipo Generativo
Instalación interactiva de transformación facial en deidad sintética.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import importlib
import cv2
from PIL import Image
import streamlit as st

import gemini_service
importlib.reload(gemini_service)
from gemini_service import (
    DEFAULT_MODEL,
    LEGACY_REQUESTED_MODEL,
    GeminiConfigError,
    GeminiGenerationError,
    generate_deity_image,
    get_api_key,
    save_api_key_to_env,
)
from prompt_builder import (
    BiometricSimulation,
    build_full_prompt,
    interpret_biometrics,
)

# Directorio local opcional para respaldar imágenes guardadas
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="NARCISO ARTIFICIAL — Prototipo Generativo",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_custom_css() -> None:
    """Inyecta el diseño minimalista, oscuro y ritual inspirado en la identidad visual del proyecto."""
    st.markdown(
        """
        <style>
        /* Fondo general negro profundo y atmósfera litúrgico-tecnológica */
        .stApp {
            background: radial-gradient(circle at 50% 0%, #0b162c 0%, #05070b 55%, #020305 100%);
            color: #e2f1ff;
        }

        /* Cabecera ritual */
        .narciso-header {
            text-align: center;
            padding: 1.4rem 1rem 1.2rem 1rem;
            border-bottom: 1px solid rgba(0, 229, 255, 0.28);
            margin-bottom: 1.5rem;
            background: linear-gradient(180deg, rgba(0, 229, 255, 0.06) 0%, rgba(5, 7, 11, 0) 100%);
        }
        .narciso-title {
            font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
            font-size: 2.3rem;
            font-weight: 300;
            letter-spacing: 0.45em;
            color: #ffffff;
            text-shadow: 0 0 18px rgba(0, 229, 255, 0.65), 0 0 36px rgba(0, 120, 255, 0.35);
            margin: 0;
            padding-left: 0.45em;
        }
        .narciso-subtitle {
            font-size: 0.82rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: #7ab8e6;
            margin-top: 0.55rem;
        }

        /* Paneles técnicos con borde luminoso azul cian */
        .ritual-panel {
            background: rgba(10, 16, 28, 0.78);
            border: 1px solid rgba(0, 229, 255, 0.25);
            box-shadow: 0 0 24px rgba(0, 140, 255, 0.08), inset 0 0 15px rgba(0, 229, 255, 0.04);
            border-radius: 6px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }
        .panel-heading {
            font-size: 0.85rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #00e5ff;
            border-bottom: 1px solid rgba(0, 229, 255, 0.2);
            padding-bottom: 0.45rem;
            margin-bottom: 0.85rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Etiqueta explícita de simulación */
        .sim-badge {
            display: inline-block;
            background: rgba(0, 229, 255, 0.12);
            color: #66f0ff;
            border: 1px solid rgba(0, 229, 255, 0.4);
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            padding: 0.2rem 0.6rem;
            border-radius: 3px;
            text-transform: uppercase;
        }

        /* Visualizador de los 4 puntos de apoyo (Zona de plegaria) */
        .altar-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.7rem;
            margin: 0.6rem 0 1rem 0;
        }
        .pad-box {
            background: rgba(4, 9, 18, 0.9);
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 4px;
            padding: 0.65rem;
            text-align: center;
            transition: all 0.2s ease;
        }
        .pad-label {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            color: #8bbCE0;
            text-transform: uppercase;
        }
        .pad-value {
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.7);
            margin-top: 0.15rem;
        }

        /* Marco del Espejo Vertical */
        .mirror-placeholder {
            border: 1px dashed rgba(0, 229, 255, 0.3);
            background: radial-gradient(circle at 50% 50%, rgba(0, 140, 255, 0.07) 0%, rgba(3, 5, 9, 0.95) 80%);
            border-radius: 6px;
            min-height: 460px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem;
            color: #6a96b8;
        }

        /* Botones principales */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 4px;
            letter-spacing: 0.14em;
            font-weight: 500;
            text-transform: uppercase;
            transition: all 0.25s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def capture_frame_opencv(camera_index: int = 0) -> Image.Image:
    """
    Captura un fotograma fijo utilizando OpenCV (cv2.VideoCapture)
    como alternativa directa cuando el navegador no utiliza el widget WebRTC.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo acceder a la webcam (índice {camera_index}). "
            "Verificá que la cámara esté conectada y que no esté bloqueada por otra aplicación."
        )

    try:
        # Descartar primeros cuadros para estabilizar exposición automática
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("La cámara se abrió pero no devolvió ningún fotograma válido.")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    finally:
        cap.release()


def pil_to_png_bytes(img: Image.Image) -> bytes:
    """Convierte un objeto PIL.Image a bytes en formato PNG."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def init_session_state() -> None:
    """Inicializa el estado persistente de la sesión de Streamlit."""
    if "source_image" not in st.session_state:
        st.session_state.source_image: Optional[Image.Image] = None
    if "source_origin" not in st.session_state:
        st.session_state.source_origin: str = ""
    if "generated_image" not in st.session_state:
        st.session_state.generated_image: Optional[Image.Image] = None
    if "used_model" not in st.session_state:
        st.session_state.used_model: str = ""
    if "ritual_inscription" not in st.session_state:
        st.session_state.ritual_inscription: str = ""
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt: str = ""
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active: bool = True
    if "saved_local_path" not in st.session_state:
        st.session_state.saved_local_path: str = ""


def main() -> None:
    inject_custom_css()
    init_session_state()

    # Cabecera principal
    st.markdown(
        """
        <div class="narciso-header">
            <h1 class="narciso-title">NARCISO ARTIFICIAL</h1>
            <div class="narciso-subtitle">Espejo Algorítmico &bull; Prototipo de Transformación Generativa</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Verificación del estado de la API Key en .env (opcional para la lectura de texto de Gemini; la imagen funciona siempre)
    api_key = get_api_key()
    if not api_key:
        with st.expander("🔑 Opcional: Configurar GEMINI_API_KEY (la generación visual funciona igual sin clave)", expanded=False):
            st.caption(
                "El **Motor de Esculpido y Geometría Sagrada 3D** funciona inmediatamente sin necesidad de clave. "
                "Si además querés activar la conexión con la nube de Google Gemini AI, podés pegar una clave gratuita de "
                "[Google AI Studio](https://aistudio.google.com/apikey) aquí:"
            )
            k_col1, k_col2 = st.columns([3, 1])
            with k_col1:
                input_key = st.text_input(
                    "Pegá tu API Key de Google AI Studio",
                    type="password",
                    placeholder="AIzaSy...",
                    label_visibility="collapsed",
                )
            with k_col2:
                if st.button("🔑 Guardar API Key en .env", use_container_width=True):
                    if input_key and len(input_key.strip()) > 15:
                        save_api_key_to_env(input_key)
                        st.success("¡API Key guardada en `.env` correctamente!")
                        st.rerun()
                    else:
                        st.error("Ingresá una clave válida de Google AI Studio.")

    col_left, col_center, col_right = st.columns([1.15, 1.05, 1.25], gap="large")

    # =========================================================================
    # COLUMNA 1: CAPTURA DE ROSTRO (WEBCAM O ARCHIVO)
    # =========================================================================
    with col_left:
        st.markdown(
            """
            <div class="ritual-panel">
                <div class="panel-heading">
                    <span>I. Captura del Rostro</span>
                    <span class="sim-badge">Cámara a ras del suelo</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_cam, tab_upload, tab_opencv = st.tabs(
            ["📷 Webcam en Vivo", "📁 Cargar Fotografía", "🔌 Captura Directa (OpenCV)"]
        )

        with tab_cam:
            st.caption(
                "Activá la webcam para visualizar tu rostro en tiempo real y presioná el botón de captura para fijar la fotografía."
            )
            cam_toggle = st.toggle(
                "Activar vista de cámara en tiempo real",
                value=st.session_state.webcam_active,
                key="toggle_webcam",
            )
            st.session_state.webcam_active = cam_toggle

            if st.session_state.webcam_active:
                cam_photo = st.camera_input(
                    "CAPTURAR ROSTRO",
                    key="streamlit_camera",
                    help="Visualizá el rostro en tiempo real y hacé clic en el botón inferior para tomar la foto fija.",
                )
                if cam_photo is not None:
                    try:
                        img = Image.open(cam_photo).convert("RGB")
                        st.session_state.source_image = img
                        st.session_state.source_origin = "Webcam (Tiempo Real)"
                    except Exception as exc:
                        st.error(f"Error al procesar la imagen capturada por la webcam: {exc}")
            else:
                st.info("La cámara está desactivada. Activala con el interruptor superior o cargá una imagen desde el disco.")

        with tab_upload:
            st.caption("Alternativa: seleccioná una fotografía desde tu disco local (JPG, JPEG, PNG, WEBP).")
            uploaded_file = st.file_uploader(
                "Seleccionar fotografía de rostro",
                type=["png", "jpg", "jpeg", "webp"],
                key="file_uploader",
            )
            if uploaded_file is not None:
                try:
                    img = Image.open(uploaded_file).convert("RGB")
                    st.session_state.source_image = img
                    st.session_state.source_origin = f"Disco ({uploaded_file.name})"
                except Exception as exc:
                    st.error(f"No se pudo abrir el archivo de imagen seleccionado: {exc}")

        with tab_opencv:
            st.caption(
                "Captura directa del dispositivo local mediante OpenCV (`cv2.VideoCapture`), útil si el navegador bloquea permisos WebRTC."
            )
            cam_idx = st.number_input("Índice de cámara USB", min_value=0, max_value=5, value=0, step=1)
            if st.button("CAPTURAR ROSTRO (OPENCV)", use_container_width=True):
                with st.spinner("Accediendo al dispositivo de cámara con OpenCV..."):
                    try:
                        cv_img = capture_frame_opencv(int(cam_idx))
                        st.session_state.source_image = cv_img
                        st.session_state.source_origin = f"Webcam OpenCV (Dispositivo #{cam_idx})"
                        st.success("Fotografía capturada correctamente mediante OpenCV.")
                    except Exception as exc:
                        st.error(str(exc))

        # Vista previa de la fotografía fija seleccionada
        if st.session_state.source_image is not None:
            st.markdown("##### Fotografía fija lista para el ritual")
            st.caption(f"Origen: **{st.session_state.source_origin}**")
            st.image(
                st.session_state.source_image,
                use_container_width=True,
            )
            if st.button("Descartar fotografía actual", use_container_width=True):
                st.session_state.source_image = None
                st.session_state.source_origin = ""
                st.rerun()

    # =========================================================================
    # COLUMNA 2: SIMULACIÓN DE SENSORES BIOMÉTRICOS (ZONA DE PLEGARIA)
    # =========================================================================
    with col_center:
        st.markdown(
            """
            <div class="ritual-panel">
                <div class="panel-heading">
                    <span>II. Zona de Plegaria (4 Apoyos)</span>
                    <span class="sim-badge">DATOS SIMULADOS</span>
                </div>
                <p style="font-size: 0.8rem; color: #9ab8d4; margin-bottom: 0.4rem;">
                    Simulación de las 4 celdas de carga (Load Cells + HX711). 
                    <strong>Indicador explícito:</strong> estos valores son controles simulados (0–100) y no mediciones biométricas físicas reales.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Controles independientes de los 4 puntos de apoyo (0 a 100)
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            mano_izq = st.slider(
                "🖐️ Mano Izquierda",
                min_value=0,
                max_value=100,
                value=65,
                step=1,
                help="Presión simulada en el apoyo frontal izquierdo (0 = sin apoyo, 100 = presión máxima).",
            )
            rodilla_izq = st.slider(
                "🧎 Rodilla Izquierda",
                min_value=0,
                max_value=100,
                value=65,
                step=1,
                help="Presión simulada en el apoyo posterior izquierdo (0 = sin apoyo, 100 = presión máxima).",
            )
        with s_col2:
            mano_der = st.slider(
                "🤚 Mano Derecha",
                min_value=0,
                max_value=100,
                value=65,
                step=1,
                help="Presión simulada en el apoyo frontal derecho (0 = sin apoyo, 100 = presión máxima).",
            )
            rodilla_der = st.slider(
                "🧎 Rodilla Derecha",
                min_value=0,
                max_value=100,
                value=65,
                step=1,
                help="Presión simulada en el apoyo posterior derecho (0 = sin apoyo, 100 = presión máxima).",
            )

        sim_data = BiometricSimulation(
            mano_izquierda=mano_izq,
            mano_derecha=mano_der,
            rodilla_izquierda=rodilla_izq,
            rodilla_derecha=rodilla_der,
        )
        interpretation = interpret_biometrics(sim_data)
        full_prompt = build_full_prompt(sim_data)

        # Representación visual de las 4 placas iluminadas en planta
        st.markdown(
            f"""
            <div class="altar-grid">
                <div class="pad-box" style="box-shadow: 0 0 {int(mano_izq * 0.25)}px rgba(0, 229, 255, {0.15 + mano_izq/200:.2f});">
                    <div class="pad-label">Mano Izquierda</div>
                    <div class="pad-value">{mano_izq}%</div>
                </div>
                <div class="pad-box" style="box-shadow: 0 0 {int(mano_der * 0.25)}px rgba(0, 229, 255, {0.15 + mano_der/200:.2f});">
                    <div class="pad-label">Mano Derecha</div>
                    <div class="pad-value">{mano_der}%</div>
                </div>
                <div class="pad-box" style="box-shadow: 0 0 {int(rodilla_izq * 0.25)}px rgba(0, 229, 255, {0.15 + rodilla_izq/200:.2f});">
                    <div class="pad-label">Rodilla Izquierda</div>
                    <div class="pad-value">{rodilla_izq}%</div>
                </div>
                <div class="pad-box" style="box-shadow: 0 0 {int(rodilla_der * 0.25)}px rgba(0, 229, 255, {0.15 + rodilla_der/200:.2f});">
                    <div class="pad-label">Rodilla Derecha</div>
                    <div class="pad-value">{rodilla_der}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Lectura estética traducida en tiempo real
        st.markdown("##### Modulación Estética Resultante")
        st.markdown(
            f"- **Intensidad de Aura (`{sim_data.promedio_total:.0f}/100`):** {interpretation['intensidad_es']}\n"
            f"- **Simetría Facial:** {interpretation['simetria_es']}\n"
            f"- **Eje Postural:** {interpretation['eje_es']}"
        )

        with st.expander("Inspeccionar Prompt Generativo completo enviado a Gemini"):
            st.code(full_prompt, language="text")
            model_choice = st.selectbox(
                "Motor de Transformación Generativa",
                options=[
                    f"{DEFAULT_MODEL} (Con respaldo automático 100% Gratuito si tu cuenta no tiene tarjeta)",
                    "free-hybrid (Modo 100% Gratuito Directo: Gemini Vision Free + Geometría Sagrada OpenCV/MediaPipe)",
                    f"{LEGACY_REQUESTED_MODEL} (Nano Banana 2)",
                ],
                index=0,
                help="Si tu cuenta de Google AI Studio es gratuita (sin tarjeta), utiliza automáticamente Gemini 3.5 Flash-Lite Vision + el motor sagrado local sin ningún costo.",
            )
            if model_choice.startswith("free-hybrid"):
                selected_model_id = "free-hybrid"
            elif model_choice.startswith(DEFAULT_MODEL):
                selected_model_id = DEFAULT_MODEL
            else:
                selected_model_id = LEGACY_REQUESTED_MODEL

        st.markdown("<br>", unsafe_allow_html=True)

        # Botón principal: GENERAR DEIDAD
        generate_clicked = st.button(
            "💠 GENERAR DEIDAD",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.source_image is None),
            help="Transforma la fotografía fija en una deidad artificial utilizando Gemini y los parámetros simulados.",
        )
        if st.session_state.source_image is None:
            st.caption("⚠️ Primero capturá un rostro con la webcam o cargá una fotografía en el Panel I.")

    # =========================================================================
    # COLUMNA 3: ESPEJO DE NARCISO (IMAGEN GENERADA Y GUARDADO PNG)
    # =========================================================================
    with col_right:
        st.markdown(
            """
            <div class="ritual-panel">
                <div class="panel-heading">
                    <span>III. Espejo Transformante</span>
                    <span class="sim-badge">Deidad Artificial</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if generate_clicked:
            if st.session_state.source_image is None:
                st.error("Debés capturar o cargar una fotografía antes de generar la deidad.")
            else:
                # Indicador de procesamiento ritual
                with st.status(
                    "Iniciando ritual algorítmico en el espejo de Narciso...",
                    expanded=True,
                ) as status_box:
                    st.write("🔹 Codificando rostro de entrada y vectores de presión simulada...")
                    st.write("🔹 Conectando con Google Gemini Image Generation (`google-genai`)...")
                    try:
                        deity_img, model_used, inscription = generate_deity_image(
                            source_image=st.session_state.source_image,
                            prompt=full_prompt,
                            preferred_model=selected_model_id,
                            sim=sim_data,
                        )
                        st.session_state.generated_image = deity_img
                        st.session_state.used_model = model_used
                        st.session_state.ritual_inscription = inscription
                        st.session_state.last_prompt = full_prompt
                        st.session_state.saved_local_path = ""
                        status_box.update(
                            label=f"Deidad artificial manifestada ({model_used})",
                            state="complete",
                            expanded=False,
                        )
                    except GeminiConfigError as cfg_err:
                        status_box.update(
                            label="Configuración incompleta: falta API Key",
                            state="error",
                            expanded=True,
                        )
                        st.error(str(cfg_err))
                    except GeminiGenerationError as gen_err:
                        status_box.update(
                            label="Error durante la generación con Gemini",
                            state="error",
                            expanded=True,
                        )
                        st.error(str(gen_err))
                    except Exception as unexpected_err:
                        status_box.update(
                            label="Error inesperado",
                            state="error",
                            expanded=True,
                        )
                        st.error(f"Ocurrió un error inesperado: {unexpected_err}")

        # Visualización del resultado o estado latente del espejo
        if st.session_state.generated_image is not None:
            st.image(
                st.session_state.generated_image,
                caption=f"Reflejo Divinizado • Motor: {st.session_state.used_model}",
                use_container_width=True,
            )
            if st.session_state.ritual_inscription:
                st.info(f"**Lectura del Espejo Algorítmico (Gemini AI):**\n\n_{st.session_state.ritual_inscription}_")

            png_bytes = pil_to_png_bytes(st.session_state.generated_image)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"narciso_artificial_{timestamp}.png"

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label="💾 GUARDAR IMAGEN (PNG)",
                    data=png_bytes,
                    file_name=filename,
                    mime="image/png",
                    type="primary",
                    use_container_width=True,
                )
            with btn_col2:
                if st.button("📁 Guardar copia en /outputs", use_container_width=True):
                    target_path = OUTPUT_DIR / filename
                    st.session_state.generated_image.save(target_path, format="PNG")
                    st.session_state.saved_local_path = str(target_path)

            if st.session_state.saved_local_path:
                st.success(f"Imagen guardada localmente en: `{st.session_state.saved_local_path}`")

        else:
            st.markdown(
                """
                <div class="mirror-placeholder">
                    <div style="font-size: 2.4rem; color: #00e5ff; opacity: 0.55; margin-bottom: 0.7rem;">◇</div>
                    <div style="letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.85rem; color: #9ac4e8;">
                        Espejo en Estado de Reposo
                    </div>
                    <p style="font-size: 0.8rem; max-width: 320px; margin-top: 0.6rem; color: #6989a8;">
                        Capturá un rostro, ajustá los cuatro apoyos simulados de la zona de plegaria y presioná 
                        <strong>GENERAR DEIDAD</strong> para contemplar tu reflejo sintético.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
