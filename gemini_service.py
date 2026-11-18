"""
Cliente de conexión con Google Gemini Image Generation (SDK oficial google-genai)
para Narciso Artificial, con respaldo híbrido gratuito (Gemini 3.5 Flash-Lite Vision + Motor Generativo Local)
cuando la cuenta de Google AI Studio está en el Plan Gratuito (Free Tier, limit: 0 en modelos pagos de imagen).
"""

import base64
from io import BytesIO
import os
from typing import Optional, Tuple

import importlib
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

import local_deity_engine
importlib.reload(local_deity_engine)
from local_deity_engine import render_sacred_deity_portrait
from prompt_builder import BiometricSimulation

# Cargar variables de entorno desde .env
load_dotenv(override=True)

DEFAULT_MODEL = "gemini-nano-banana-2.1"
LEGACY_REQUESTED_MODEL = "gemini-3.1-flash-image"
FREE_VISION_MODEL = "gemini-3.5-flash-lite"


class GeminiConfigError(Exception):
    """Error de configuración local (ej. falta GEMINI_API_KEY)."""


class GeminiGenerationError(Exception):
    """Error durante la generación de la imagen con la API de Gemini."""


def get_api_key() -> Optional[str]:
    """Obtiene y valida la clave GEMINI_API_KEY desde el entorno (.env) o desde Streamlit Secrets (Cloud)."""
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                key = str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass
    placeholders = {
        "",
        "tu_api_key_de_google_ai_studio_aqui",
        "your_api_key_here",
        "tu_api_key_aqui",
        "AIza...",
    }
    if not key or key in placeholders:
        return None
    return key


def save_api_key_to_env(api_key: str) -> None:
    """Guarda la clave GEMINI_API_KEY directamente en el archivo .env local."""
    clean_key = api_key.strip()
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY="):
                    lines.append(f"GEMINI_API_KEY={clean_key}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"\nGEMINI_API_KEY={clean_key}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.environ["GEMINI_API_KEY"] = clean_key
    load_dotenv(override=True)


def get_configured_model() -> str:
    """Devuelve el modelo configurado en .env o el modelo activo por defecto."""
    load_dotenv(override=True)
    return os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _pil_to_base64_and_bytes(image: Image.Image) -> Tuple[str, bytes, str]:
    """Convierte una imagen PIL a RGB PNG en bytes y base64."""
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (0, 0, 0))
        bg.paste(image, mask=image.split()[3])
        image = bg

    max_dim = 1536
    if max(image.size) > max_dim:
        image = image.copy()
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    raw_bytes = buffer.getvalue()
    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return b64_str, raw_bytes, "image/png"


def _extract_image_from_interaction(interaction) -> Optional[Image.Image]:
    """Extrae el objeto PIL.Image desde una respuesta de Interactions API."""
    output_img = getattr(interaction, "output_image", None)
    if output_img is not None:
        data = getattr(output_img, "data", None)
        if data:
            raw = base64.b64decode(data) if isinstance(data, str) else data
            return Image.open(BytesIO(raw)).convert("RGB")

    steps = getattr(interaction, "steps", None) or []
    for step in reversed(steps):
        if getattr(step, "type", "") == "model_output":
            contents = getattr(step, "content", None) or []
            for block in reversed(contents):
                b_type = getattr(block, "type", "")
                if b_type in ("image", "output_image"):
                    if hasattr(block, "as_image"):
                        return block.as_image()
                    data = getattr(block, "data", None)
                    if data:
                        raw = base64.b64decode(data) if isinstance(data, str) else data
                        return Image.open(BytesIO(raw)).convert("RGB")
    return None


def _generate_gemini_ritual_inscription(
    client: genai.Client,
    source_image: Image.Image,
    sim: BiometricSimulation,
) -> str:
    """
    Utiliza el modelo multimodal gratuito de la cuenta del usuario (`gemini-3.5-flash-lite`)
    para analizar la fotografía del rostro junto con los 4 sensores simulados y devolver
    la lectura ritual / diagnóstico divino del espejo de Narciso Artificial.
    """
    try:
        thumb = source_image.copy()
        thumb.thumbnail((512, 512), Image.Resampling.LANCZOS)
        ritual_prompt = (
            "Sos el Espejo Algorítmico de la instalación artística 'Narciso Artificial'. "
            "Observá el rostro de la persona en la fotografía y los valores simulados de sus 4 apoyos de plegaria "
            f"(Mano Izquierda={sim.mano_izquierda}%, Mano Derecha={sim.mano_derecha}%, "
            f"Rodilla Izquierda={sim.rodilla_izquierda}%, Rodilla Derecha={sim.rodilla_derecha}%). "
            "Escribí en español un breve texto poético, solemne y tecnológico (máximo 3 oraciones) "
            "describiendo cómo sus facciones humanas se han transfigurado en esta deidad sintética "
            "de geometría sagrada azul eléctrico y redes neuronales."
        )
        res = client.models.generate_content(
            model=FREE_VISION_MODEL,
            contents=[thumb, ritual_prompt],
        )
        return (res.text or "").strip()
    except Exception:
        return ""


def generate_deity_image(
    source_image: Image.Image,
    prompt: str,
    preferred_model: Optional[str] = None,
    sim: Optional[BiometricSimulation] = None,
) -> Tuple[Image.Image, str, str]:
    """
    1. Intenta generar la imagen con Google Gemini Image Generation (`gemini-nano-banana-2.1` / `gemini-3.1-flash-image`).
    2. Si la cuenta de Google AI Studio del usuario es Gratuita (Free Tier, `limit: 0` en modelos pagos de imagen),
       activa automáticamente el modo híbrido 100% gratuito:
       - Analiza la fotografía y los sensores con Gemini Multimodal Gratuito (`gemini-3.5-flash-lite`).
       - Sintetiza el retrato de la Deidad Artificial con el Motor Generativo Sagrado Local (OpenCV + MediaPipe + Geometría Sagrada).

    Retorna:
        (imagen_generada_pil, etiqueta_del_motor_utilizado, inscripcion_ritual_de_gemini)
    """
    api_key = get_api_key()
    if sim is None:
        sim = BiometricSimulation(65, 65, 65, 65)

    # Si se ejecuta en una computadora o nube donde todavía no configuraron GEMINI_API_KEY,
    # sintetiza igualmente la Deidad Artificial con el motor sagrado local para que siempre funcione.
    if not api_key:
        deity_img = render_sacred_deity_portrait(source_image, sim)
        return (
            deity_img,
            "Motor Sagrado Local (OpenCV + Geometría Sagrada 3D)",
            "",
        )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=25000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    # Si el usuario eligió explícitamente el motor híbrido gratuito
    if preferred_model == "free-hybrid":
        inscription = _generate_gemini_ritual_inscription(client, source_image, sim)
        deity_img = render_sacred_deity_portrait(source_image, sim)
        return (
            deity_img,
            f"Modo Gratuito Activo ({FREE_VISION_MODEL} + Motor Sagrado OpenCV/MediaPipe)",
            inscription,
        )

    b64_image, _, mime_type = _pil_to_base64_and_bytes(source_image)

    primary_model = preferred_model or get_configured_model()
    models_to_try = [primary_model]
    if primary_model != DEFAULT_MODEL:
        models_to_try.append(DEFAULT_MODEL)
    elif LEGACY_REQUESTED_MODEL not in models_to_try:
        models_to_try.append(LEGACY_REQUESTED_MODEL)

    last_error = None

    for model_name in models_to_try:
        try:
            interaction = client.interactions.create(
                model=model_name,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": b64_image,
                        "mime_type": mime_type,
                    },
                ],
                response_format={
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "aspect_ratio": "3:4",
                },
            )

            result_img = _extract_image_from_interaction(interaction)
            if result_img is not None:
                return result_img, model_name, ""

        except Exception as exc:
            last_error = exc
            err_msg = str(exc)
            err_lower = err_msg.lower()

            # Si la cuenta es Free Tier (limit: 0 en modelos pagos de imagen), usar el Modo Híbrido Gratuito automáticamente
            if "limit: 0" in err_lower or "free_tier" in err_lower or "429" in err_lower or "resource_exhausted" in err_lower:
                inscription = _generate_gemini_ritual_inscription(client, source_image, sim)
                deity_img = render_sacred_deity_portrait(source_image, sim)
                return (
                    deity_img,
                    f"Modo Gratuito ({FREE_VISION_MODEL} + Síntesis Geométrica Sagrada Local)",
                    inscription,
                )

            if "api_key_invalid" in err_lower or "401" in err_lower or "403" in err_lower or "permission" in err_lower:
                raise GeminiGenerationError(
                    f"Error de autenticación con la API Key en Google AI Studio (`{err_msg}`). "
                    "Verificá que `GEMINI_API_KEY` en tu archivo `.env` sea válida."
                ) from exc

            if ("not found" in err_lower or "404" in err_lower or "deprecated" in err_lower) and model_name != models_to_try[-1]:
                continue

    # Fallback final gratuito si hubo timeout del endpoint cloud de imagen
    inscription = _generate_gemini_ritual_inscription(client, source_image, sim)
    deity_img = render_sacred_deity_portrait(source_image, sim)
    return (
        deity_img,
        f"Modo Gratuito ({FREE_VISION_MODEL} + Síntesis Geométrica Sagrada Local)",
        inscription,
    )
