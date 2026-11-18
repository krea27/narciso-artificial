# NARCISO ARTIFICIAL — Prototipo Generativo

Prototipo funcional de la instalación interactiva **Narciso Artificial**, que captura el rostro del interactor mediante una webcam (o desde una fotografía local), incorpora la simulación de los cuatro sensores de presión de la zona de plegaria y transforma el retrato en una **deidad artificial sintética** utilizando Google Gemini Image Generation (`google-genai`).

---

## 1. Concepto y Dirección Estética

En la instalación **Narciso Artificial**, el espectador adopta una postura de ofrenda sobre cuatro superficies de apoyo (manos y rodillas) frente a un altar monolítico y una pantalla vertical que actúa como espejo algorítmico. El sistema captura su rostro y sus datos de presión corporal para devolverle una versión idealizada, geométrica y divinizada de sí mismo: simultáneamente dios y siervo de su propia imagen.

Este prototipo implementa el núcleo generativo de la obra:
1. **Captura de rostro** en tiempo real mediante webcam (con opción alternativa de carga desde disco o captura directa por OpenCV).
2. **Simulación de sensores biométricos (0 a 100)** para los cuatro apoyos del cuerpo:
   - **Mano izquierda**
   - **Mano derecha**
   - **Rodilla izquierda**
   - **Rodilla derecha**
3. **Traducción estética a prompt ritual**: los valores simulados modulan la simetría u ornamentación lateral, la intensidad del halo geométrico y la densidad de redes neuronales sobre la piel, manteniendo intacto el lenguaje visual oscuro, solemne y azul eléctrico/cian del proyecto.
4. **Transformación generativa con Google Gemini** conservando la identidad facial reconocible de la fotografía original.
5. **Visualización y descarga del resultado en formato PNG**.

---

## 2. Estructura del Proyecto

```text
narciso-artificial/
├── app.py                 # Aplicación e interfaz visual oscura en Streamlit
├── prompt_builder.py      # Prompt base fijo y traducción de sensores simulados (0-100)
├── gemini_service.py      # Cliente de Google GenAI SDK y manejo de errores de API
├── requirements.txt       # Dependencias de Python
├── .env.example           # Plantilla para configurar GEMINI_API_KEY
├── .gitignore             # Protección para no subir .env ni archivos temporales
├── .streamlit/
│   └── config.toml        # Tema visual oscuro/cian de la interfaz
└── outputs/               # Carpeta donde se pueden guardar copias locales en PNG
```

---

## 3. Requisitos Previos

- **Python 3.10 o superior** instalado en el sistema.
- Una **API Key de Google AI Studio**: podés generarla en [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey).
- Cámara web conectada (opcional, ya que también permite cargar fotografías desde el disco).

---

## 4. Instalación Paso a Paso

### Paso 1: Abrir una terminal en la carpeta del proyecto

```powershell
cd c:\Users\dedic\Desktop\am3\narciso-artificial
```

### Paso 2: (Opcional) Crear y activar un entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Paso 3: Instalar las dependencias

```powershell
pip install -r requirements.txt
```

---

## 5. Configuración de la API Key (`.env`)

La clave de API **nunca debe escribirse directamente en el código**.

1. Copiá el archivo `.env.example` con el nombre `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Abrí el archivo `.env` con cualquier editor de texto y reemplazá el valor de `GEMINI_API_KEY` por tu clave real de Google AI Studio:
   ```env
   GEMINI_API_KEY=AIzaSyTuClaveRealDeGoogleAIStudio...
   ```

> **Nota sobre el modelo de imagen (`Nano Banana 2 / 2.1`)**:
> El proyecto está configurado con el SDK oficial `google-genai` (`>= 3.2.0`) utilizando la API `client.interactions.create`. Soporta tanto el identificador `gemini-3.1-flash-image` (Nano Banana 2) como su versión activa en la API actual `gemini-nano-banana-2.1` (Nano Banana 2.1), realizando fallback automático para garantizar compatibilidad continua.

---

## 6. Ejecución de la Aplicación

Ejecutá el siguiente comando en la terminal:

```powershell
streamlit run app.py
```

Si el comando `python` o `streamlit` no está en tu `PATH` global de Windows, podés ejecutarlo directamente con el ejecutable de Python instalado en tu sistema:

```powershell
& "C:\Users\dedic\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m streamlit run app.py
```

Luego abrí en tu navegador la dirección local que indica la consola (habitualmente `http://localhost:8501`).

---

## 7. Uso de la Interfaz

1. **Panel I — Captura del Rostro**:
   - En la pestaña **Webcam en Vivo**, activá la cámara y presioná **CAPTURAR ROSTRO** para tomar una fotografía fija.
   - Si no tenés cámara disponible o preferís usar un retrato existente, abrí la pestaña **Cargar Fotografía** y seleccioná una imagen desde tu disco (`PNG`, `JPG`, `WEBP`), o utilizá la pestaña **Captura Directa (OpenCV)**.
2. **Panel II — Zona de Plegaria (4 Apoyos Simulados)**:
   - Ajustá de forma independiente los controles de **Mano Izquierda**, **Mano Derecha**, **Rodilla Izquierda** y **Rodilla Derecha** entre `0` y `100`.
   - Observá cómo cambian en tiempo real la luminancia de las 4 placas y la traducción estética (simetría bilateral vs. asimetría ornamental, densidad de circuitos y foco estructural).
3. **Panel III — Espejo Transformante**:
   - Presioná **GENERAR DEIDAD**.
   - Una vez procesada la transformación por Gemini, la deidad artificial se mostrará en el espejo vertical.
   - Hacé clic en **GUARDAR IMAGEN (PNG)** para descargar el archivo PNG o guardá una copia directa en la carpeta `outputs/`.
