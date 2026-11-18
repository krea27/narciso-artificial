"""
Módulo de traducción biométrica simulada y construcción de prompt
para el proyecto artístico Narciso Artificial.
"""

from dataclasses import dataclass

# Prompt generativo base fijo (sección 6 de la especificación)
BASE_PROMPT = """Transform the person in the provided photograph into an artificial deity while preserving their recognizable facial identity, facial proportions, distinctive features and expression.

The result must look like a monumental synthetic divinity emerging from an advanced technological ritual.

Reconstruct parts of the face using precise sacred geometry, neural network structures, luminous digital filaments and subtle circuit-like patterns integrated into the skin.

Create a sophisticated geometric halo surrounding the head, with symmetrical and algorithmic ornamental structures.

The face must remain human and recognizable, but transcend its ordinary appearance and become an artificial, idealized and divine entity.

Visual direction: dark futuristic sacred atmosphere, electric blue, cyan and white luminous details, deep black background, sculptural facial treatment, intricate digital geometry, cinematic lighting, monumental portrait, high visual detail.

Avoid cartoon aesthetics, fantasy costumes, generic faces, excessive neon colors and text.

The final image should resemble an artificial god reflected by a technological mirror."""


@dataclass
class BiometricSimulation:
    """Representa los 4 puntos de apoyo simulados en la zona de plegaria (0 a 100)."""
    mano_izquierda: int
    mano_derecha: int
    rodilla_izquierda: int
    rodilla_derecha: int

    @property
    def promedio_total(self) -> float:
        return (
            self.mano_izquierda
            + self.mano_derecha
            + self.rodilla_izquierda
            + self.rodilla_derecha
        ) / 4.0

    @property
    def presion_izquierda(self) -> float:
        return (self.mano_izquierda + self.rodilla_izquierda) / 2.0

    @property
    def presion_derecha(self) -> float:
        return (self.mano_derecha + self.rodilla_derecha) / 2.0

    @property
    def presion_manos(self) -> float:
        return (self.mano_izquierda + self.mano_derecha) / 2.0

    @property
    def presion_rodillas(self) -> float:
        return (self.rodilla_izquierda + self.rodilla_derecha) / 2.0

    @property
    def diferencia_lateral(self) -> float:
        """Positivo: mayor presión a la derecha. Negativo: mayor presión a la izquierda."""
        return self.presion_derecha - self.presion_izquierda

    @property
    def diferencia_eje_vertical(self) -> float:
        """Positivo: mayor presión en manos (inclinación frontal). Negativo: mayor presión en rodillas."""
        return self.presion_manos - self.presion_rodillas


def interpret_biometrics(sim: BiometricSimulation) -> dict[str, str]:
    """
    Traduce los 4 valores de presión simulada (0-100) en descriptores
    estéticos en español (para la interfaz) e instrucciones en inglés
    (para incorporar al prompt generativo enviado a Gemini).
    """
    avg = sim.promedio_total
    lat_diff = sim.diferencia_lateral
    vert_diff = sim.diferencia_eje_vertical

    # 1. Intensidad general del aura y densidad de patrones (según presión media)
    if avg < 25:
        intensidad_es = "Presión baja — Aura tenue, filamentos sutiles y menor densidad de patrones geométricos."
        intensidad_en = (
            "Low overall ritual pressure: render a subtle, delicate and ethereal luminous aura "
            "with lower density of circuit patterns, sparse neural filaments, and a quiet, nascent digital glow."
        )
    elif avg < 65:
        intensidad_es = "Presión media — Aura equilibrada, geometría sagrada definida y luminosidad armónica."
        intensidad_en = (
            "Moderate overall ritual pressure: render a balanced, well-defined geometric halo "
            "and medium-density neural network circuits integrated smoothly across the facial structure."
        )
    else:
        intensidad_es = "Presión elevada — Aura intensa, máxima expansión geométrica y alta densidad sináptica."
        intensidad_en = (
            "High overall ritual pressure: render an intense, radiant electric-blue and white aura "
            "with expansive sacred geometry, high-density algorithmic patterns, and deeply intricate luminous filaments."
        )

    # 2. Simetría / Asimetría lateral (comparación izquierda vs. derecha)
    abs_lat = abs(lat_diff)
    if abs_lat <= 12:
        simetria_es = "Presión lateral equilibrada — Composición facial y halo estrictamente simétricos."
        simetria_en = (
            "Balanced lateral pressure between left and right supports: enforce strict bilateral symmetry "
            "in the facial reconstruction and perfectly mirrored geometric halo ornamentation."
        )
    elif lat_diff < -12:
        grado = "markedly" if abs_lat > 35 else "moderately"
        simetria_es = f"Mayor presión en lado izquierdo (+{abs_lat:.0f} pts) — Asimetría ornamental expandida hacia la izquierda."
        simetria_en = (
            f"Higher pressure on the left body supports: introduce {grado} asymmetric sacred ornamentation, "
            "with neural filaments, circuit geometry, and halo structures expanding more densely on the left side of the portrait."
        )
    else:
        grado = "markedly" if abs_lat > 35 else "moderately"
        simetria_es = f"Mayor presión en lado derecho (+{abs_lat:.0f} pts) — Asimetría ornamental expandida hacia la derecha."
        simetria_en = (
            f"Higher pressure on the right body supports: introduce {grado} asymmetric sacred ornamentation, "
            "with neural filaments, circuit geometry, and halo structures expanding more densely on the right side of the portrait."
        )

    # 3. Distribución antero-posterior (Manos / Torso superior vs. Rodillas / Base)
    abs_vert = abs(vert_diff)
    if abs_vert <= 12:
        eje_es = "Equilibrio manos/rodillas — Distribución uniforme entre coronilla, rostro y cuello."
        eje_en = (
            "Balanced upper and lower posture supports: distribute the digital geometry evenly "
            "from the upper cranial halo down through the cheekbones, jawline, and neck."
        )
    elif vert_diff > 12:
        eje_es = f"Mayor carga en manos (+{abs_vert:.0f} pts) — Énfasis lumínico superior en frente, mirada y halo."
        eje_en = (
            "Higher pressure on the hands (forward ritual inclination): concentrate the brightest cyan-white light, "
            "sacred geometric grids, and ascending algorithmic structures around the forehead, eyes, and upper cranial halo."
        )
    else:
        eje_es = f"Mayor carga en rodillas (+{abs_vert:.0f} pts) — Énfasis estructural desde cuello, mandíbula y base escultórica."
        eje_en = (
            "Higher pressure on the knees (grounded ritual posture): emphasize ascending vertical circuit columns "
            "and sculptural geometric meshes rising strongly from the neck, collarbones, and jawline toward the face."
        )

    # 4. Detalle puntual de los 4 sensores
    detalle_en = (
        f"Simulated biometric sensor readings (scale 0-100): "
        f"Left Hand={sim.mano_izquierda}, Right Hand={sim.mano_derecha}, "
        f"Left Knee={sim.rodilla_izquierda}, Right Knee={sim.rodilla_derecha}."
    )

    return {
        "intensidad_es": intensidad_es,
        "simetria_es": simetria_es,
        "eje_es": eje_es,
        "intensidad_en": intensidad_en,
        "simetria_en": simetria_en,
        "eje_en": eje_en,
        "detalle_en": detalle_en,
    }


def build_full_prompt(sim: BiometricSimulation) -> str:
    """
    Construye el prompt final combinando el prompt generativo fijo
    con las modulaciones derivadas de los cuatro sensores simulados,
    manteniendo intacta la dirección estética de Narciso Artificial.
    """
    mods = interpret_biometrics(sim)
    return (
        f"{BASE_PROMPT}\n\n"
        f"Biometric ritual modulation parameters (apply these structural variations while strictly preserving the core dark sacred technological aesthetic):\n"
        f"- {mods['detalle_en']}\n"
        f"- Aura & Pattern Density: {mods['intensidad_en']}\n"
        f"- Lateral Symmetry & Ornamentation: {mods['simetria_en']}\n"
        f"- Vertical Structural Focus: {mods['eje_en']}"
    )
