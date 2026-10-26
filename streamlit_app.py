from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from model import ejecutar_modelo, preservar_color

st.set_page_config(
    page_title="Transformación del histograma — V3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Transformación del histograma — Versión 3")
st.caption(
    "Selección del intervalo mínimo por porcentaje, aproximación del histograma, "
    "transformación parametrizada y reasignación ordenada de intensidades."
)

with st.sidebar:
    st.header("Experimento")
    uploaded_file = st.file_uploader(
        "Carga una imagen",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help="La imagen se analiza en escala de grises. Las vistas comparativas conservan el color cuando la imagen original es a color.",
    )
    st.divider()
    porcentaje = st.slider(
        "Porcentaje mínimo de píxeles dentro del intervalo",
        min_value=50.0,
        max_value=100.0,
        value=98.5,
        step=0.1,
        format="%.1f%%",
        help="La aplicación busca el intervalo de menor amplitud que contiene al menos este porcentaje del total.",
    )
    metodo = st.selectbox(
        "Método de aproximación",
        options=["PCHIP", "LINEAL", "FOURIER"],
        index=0,
    )
    if metodo in ("PCHIP", "LINEAL"):
        num_puntos = st.slider(
            "Número de puntos de control",
            min_value=2,
            max_value=256,
            value=250,
            step=1,
            help="Se limita automáticamente al número de intensidades del intervalo seleccionado.",
        )
        num_armonicos = 12
    else:
        num_armonicos = st.slider(
            "Número de armónicos de Fourier",
            min_value=0,
            max_value=128,
            value=12,
            step=1,
        )
        num_puntos = 250
    st.caption("Los parámetros A, B y C se calculan siguiendo el procedimiento de la versión 3 del notebook. D permanece en 0.")

if uploaded_file is None:
    st.info("Carga una imagen desde el panel lateral para comenzar el experimento.")
    st.markdown(
        "### Qué podrás observar\n"
        "1. Imagen original e histograma con su aproximación.\n"
        "2. Función transformada e histograma objetivo.\n"
        "3. Comparación de histogramas normalizados y CDF de los cuatro modelos.\n"
        "4. Comparación visual de las imágenes y tabla de métricas."
    )
    st.stop()

try:
    uploaded_file.seek(0)
    original_pil = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    imagen_rgb_original = np.asarray(original_pil, dtype=np.uint8)
except Exception as exc:
    st.error(f"No fue posible abrir la imagen. Detalle: {exc}")
    st.stop()

# Reproduce el criterio de la versión de Colab: considerar color si algún píxel
# tiene canales RGB distintos; el cálculo del modelo se efectúa en escala de grises.
es_color = bool(np.any(np.max(imagen_rgb_original, axis=2) != np.min(imagen_rgb_original, axis=2)))
imagen_gris = np.asarray(Image.fromarray(imagen_rgb_original).convert("L"), dtype=np.uint8)

try:
    with st.spinner("Calculando el histograma, la aproximación y las imágenes de comparación..."):
        datos = ejecutar_modelo(
            imagen_gris,
            porcentaje=porcentaje,
            metodo=metodo,
            num_puntos=num_puntos,
            num_armonicos=num_armonicos,
        )
except Exception as exc:
    st.error(f"No fue posible procesar esta imagen con la configuración actual. Detalle: {exc}")
    st.stop()

r_min = datos["r_min"]
r_max = datos["r_max"]
histograma = datos["histograma"]
hist_aprox = datos["aproximacion"]["histograma_aproximado"]
niveles = np.arange(256)
transformacion = datos["transformacion"]
funcion_transformada = transformacion["funcion_transformada"]
frecuencias_objetivo = transformacion["frecuencias_objetivo"]

# Mantener la apariencia original en imágenes a color: se reemplaza solamente
# la luminosidad LAB por la imagen procesada en grises.
resultados_visuales = {
    "Original": imagen_rgb_original if es_color else imagen_gris,
    "Modelo V3": preservar_color(imagen_rgb_original, datos["imagen_propuesta"], es_color),
    "Ecualización clásica (HE)": preservar_color(imagen_rgb_original, datos["imagen_he"], es_color),
    "CLAHE": preservar_color(imagen_rgb_original, datos["imagen_clahe"], es_color),
}

# Construir bytes PNG para el botón de descarga.
bytes_png = BytesIO()
Image.fromarray(resultados_visuales["Modelo V3"]).save(bytes_png, format="PNG")
png_modelo = bytes_png.getvalue()

# Tabs: la vista principal está enfocada en resultados; los datos de diagnóstico
# y variables internas se mantienen en la pestaña Desarrollador.
tab_principal, tab_desarrollador = st.tabs(["Resultados", "Desarrollador"])

with tab_principal:
    st.header("1. Imagen original")
    st.image(
        imagen_rgb_original if es_color else imagen_gris,
        caption="Imagen original (se calcula el modelo en escala de grises).",
        use_container_width=True,
    )

    st.header("2. Histograma original y aproximación")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.bar(niveles, histograma, width=1.0, alpha=0.24, color="steelblue", label="Histograma original")
    ax.axvspan(r_min - 0.5, r_max + 0.5, color="orange", alpha=0.09, label="Intervalo aproximado")
    ax.plot(
        datos["niveles_absolutos_rango"],
        hist_aprox,
        color="darkorange",
        linewidth=2.2,
        label=f"Aproximación {metodo}",
    )
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    ax.set_xlim(0, 255)
    ax.set_title("Histograma original y función aproximada")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.header("3. Transformación de la función e histograma objetivo")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(datos["niveles_absolutos_rango"], hist_aprox, label="Aproximación original", linewidth=2)
        ax.plot(niveles, funcion_transformada, label="Función transformada G(r)", linewidth=2, color="darkorange")
        ax.set_title("Función antes y después de transformar")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia propuesta")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(niveles, histograma, label="Histograma original", linewidth=1.2, alpha=0.75)
        ax.plot(niveles, frecuencias_objetivo, label="Frecuencias objetivo", linewidth=1.8, color="darkorange")
        ax.set_title("Histograma original frente al objetivo")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Número de píxeles propuesto")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.header("4. Comparación de histogramas y CDF")
    col_h, col_cdf = st.columns(2)
    with col_h:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for nombre, img in datos["resultados_grises"].items():
            h = np.bincount(img.ravel(), minlength=256)
            ax.plot(niveles, h / img.size, label=nombre, linewidth=1.5)
        ax.set_title("Histogramas normalizados")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia relativa")
        ax.set_xlim(0, 255)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col_cdf:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for nombre, cdf in datos["cdfs_modelos"].items():
            ax.plot(niveles, cdf, label=nombre, linewidth=1.7)
        ax.set_title("Funciones de distribución acumulada (CDF)")
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Proporción acumulada")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.header("5. Comparación visual de resultados")
    st.caption("Cuando la imagen original es a color, las transformaciones se visualizan sustituyendo la luminosidad y conservando sus componentes cromáticos.")
    cols = st.columns(4)
    for col, (nombre, img) in zip(cols, resultados_visuales.items()):
        with col:
            st.image(img, caption=nombre, use_container_width=True)

    st.header("6. Comparación de métricas")
    st.caption("Las métricas se calculan sobre las imágenes en escala de grises usadas por el modelo, para comparar el efecto sobre las intensidades.")
    st.dataframe(datos["tabla_metricas"].round(4), use_container_width=True)

    st.download_button(
        label="Descargar imagen generada por el modelo (PNG)",
        data=png_modelo,
        file_name="imagen_transformada_histograma_V3.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
    )

with tab_desarrollador:
    st.header("Información técnica del experimento")
    st.caption("Esta pestaña reúne los datos internos y diagnósticos del modelo; no son necesarios para interpretar visualmente los resultados principales.")

    st.subheader("Selección del intervalo")
    d1, d2, d3 = st.columns(3)
    d1.metric("Intervalo seleccionado", f"[{r_min}, {r_max}]")
    d2.metric("Ancho del intervalo", f"{datos['rango_dinamico']} niveles")
    d3.metric("Cobertura real", f"{datos['porcentaje_real']:.5f}%")
    d4, d5, d6 = st.columns(3)
    d4.metric("Píxeles requeridos", f"{datos['pixeles_requeridos']:,}")
    d5.metric("Píxeles dentro", f"{datos['pixeles_dentro']:,}")
    d6.metric("Píxeles fuera del intervalo", f"{datos['pixeles_fuera']:,}")
    st.write(
        f"El porcentaje solicitado es **{datos['porcentaje_solicitado']:.3f}%**. "
        "El intervalo es el de menor amplitud que cumple la cobertura, con los desempates de la versión 3. "
        "Los píxeles fuera del intervalo no se eliminan; participan en la reasignación posterior."
    )

    st.subheader("Parámetros de la transformación")
    tabla_parametros = pd.DataFrame(
        {
            "Parámetro": ["A", "B", "C final", "D", "C por área de aproximación", "C por total real de píxeles"],
            "Valor": [
                transformacion["A"],
                transformacion["B"],
                transformacion["C"],
                transformacion["D"],
                transformacion["C_area_aproximacion"],
                transformacion["C_area_total_pixeles"],
            ],
            "Interpretación": [
                "Escala horizontal: 255 / (r_max - r_min), con fallback si el rango es puntual.",
                "Desplazamiento horizontal: -r_min.",
                "Escala vertical ajustada para que las frecuencias objetivo redondeadas alcancen el total de píxeles.",
                "Desplazamiento vertical; fijado en 0.",
                "Iguala el área discreta de la aproximación original al área de la función transformada.",
                "Iguala el área discreta sin redondear con el número real de píxeles.",
            ],
        }
    )
    st.dataframe(tabla_parametros, hide_index=True, use_container_width=True)

    st.subheader("Diagnóstico de frecuencias y reasignación")
    transformacion = datos["transformacion"]
    dx1, dx2, dx3 = st.columns(3)
    dx1.metric("Píxeles originales", f"{imagen_gris.size:,}")
    dx2.metric("Suma de frecuencias objetivo", f"{transformacion['suma_objetivo']:,}")
    dx3.metric("Exceso objetivo − total", f"{transformacion['diferencia_objetivo_total']:,}")
    dx4, dx5, dx6 = st.columns(3)
    dx4.metric("Sobrantes asignados a 255", f"{datos['sobrantes_255']:,}")
    dx5.metric("Área original aprox. (discreta)", f"{transformacion['area_aproximacion_original']:.3f}")
    dx6.metric("Área transformada sin C", f"{transformacion['area_transformada_sin_C']:.3f}")
    st.write(f"Error absoluto medio de la aproximación en el intervalo: **{datos['aproximacion']['error_absoluto_medio']:.6f}**.")
    st.write(f"Método de aproximación: **{metodo}**. Puntos efectivos: **{datos['aproximacion']['puntos_efectivos']}**. Armónicos efectivos: **{datos['aproximacion']['armonicos_efectivos']}**.")

    with st.expander("Tabla de frecuencias por intensidad"):
        tabla_frecuencias = pd.DataFrame({
            "intensidad": niveles,
            "histograma_original": datos["histograma"].astype(int),
            "intervalo_minimo": ((niveles >= r_min) & (niveles <= r_max)).astype(int),
            "aproximacion_histograma": np.pad(
                datos["aproximacion"]["histograma_aproximado"],
                (r_min, 255 - r_max),
                mode="constant",
                constant_values=0,
            ),
            "funcion_transformada_bruta": transformacion["funcion_transformada_bruta"],
            "funcion_transformada_no_negativa": transformacion["funcion_transformada"],
            "frecuencia_objetivo": transformacion["frecuencias_objetivo"],
            "histograma_logrado": datos["histograma_logrado"],
        })
        st.dataframe(tabla_frecuencias, use_container_width=True, height=380)
        st.download_button(
            "Descargar frecuencias por intensidad (CSV)",
            data=tabla_frecuencias.to_csv(index=False).encode("utf-8-sig"),
            file_name="frecuencias_histograma_V3.csv",
            mime="text/csv",
        )

    st.download_button(
        "Descargar métricas (CSV)",
        data=datos["tabla_metricas"].to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="metricas_comparacion_histograma_V3.csv",
        mime="text/csv",
    )

    with st.expander("Notas metodológicas"):
        st.markdown(
            "- El intervalo se selecciona con el histograma empírico, no con el área de la interpolación.\n"
            "- Los valores negativos de la función transformada se convierten en cero.\n"
            "- Las frecuencias objetivo se redondean a enteros y no se normalizan posteriormente.\n"
            "- La reasignación ordena los píxeles de menor a mayor intensidad original. Si quedan píxeles tras recorrer el objetivo, se asignan a 255.\n"
            "- Las métricas comparan las versiones en escala de grises, mientras que la vista visual puede conservar el color por sustitución del canal de luminosidad LAB."
        )
