# Transformación del histograma — Versión 3

Aplicación de Streamlit basada en el notebook `Transformacion_histograma_V3_intervalo_minimo.ipynb`.

## Qué hace

1. Convierte la imagen cargada a escala de grises para los cálculos del modelo.
2. Calcula histograma y CDF.
3. Busca el intervalo de menor amplitud que contiene al menos el porcentaje solicitado de píxeles.
4. Aplica la aproximación elegida: PCHIP, lineal o Fourier.
5. Calcula automáticamente los parámetros horizontales `A=255/(r_max-r_min)` y `B=-r_min` (con un caso especial si el intervalo contiene una única intensidad), y calcula `C` para que la suma de frecuencias objetivo redondeadas alcance el número total de píxeles. `D=0`.
6. Construye el histograma objetivo y reasigna los píxeles según el orden de intensidad original.
7. Compara el resultado con ecualización clásica (HE) y CLAHE, con gráficos de histogramas/CDF y una tabla de métricas.
8. Cuando la imagen original es a color, la vista visual de cada método reemplaza el canal de luminosidad LAB y conserva los componentes cromáticos originales tanto como permite la conversión.

## Ejecutar localmente

Requiere Python 3.11 recomendado.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Instala dependencias e inicia Streamlit:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Publicar en GitHub y Streamlit Community Cloud

1. Crea un repositorio nuevo en GitHub, por ejemplo `laboratorio-transformacion-histograma-v3`.
2. Descomprime el ZIP y sube todos los archivos y la carpeta de proyecto al nivel principal del repositorio (no subas solo el ZIP).
3. Entra en <https://share.streamlit.io/> y conecta tu cuenta de GitHub.
4. Crea una aplicación nueva, elige el repositorio y la rama `main`.
5. Indica `streamlit_app.py` como archivo principal y pulsa **Deploy**.

## Archivos

- `streamlit_app.py`: interfaz principal con pestaña Resultados y pestaña Desarrollador.
- `model.py`: cálculo matemático y procesamiento de imágenes.
- `requirements.txt`: dependencias de Python.
- `Transformacion_histograma_V3_intervalo_minimo.ipynb`: notebook de referencia para Google Colab.
- `.gitignore`: exclusiones recomendadas de Git.

## Sobre la pestaña Desarrollador

Contiene el intervalo seleccionado, porcentajes, cantidades de píxeles, parámetros `A`, `B`, `C` y `D`, suma de frecuencias objetivo, sobras asignadas a 255, error de aproximación y exportación CSV. La pestaña Resultados prioriza la imagen original, gráficos, comparación visual y métricas.

## Nota de interpretación

La selección del intervalo optimiza la amplitud para una cobertura dada; no garantiza por sí sola que se conserven todos los detalles perceptualmente importantes. El resultado depende también de la aproximación, los parámetros derivados y la regla de reasignación. La aplicación tiene una finalidad experimental y comparativa.
