"""Funciones de la versión 3 del modelo de transformación del histograma.

Implementa el procedimiento del notebook: intervalo mínimo por porcentaje,
aproximación (PCHIP/lineal/Fourier), transformación A/B/C/D, y reasignación
ordenada de píxeles según las frecuencias objetivo.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator


def calcular_metricas(img: np.ndarray) -> dict[str, float | int]:
    img = np.asarray(img, dtype=np.uint8)
    hist = np.bincount(img.ravel(), minlength=256)
    probabilidades = hist / img.size
    probabilidades = probabilidades[probabilidades > 0]
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    p1, p99 = np.percentile(img, [1, 99])
    return {
        "Media": float(np.mean(img)),
        "Desviación estándar": float(np.std(img)),
        "Mínimo": int(img.min()),
        "Máximo": int(img.max()),
        "Rango dinámico": int(img.max() - img.min()),
        "Rango P1-P99": float(p99 - p1),
        "Entropía (bits)": float(entropia),
        "Píxeles en 0 (%)": float(100 * np.mean(img == 0)),
        "Píxeles en 255 (%)": float(100 * np.mean(img == 255)),
    }


def encontrar_intervalo_minimo(hist: np.ndarray, porcentaje: float) -> tuple[int, int, int, int, int]:
    """Intervalo más corto que contiene el porcentaje solicitado.

    Desempates: 1) menor ancho; 2) mayor cantidad de píxeles; 3) menor inicio.
    Retorna r_min, r_max, pixeles_dentro, total, pixeles_requeridos.
    """
    h = np.asarray(hist, dtype=np.int64).ravel()
    if h.size != 256:
        raise ValueError("El histograma debe tener 256 frecuencias.")
    if np.any(h < 0):
        raise ValueError("Las frecuencias no pueden ser negativas.")
    if not (0 < float(porcentaje) <= 100):
        raise ValueError("El porcentaje debe ser mayor que 0 y como máximo 100.")
    total = int(h.sum())
    if total <= 0:
        raise ValueError("El histograma está vacío.")

    pixeles_requeridos = int(math.ceil(total * float(porcentaje) / 100.0))
    prefijos = np.concatenate(([0], np.cumsum(h, dtype=np.int64)))
    candidatos = []
    for inicio in range(256):
        umbral_prefijo = int(prefijos[inicio]) + pixeles_requeridos
        if umbral_prefijo > total:
            continue
        final_exclusivo = int(np.searchsorted(prefijos, umbral_prefijo, side="left"))
        final = final_exclusivo - 1
        if final < inicio or final > 255:
            continue
        conteo = int(prefijos[final + 1] - prefijos[inicio])
        ancho = final - inicio + 1
        candidatos.append((ancho, -conteo, inicio, final, conteo))

    if not candidatos:
        raise RuntimeError("No se encontró un intervalo válido.")
    ancho, conteo_negativo, r_min, r_max, pixeles_dentro = min(candidatos)
    return int(r_min), int(r_max), int(pixeles_dentro), total, int(pixeles_requeridos)


def aproximar_histograma(histograma_rango: np.ndarray, metodo: str, num_puntos: int = 250, num_armonicos: int = 12) -> dict[str, Any]:
    histograma_rango = np.asarray(histograma_rango, dtype=float)
    longitud_rango = len(histograma_rango)
    coordenadas_locales = np.arange(longitud_rango, dtype=float)
    metodo = metodo.upper()

    if longitud_rango == 1:
        puntos_x = np.array([0], dtype=int)
        puntos_y = histograma_rango.copy()
        histograma_aproximado = histograma_rango.copy()
        armonicos_efectivos = 0
        puntos_efectivos = 1
    elif metodo in ("PCHIP", "LINEAL"):
        puntos_efectivos = min(max(2, int(num_puntos)), longitud_rango)
        puntos_x = np.unique(np.rint(np.linspace(0, longitud_rango - 1, puntos_efectivos)).astype(int))
        puntos_y = histograma_rango[puntos_x]
        if metodo == "PCHIP":
            interpolante = PchipInterpolator(puntos_x, puntos_y, extrapolate=False)
            histograma_aproximado = np.asarray(interpolante(coordenadas_locales), dtype=float)
        else:
            histograma_aproximado = np.interp(coordenadas_locales, puntos_x, puntos_y)
        armonicos_efectivos = 0
    elif metodo == "FOURIER":
        puntos_x = puntos_y = None
        puntos_efectivos = 0
        coeficientes = np.fft.rfft(histograma_rango)
        max_armonicos = len(coeficientes) - 1
        armonicos_efectivos = min(max(0, int(num_armonicos)), max_armonicos)
        coeficientes_filtrados = np.zeros_like(coeficientes)
        coeficientes_filtrados[:armonicos_efectivos + 1] = coeficientes[:armonicos_efectivos + 1]
        histograma_aproximado = np.fft.irfft(coeficientes_filtrados, n=longitud_rango)
    else:
        raise ValueError("Método no válido. Usa PCHIP, LINEAL o FOURIER.")

    return {
        "metodo": metodo,
        "histograma_aproximado": np.asarray(histograma_aproximado, dtype=float),
        "puntos_x": puntos_x,
        "puntos_y": puntos_y,
        "puntos_efectivos": int(puntos_efectivos),
        "armonicos_efectivos": int(armonicos_efectivos),
        "coordenadas_locales": coordenadas_locales,
        "error_absoluto_medio": float(np.mean(np.abs(histograma_rango - histograma_aproximado))),
    }


def calcular_transformacion(
    imagen_gris: np.ndarray,
    r_min: int,
    r_max: int,
    histograma_aproximado: np.ndarray,
) -> dict[str, Any]:
    """Aplica la transformación V3 y calcula C para alcanzar N tras redondear."""
    n_pixeles = int(imagen_gris.size)
    longitud_rango = int(r_max - r_min + 1)
    rd = int(r_max - r_min)  # Diferencia de extremos, como en el notebook V3.
    A = 255.0 / rd if rd > 0 else 1.0
    B = -float(r_min)
    D = 0.0

    niveles_float = np.arange(256, dtype=float)
    coordenadas_locales = np.arange(longitud_rango, dtype=float)
    coordenada_fuente = (niveles_float - r_min - B) / A
    soporte_transformado = (coordenada_fuente >= 0) & (coordenada_fuente <= longitud_rango - 1)

    valores_fuente = np.interp(
        coordenada_fuente,
        coordenadas_locales,
        np.asarray(histograma_aproximado, dtype=float),
        left=0.0,
        right=0.0,
    )

    base_transformada = np.zeros(256, dtype=float)
    base_transformada[soporte_transformado] = np.maximum(valores_fuente[soporte_transformado], 0.0)
    suma_base = float(base_transformada.sum())
    if suma_base <= 0:
        raise ValueError("La curva transformada tiene área cero; no se puede calcular C.")

    area_aproximacion_original = float(np.maximum(np.asarray(histograma_aproximado, dtype=float), 0.0).sum())
    C_area_aproximacion = area_aproximacion_original / suma_base
    C_area_total_pixeles = n_pixeles / suma_base

    def frecuencias_para_c(c: float) -> np.ndarray:
        return np.rint(c * base_transformada).astype(np.int64)

    def suma_frecuencias_para_c(c: float) -> int:
        return int(frecuencias_para_c(c).sum())

    inferior = 0.0
    superior = max(C_area_total_pixeles, np.finfo(float).tiny)
    while suma_frecuencias_para_c(superior) < n_pixeles:
        superior *= 2.0

    for _ in range(200):
        medio = inferior + (superior - inferior) / 2.0
        if medio == inferior or medio == superior:
            break
        if suma_frecuencias_para_c(medio) >= n_pixeles:
            superior = medio
        else:
            inferior = medio

    C = superior
    for _ in range(1000):
        anterior = np.nextafter(C, -np.inf)
        if anterior <= 0 or suma_frecuencias_para_c(anterior) < n_pixeles:
            break
        C = anterior

    funcion_transformada_bruta = np.zeros(256, dtype=float)
    funcion_transformada_bruta[soporte_transformado] = C * valores_fuente[soporte_transformado] + D
    funcion_transformada = np.maximum(funcion_transformada_bruta, 0.0)
    frecuencias_objetivo = np.maximum(np.rint(funcion_transformada).astype(np.int64), 0)

    return {
        "rd": rd,
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D),
        "C_area_aproximacion": float(C_area_aproximacion),
        "C_area_total_pixeles": float(C_area_total_pixeles),
        "area_aproximacion_original": float(area_aproximacion_original),
        "area_transformada_sin_C": float(suma_base),
        "coordenada_fuente": coordenada_fuente,
        "soporte_transformado": soporte_transformado,
        "valores_fuente": valores_fuente,
        "base_transformada": base_transformada,
        "funcion_transformada_bruta": funcion_transformada_bruta,
        "funcion_transformada": funcion_transformada,
        "frecuencias_objetivo": frecuencias_objetivo,
        "suma_objetivo": int(frecuencias_objetivo.sum()),
        "diferencia_objetivo_total": int(frecuencias_objetivo.sum() - n_pixeles),
    }


def reasignar_pixeles_por_frecuencia(imagen_gris: np.ndarray, frecuencias: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    original = np.asarray(imagen_gris, dtype=np.uint8)
    plano = original.ravel()
    total = plano.size
    frecuencias = np.maximum(np.asarray(frecuencias, dtype=np.int64).ravel(), 0)
    if frecuencias.size != 256:
        raise ValueError("Se requieren exactamente 256 frecuencias objetivo.")

    orden = np.argsort(plano, kind="stable")
    salida_plana = np.empty_like(plano)
    cursor = 0
    asignados_por_nivel = np.zeros(256, dtype=np.int64)
    for nivel in range(256):
        disponibles = total - cursor
        if disponibles <= 0:
            break
        cantidad = min(int(frecuencias[nivel]), disponibles)
        if cantidad > 0:
            indices = orden[cursor:cursor + cantidad]
            salida_plana[indices] = nivel
            asignados_por_nivel[nivel] += cantidad
            cursor += cantidad

    sobrantes = total - cursor
    if sobrantes > 0:
        salida_plana[orden[cursor:]] = 255
        asignados_por_nivel[255] += sobrantes

    salida = salida_plana.reshape(original.shape)
    histograma_logrado = np.bincount(salida.ravel(), minlength=256)
    return salida, histograma_logrado, asignados_por_nivel, int(sobrantes)


def preservar_color(imagen_rgb: np.ndarray, imagen_intensidad: np.ndarray, es_color: bool) -> np.ndarray:
    """Sustituye la luminosidad LAB y conserva los canales cromáticos cuando la original es color."""
    imagen_intensidad = np.asarray(imagen_intensidad, dtype=np.uint8)
    if not es_color:
        return imagen_intensidad
    imagen_lab = cv2.cvtColor(np.asarray(imagen_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    imagen_lab[:, :, 0] = imagen_intensidad
    return cv2.cvtColor(imagen_lab, cv2.COLOR_LAB2RGB)


def ejecutar_modelo(
    imagen_gris: np.ndarray,
    porcentaje: float,
    metodo: str,
    num_puntos: int = 250,
    num_armonicos: int = 12,
) -> dict[str, Any]:
    histograma = np.bincount(np.asarray(imagen_gris, dtype=np.uint8).ravel(), minlength=256).astype(np.int64)
    cdf_original = np.cumsum(histograma) / imagen_gris.size
    r_min, r_max, pixeles_dentro, total_pixeles, pixeles_requeridos = encontrar_intervalo_minimo(histograma, porcentaje)
    histograma_rango = histograma[r_min:r_max + 1].astype(float).copy()
    aproximacion = aproximar_histograma(histograma_rango, metodo, num_puntos, num_armonicos)
    transformacion = calcular_transformacion(imagen_gris, r_min, r_max, aproximacion["histograma_aproximado"])
    imagen_propuesta, histograma_logrado, asignados_por_nivel, sobrantes_255 = reasignar_pixeles_por_frecuencia(imagen_gris, transformacion["frecuencias_objetivo"])

    imagen_he = cv2.equalizeHist(np.asarray(imagen_gris, dtype=np.uint8))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagen_clahe = clahe.apply(np.asarray(imagen_gris, dtype=np.uint8))
    resultados_grises = {
        "Original": np.asarray(imagen_gris, dtype=np.uint8),
        "Modelo V3 intervalo mínimo": imagen_propuesta,
        "Ecualización clásica (HE)": imagen_he,
        "CLAHE": imagen_clahe,
    }
    histos_modelos = {}
    cdfs_modelos = {}
    for nombre, img in resultados_grises.items():
        h = np.bincount(img.ravel(), minlength=256)
        histos_modelos[nombre] = h
        cdfs_modelos[nombre] = np.cumsum(h) / img.size

    import pandas as pd
    tabla_metricas = pd.DataFrame({nombre: calcular_metricas(img) for nombre, img in resultados_grises.items()}).T

    return {
        "histograma": histograma,
        "cdf_original": cdf_original,
        "r_min": r_min,
        "r_max": r_max,
        "rango_dinamico": r_max - r_min + 1,
        "pixeles_dentro": pixeles_dentro,
        "total_pixeles": total_pixeles,
        "pixeles_requeridos": pixeles_requeridos,
        "pixeles_fuera": total_pixeles - pixeles_dentro,
        "porcentaje_real": 100 * pixeles_dentro / total_pixeles,
        "porcentaje_solicitado": float(porcentaje),
        "histograma_rango": histograma_rango,
        "niveles_absolutos_rango": np.arange(r_min, r_max + 1),
        "aproximacion": aproximacion,
        "transformacion": transformacion,
        "imagen_propuesta": imagen_propuesta,
        "histograma_logrado": histograma_logrado,
        "asignados_por_nivel": asignados_por_nivel,
        "sobrantes_255": sobrantes_255,
        "imagen_he": imagen_he,
        "imagen_clahe": imagen_clahe,
        "resultados_grises": resultados_grises,
        "histos_modelos": histos_modelos,
        "cdfs_modelos": cdfs_modelos,
        "tabla_metricas": tabla_metricas,
    }
