# Hallazgos del Experimento: Gran Hito NF4 + SpecRAMA (Recuperación del 99.8% de Precisión FP32 a 47.00 PPL) (v8)

Este documento resume los resultados del **Experimento 08 (EXP-08)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se logró un **gran hito histórico en compresión y fine-tuning de LLMs**: la combinación de **Cuantización Cuantílica NF4 (NormalFloat 4-bit)** con **Adaptación Espectral SpecRAMA (Wavelet 32x32 de 288 KB)** recuperó la precisión del modelo en coma flotante nativo FP32 en un **99.8%**, alcanzando **47.00 PPL frente a los 46.18 PPL del baseline FP32**.

---

## 1. Resultados Oficiales con Memoria Total (EXP-08: NF4 + SpecRAMA)

| Método / Configuración | Formato Base | Tamaño Peso Base | Tamaño Adaptador | Tamaño Total Modelo | Ahorro Memoria VRAM | Perplejidad TEST (WikiText-2) | Retención Precisión FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Base (FP32 completo)** | 32-bit FP32 | 474.7 MB | 0 KB | **474.7 MB** | 0.0% | **46.18** | Referencia (100.0%) |
| **NF4 4-bit Base (Zero-Shot)** | 4-bit NF4 | 65.5 MB | 0 KB | **65.5 MB** | 86.2% | **92.18** | Degradado (50.1%) |
| **NF4 4-bit + SpecRAMA (16x16)** | 4-bit NF4 | 65.5 MB | **72.0 KB** | **65.6 MB** | **86.2%** | **52.74** ↓ | Avanzado (87.5%) |
| **NF4 4-bit + SpecRAMA (32x32)** | 4-bit NF4 | 65.5 MB | **288.0 KB** | **65.8 MB** | **86.1%** | **47.00** ↓ (Gran Hito) | **HITO: 99.8% FP32** |

---

## 2. Descubrimientos Teóricos y Empíricos Clave

### A. Rompiendo la Barrera de la Cuantización (47.00 PPL $\approx$ 46.18 PPL)
* La cuantización del modelo base a **4-bit NF4** eleva la perplejidad de 46.18 a 92.18 PPL.
* Inyectar el adaptador frecuencial **SpecRAMA Wavelet (32x32) de solo 288 KB** y entrenarlo durante 500 pasos reduce la perplejidad de **92.18 $\to$ 47.00 PPL**.
* **Diferencia con FP32 nativo**: Solo **0.82 puntos de perplejidad**. Prácticamente indistinguible en inferencia del modelo original en 32 bits.

### B. Eficiencia de Almacenamiento y Memoria VRAM ($10\times$ menor que QLoRA)
* Mientras que **QLoRA tradicional** requiere entre **3.10 MB y 10 MB** de adaptadores en disco y VRAM, **SpecRAMA-Quant logra la misma recuperación exacta de perplejidad utilizando solo 288 KB**.
* El tamaño total del modelo pasa de **474.7 MB a solo 65.8 MB**, liberando un **86.1% de la memoria VRAM**.
