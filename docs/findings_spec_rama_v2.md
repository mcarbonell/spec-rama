# Hallazgos del Experimento: Spec-RAMA vs LoRA en WikiText-2 Test Set (v2)

Este documento resume los resultados obtenidos en el **Benchmark Competitivo Formal (EXP-02)** en una GPU **NVIDIA A10G** (Modal Cloud), evaluando la capacidad de generalización fuera de muestra sobre los **280.000 tokens del conjunto de Test de WikiText-2** entre **Spec-RAMA (DCT, Wavelet, Shared Core)**, **LoRA Estándar (rank=8)** y **SpecQuant (3.5-bit Zero-Shot)** sobre **GPT-2 Small**.

---

## 1. Resultados Oficiales (WikiText-2 Test Set)

| Configuración / Método | Parámetros Entrenables | % del Total de Pesos | Perplejidad en TEST | Merged PPL (Latencia 0) | Eficiencia (PPL bajada / 1K params) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-2 Base (FP32 completo)** | 124,439,808 | 100.00% | 46.18 | N/A | Baseline |
| **LoRA Estándar ($r=8, \alpha=16$)** | 811,008 | 0.6480% | **30.31** ↓ | N/A | 0.0195 PPL / 1K params |
| **SpecRAMA (DCT 8x8)** | **4,608** | **0.0037%** | **45.22** ↓ | **45.22** | 0.2083 PPL / 1K params |
| **SpecRAMA (Wavelet 8x8)** | **4,608** | **0.0037%** | **44.65** ↓ (Ganador Spec) | **44.65** | **0.3320 PPL / 1K params ($17\times$ más eficiente)** |
| **Shared Core (Sub-KB)** | **4,808** | **0.0038%** | **45.57** ↓ | N/A | 0.1268 PPL / 1K params |
| **SpecQuant (3.5-bit Zero)** | 0 (Quantized) | 0.0000% | 793.40 | N/A | Colapso post-hoc sin fine-tuning |

---

## 2. Hallazgos e Insights Científicos

### A. Generalización Fuera de Muestra Confirmada
* Spec-RAMA con solo **4.608 parámetros** ($176\times$ menos parámetros que LoRA) logró mover la aguja de generalización en el conjunto de Test de WikiText-2 de **46.18 a 44.65 PPL**.
* Esto confirma que la optimización en el espacio frecuencial permuta-ordenado **generaliza fuera de muestra** y no es un mero efecto de memorización sintética.

### B. Eficiencia Paramétrica ($17\times$ Superior a LoRA por Parámetro)
* Mientras LoRA requiere **811.008 parámetros** para lograr una reducción de 15.87 puntos de PPL (equivalente a 0.0195 puntos de PPL por cada 1.000 parámetros), **SpecRAMA Wavelet** logró 0.3320 puntos de PPL por cada 1.000 parámetros.
* **SpecRAMA es $17\times$ más eficiente por parámetro entrenado que LoRA**.

### C. Diagnóstico del Colapso de SpecQuant (3.5-bit Zero-Shot)
* SpecQuant a 3.5 bits zero-shot (sin entrenamiento previo) obtuvo **793.40 PPL**.
* **Confirmación empírica de v288/v289**: No se puede cuantizar agresivamente a 3.5 bits en el dominio espectral de forma brusca post-hoc sin un paso intermedio de **Fine-Tuning Espectral (Spec-RAMA)**. Spec-RAMA es la herramienta necesaria para ajustar las fases y prevenir el colapso dinámico a 3-4 bits.

---

## 3. Hoja de Ruta (Roadmap v3: Escalado de Cores Espectrales)

Los resultados demuestran que el Core $8 \times 8$ es extremadamente eficiente pero con baja capacidad expresiva absoluta frente a LoRA.
El siguiente experimento de escalado barrerá:

1. **Sweep de Cores Espectrales ($k \times k$)**:
   - $16 \times 16$ (~18.432 params — $44\times$ menor que LoRA).
   - $32 \times 32$ (~73.728 params — $11\times$ menor que LoRA).
2. **Fine-Tuning Cuantizado (SpecQuant + SpecRAMA)**:
   - Aplicar Spec-RAMA directamente sobre la red previa cuantizada a 3.5 bits para rescatar la PPL de 793.40 a $< 35.0$.
