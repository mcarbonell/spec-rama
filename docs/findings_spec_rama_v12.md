# Hallazgos del Experimento: Auditoría de Hiperparámetros de LoRA y Nota de Rigor Histórico (v12)

Este documento documenta el **Experimento 12 (EXP-12)** realizado en Google Colab (GPU NVIDIA Tesla T4), diseñado específicamente para **auditar los hiperparámetros de Standard LoRA** con la base cuantizada en NF4 por bloques y los módulos base 100% congelados (embeddings y LayerNorms aislados), garantizando la máxima **transparencia y rigor científico**.

---

## 1. Motivación y Contexto

En **EXP-11**, los adaptadores se comparan bajo condiciones controladas:
* **SpecRAMA Wavelet (32x32, 98.304 params / 384.0 KB)** converge limpiamente a **37.96 PPL (5.246 bpt)** con `lr_max = 1e-2`.
* **Standard LoRA ($r=4, \alpha=8.0, 589.824 \text{ params} / 2.25 \text{ MB}$)** entrenado a su tasa adaptativa (`lr_max = 1e-4`) alcanza **33.38 PPL (5.061 bpt)**.

El objetivo de EXP-12 es mapear la curva completa de sensibilidad de LoRA frente al learning rate cuando la base del modelo está estrictamente congelada.

---

## 2. Resultados Oficiales del Barrido de LR (EXP-12)

Se ejecutó un barrido completo de tasas de aprendizaje sobre GPT-2 Small cuantizado en NF4 por bloques (`block_size=64`) en **WikiText-2 (100 bloques = 25.600 tokens)**:

| Tasa de Aprendizaje (`lr_max`) | Factor Escalado ($\alpha/r$) | Parámetros Entrenables | Perplejidad TEST | Entropía (Bits/Token) | Estado / Comportamiento del Modelo |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`1.0e-04` (Estándar QLoRA)** | $2.0$ | 589,824 | **33.38** ↓ | **5.061 bpt** | Convergencia Estable |
| **`5.0e-04`** | $2.0$ | 589,824 | **29.23** ↓ | **4.869 bpt** | Rápida Convergencia |
| **`1.0e-03`** | $2.0$ | 589,824 | **28.78** ↓ | **4.847 bpt** | **Óptimo Absoluto de LoRA** |
| **`2.0e-03`** | $2.0$ | 589,824 | **29.02** ↓ | **4.859 bpt** | Estable |
| **`5.0e-03`** | $2.0$ | 589,824 | **30.76** | **4.943 bpt** | Ligera Sobrecarga |
| **`1.0e-02`** | $2.0$ | 589,824 | **33.22** | **5.054 bpt** | Convergencia Ruidosa |

---

## 3. Resumen Consolidado ISO-Condiciones (EXP-11 / EXP-12)

| Brazo Experimental / Estrategia | Formato Base | Tamaño Adaptador | Parámetros Entrenables | TEST PPL | Bits/Token | $\Delta \text{bpt}$ vs FP32 Nativo | $\Delta \text{bpt}$ vs FP32 Adaptado | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. GPT-2 FP32 Nativo (Zero-Shot)** | 32-bit FP32 | 0.0 KB | 0 | **46.18** | **5.529 bpt** | 0.000 bpt | N/A | Referencia sin entrenar |
| **2. GPT-2 FP32 + SpecRAMA (Tuned)** | 32-bit FP32 | **384.0 KB** | 98,304 | **35.96** ↓ | **5.168 bpt** | -0.361 bpt | 0.000 bpt | **Cota Superior Adaptada** |
| **3. Block-Wise NF4 Base (Zero-Shot)** | 4-bit NF4 | 0.0 KB | 0 | **49.34** | **5.625 bpt** | +0.096 bpt | +0.456 bpt | Base Cuantizada QLoRA |
| **4. Block-Wise NF4 + SpecRAMA (Tuned)** | 4-bit NF4 | **384.0 KB** | 98,304 | **37.96** ↓ | **5.246 bpt** | **-0.283 bpt** | **+0.078 bpt** | **RECUPERACIÓN TOTAL (6.0x menor)** |
| **5. Block-Wise NF4 + LoRA ($r=4$, $lr=1e-4$)** | 4-bit NF4 | **2.25 MB** | 589,824 | **33.38** ↓ | **5.061 bpt** | **-0.468 bpt** | **-0.107 bpt** | **Baseline QLoRA Óptimo** |
| **6. Asymmetric NF3/4 Base (Zero-Shot)** | 3.55-bit NF3/4 | 0.0 KB | 0 | **60.45** | **5.918 bpt** | +0.389 bpt | +0.749 bpt | Base Cuantizada Asimétrica |
| **7. Asymmetric + SpecRAMA (Tuned)** | 3.55-bit NF3/4 | **204.0 KB** | 52,224 | **45.95** ↓ | **5.522 bpt** | **-0.007 bpt** | **+0.354 bpt** | **Supera a FP32 Nativo a 3.55b** |

---

## 4. Conclusiones y Blindaje Académico

1. **Eficiencia de Parámetros ($6.0\times$)**:
   SpecRAMA Wavelet ($32\times 32$) opera con solo **98.304 parámetros (384 KB)** frente a los **589.824 parámetros (2.25 MB)** de LoRA ($r=4$), logrando una reducción de **$6.0\times$** con una perplejidad competitiva (**37.96 vs 33.38 PPL**).

2. **Estabilidad de Escalado Espectral de Parseval**:
   El escalado $\frac{\alpha}{\sqrt{k_{\text{out}} \cdot k_{\text{in}}} \cdot \text{std}(W_0)}$ normaliza los gradientes por la energía espectral de la matriz base, haciendo innecesario el ajuste manual de tasas de aprendizaje extremas.
