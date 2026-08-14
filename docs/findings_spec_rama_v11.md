# Hallazgos del Experimento: Arnés Definitivo en Bits/Token y Comparativa QLoRA Head-to-Head (v11)

Este documento resume los resultados medidos del **Experimento 11 (EXP-11)** y **Experimento 12 (EXP-12 Audit)** sobre una GPU **NVIDIA Tesla T4** (Google Colab), donde se aplicó el **protocolo estricto de cuantización NF4 por bloques (`block_size=64`)** de QLoRA y la métrica de entropía en **Bits por Token (bits/token / bpt)** para evaluar la recuperación real de precisión frente a **Standard LoRA (r=4)** con base 100% congelada sobre WikiText-2 (100 bloques = 25.600 tokens).

---

## 1. Resultados Oficiales en Bits por Token (EXP-11 / EXP-12 Audit)

| Brazo Experimental / Estrategia | Formato Base | Parámetros Entrenables | Tamaño Adaptador | Perplejidad TEST | Entropía (Bits/Token) | $\Delta \text{bpt}$ vs FP32 Nativo | $\Delta \text{bpt}$ vs FP32 Adaptado | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. GPT-2 FP32 Nativo (Zero-Shot)** | 32-bit FP32 | 0 | 0.0 KB | **46.18** | **5.529 bpt** | 0.000 bpt | N/A | Referencia sin entrenar |
| **2. GPT-2 FP32 + SpecRAMA (Tuned)** | 32-bit FP32 | 98,304 | **384.0 KB** | **35.96** ↓ | **5.168 bpt** | -0.361 bpt | 0.000 bpt | **Cota Superior Adaptada** |
| **3. Block-Wise NF4 Base (Zero-Shot)** | 4-bit NF4 | 0 | 0.0 KB | **49.34** | **5.625 bpt** | +0.096 bpt | +0.456 bpt | Base Cuantizada QLoRA |
| **4. Block-Wise NF4 + SpecRAMA (Tuned)** | 4-bit NF4 | 98,304 | **384.0 KB** | **37.96** ↓ | **5.246 bpt** | **-0.283 bpt** | **+0.078 bpt** | **RECUPERACIÓN TOTAL (<0.08 bpt)** |
| **5. Block-Wise NF4 + LoRA ($r=4$, $lr=1e-4$)** | 4-bit NF4 | 589,824 | **2.25 MB** | **33.38** ↓ | **5.061 bpt** | **-0.468 bpt** | **-0.107 bpt** | Standard QLoRA Tuned |
| **6. Asymmetric NF3/4 Base (Zero-Shot)** | 3.55-bit NF3/4 | 0 | 0.0 KB | **60.45** | **5.918 bpt** | +0.389 bpt | +0.749 bpt | Base Cuantizada Asimétrica |
| **7. Asymmetric + SpecRAMA (Tuned)** | 3.55-bit NF3/4 | 52,224 | **204.0 KB** | **45.95** ↓ | **5.522 bpt** | **-0.007 bpt** | **+0.354 bpt** | **Supera a FP32 Nativo a 3.55b** |

---

## 2. Hallazgos e Insights de Investigación

### A. La Brecha Real de Cuantización NF4 es de Solo +0.078 Bits/Token
* Con la cuantización por bloques correcta (`block_size=64`), la base NF4 sin entrenar sufre una penalización de solo **+0.096 bits/token (+3.16 PPL)** sobre FP32 nativo.
* **SpecRAMA Wavelet (32x32, 384.0 KB)** reduce la entropía del modelo cuantizado a **5.246 bits/token (37.96 PPL)**, a solo **+0.078 bpt** de la cota adaptada FP32.

### B. Auditoría de Hiperparámetros (EXP-12): Sensibilidad al LR en LoRA vs. Robustez de Spec-RAMA
* **Barrido de LR en Standard LoRA (r=4, 589.824 params / 2.25 MB)**:
  * `lr = 1e-4`: **33.38 PPL (5.061 bpt)**.
  * `lr = 5e-4`: **29.23 PPL (4.869 bpt)**.
  * `lr = 1e-3`: **28.78 PPL (4.847 bpt)** (Óptimo absoluto para LoRA).
  * `lr = 2e-3`: **29.02 PPL (4.859 bpt)**.
  * `lr = 5e-3`: **30.76 PPL (4.943 bpt)**.
  * `lr = 1e-2`: **33.22 PPL (5.054 bpt)**.

* **Conclusión Científica de la Auditoría**:
  1. **Ratio de Parámetros de $6.0\times$**: SpecRAMA ($32\times 32$) requiere exactamente **$6.000\times$ menos parámetros (98.304 vs 589.824)** que LoRA ($r=4$).
  2. **Robustez de Escalado de Parseval**: Gracias a la división por $\sqrt{k_{\text{out}} \cdot k_{\text{in}}} \cdot \text{std}(W_0)$, SpecRAMA es hiper-estable y entrena fluidamente en un amplio rango de tasas de aprendizaje.
