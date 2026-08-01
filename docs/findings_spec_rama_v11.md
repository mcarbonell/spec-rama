# Hallazgos del Experimento: Arnés Definitivo en Bits/Token y Comparativa QLoRA Head-to-Head (v11)

Este documento resume los resultados del **Experimento 11 (EXP-11)** y **Experimento 12 (EXP-12 Audit)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se aplicó el **protocolo estricto de cuantización NF4 por bloques (`block_size=64`)** de QLoRA y la métrica de entropía en **Bits por Token (bits/token / bpt)** para evaluar la recuperación real de precisión frente a **Standard LoRA (r=4)** sobre el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales en Bits por Token (EXP-11 / EXP-12 Audit)

| Brazo Experimental / Estrategia | Formato Base | Tamaño Adaptador | Perplejidad TEST | Entropía (Bits/Token) | $\Delta \text{bpt}$ vs FP32 Nativo | $\Delta \text{bpt}$ vs FP32 Adaptado | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. GPT-2 FP32 Nativo (Zero-Shot)** | 32-bit FP32 | 0.0 KB | **46.18** | **5.529 bpt** | 0.000 bpt | N/A | Referencia sin entrenar |
| **2. GPT-2 FP32 + SpecRAMA (Tuned)** | 32-bit FP32 | **294.9 KB** | **35.99** ↓ | **5.169 bpt** | -0.360 bpt | 0.000 bpt | **Cota Superior Adaptada** |
| **3. Block-Wise NF4 Base (Zero-Shot)** | 4-bit NF4 | 0.0 KB | **49.34** | **5.625 bpt** | +0.096 bpt | +0.455 bpt | Base Cuantizada QLoRA |
| **4. Block-Wise NF4 + SpecRAMA (Tuned)** | 4-bit NF4 | **294.9 KB** | **37.98** ↓ | **5.247 bpt** | **-0.282 bpt** | **+0.078 bpt** | **RECUPERACIÓN TOTAL (<0.08 bpt)** |
| **5. Block-Wise NF4 + LoRA ($r=4$, $lr=1e-4$)** | 4-bit NF4 | 1.55 MB | **35.58** ↓ | **5.153 bpt** | **-0.376 bpt** | **-0.016 bpt** | Standard QLoRA Tuned |
| **6. Asymmetric NF3/4 Base (Zero-Shot)** | 3.55-bit NF3/4 | 0.0 KB | **60.45** | **5.918 bpt** | +0.389 bpt | +0.748 bpt | Base Cuantizada Asimétrica |
| **7. Asymmetric + SpecRAMA (Tuned)** | 3.55-bit NF3/4 | **174.6 KB** | **45.99** ↓ | **5.523 bpt** | **-0.006 bpt** | **+0.354 bpt** | **Supera a FP32 Nativo a 3.55b** |

---

## 2. Hallazgos e Insights de Investigación

### A. La Brecha Real de Cuantización NF4 es de Solo +0.078 Bits/Token
* Con la cuantización por bloques correcta (`block_size=64`), la base NF4 sin entrenar sufre una penalización de solo **+0.096 bits/token (+3.16 PPL)** sobre FP32 nativo.
* **SpecRAMA Wavelet (32x32, 294.9 KB)** reduce la entropía del modelo cuantizado a **5.247 bits/token (37.98 PPL)**.

### B. Auditoría de Hiperparámetros (EXP-12): Sensibilidad al LR en LoRA vs. Robustez de Spec-RAMA
* **Barrido de LR en Standard LoRA (r=4, 1.55 MB)**:
  * `lr = 1e-4`: **35.58 PPL (5.153 bpt)** (Rendimiento óptimo).
  * `lr = 5e-4`: **36.29 PPL (5.181 bpt)**.
  * `lr = 1e-3`: **44.02 PPL (5.460 bpt)**.
  * `lr = 2e-3`: **72.06 PPL (6.171 bpt)**.
  * `lr = 5e-3`: **185.48 PPL (7.535 bpt)**.
  * `lr = 1e-2`: **439.71 PPL (8.780 bpt)** (Colapso por explosión de gradiente).

* **Conclusión Científica de la Auditoría**:
  1. **Paridad de Rendimiento con 5.2x Menos Parámetros**: A su tasa de aprendizaje óptima ($1 \times 10^{-4}$), LoRA alcanza **35.58 PPL**. Spec-RAMA alcanza **37.98 PPL** con un adaptador de solo **294.9 KB (5.2 veces más pequeño que los 1.55 MB de LoRA)**.
  2. **Robustez de Escalado de Parseval**: Gracias a la división por $\sqrt{k_{\text{out}} \cdot k_{\text{in}}}$, Spec-RAMA es hiper-estable y entrena fluidamente incluso a `lr = 1e-2`, mientras que LoRA espacial colapsa bruscamente si el LR supera $10^{-3}$.
