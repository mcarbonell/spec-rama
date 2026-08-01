# Hallazgos del Experimento: Control Riguroso de 6 Brazos (Daño de Cuantización vs. Adaptación de Dominio) (v10)

Este documento resume los resultados del **Experimento 10 (EXP-10)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se ejecutó un **arnés de control riguroso de 6 brazos de estudio** para desglosar y separar quirúrgicamente el **efecto de adaptación de dominio (WikiText-2 Train)** de la **recuperación real de daño de cuantización espectral (NF4 4-bit y NF3 3.55-bit)**.

---

## 1. Tabla Oficial de Control Riguroso de 6 Brazos (EXP-10)

| Brazo Experimental / Estrategia | Formato Base | Tamaño Adaptador | Perplejidad TEST (WikiText-2) | Retención vs FP32 Nativo (46.18 PPL) | Retención vs FP32 Adaptado (37.69 PPL) | Estado del Modelo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. GPT-2 FP32 Nativo (Zero-Shot)** | 32-bit FP32 | 0.0 KB | **46.18** | 100.0% (46.18 PPL) | N/A | Referencia sin entrenar |
| **2. GPT-2 FP32 + SpecRAMA (Tuned)** | 32-bit FP32 | **294.9 KB** | **37.69** ↓ | **122.5%** (Mejora de Dominio) | **100.0%** (37.69 PPL) | **Cota Superior Adaptada** |
| **3. Symmetric NF4 Base (Zero-Shot Sin Adaptador)** | 4-bit NF4 | 0.0 KB | **92.18** | 50.1% | 40.9% | Degradación de Cuantización |
| **4. Symmetric NF4 + SpecRAMA (Tuned)** | 4-bit NF4 | **294.9 KB** | **46.98** ↓ | **98.3%** | **80.2%** | **Recuperación Masiva (-45.2 PPL)** |
| **5. Asymmetric Base (Zero-Shot Sin Adaptador)** | 3.55-bit NF3/4 | 0.0 KB | **597.43** | 7.7% | 6.3% | Colapso de Cuantización |
| **6. Asymmetric (Attn-4b/FFN-3b) + SpecRAMA** | 3.55-bit NF3/4 | **174.6 KB** | **88.55** ↓ | **52.1%** | **42.6%** | **Recuperación de -508.9 PPL** |

---

## 2. Hallazgos e Insights Teóricos y Empíricos

### A. Aislamiento del Confound de Adaptación de Dominio
* **Cota Superior Adaptada**: Entrenar un adaptador SpecRAMA sobre el modelo en coma flotante FP32 reduce la perplejidad de **46.18 $\to$ 37.69 PPL** debido al ajuste de dominio a WikiText-2.
* **Retención Real del Modelo Cuantizado**:
  * Frente a **FP32 Nativo sin entrenar (46.18 PPL)**: El modelo a 4 bits rescata el **98.3% de la precisión**.
  * Frente a **FP32 Adaptado (37.69 PPL)**: La cuantización NF4 impone una penalización real de **9.29 puntos de perplejidad** (retención del **80.2%**).

### B. Absorción Directa del Ruido de Cuantización
* Sin el adaptador (`Brazo 3`), la cuantización a 4-bit NF4 causa un daño que eleva la perplejidad a **92.18 PPL**.
* Al inyectar el adaptador SpecRAMA de **294.9 KB** (`Brazo 4`), la perplejidad cae de **92.18 $\to$ 46.98 PPL**, absorbiendo **45.20 puntos absolutos de perplejidad**.

### C. Honestidad Reportada en el Régimen Asimétrico a 3.55 Bits
* El modelo sin entrenar a 3.55 bits (`Brazo 5`) colapsa a **597.43 PPL**.
* SpecRAMA (`Brazo 6`) absorbe **508.88 puntos de perplejidad** dejando el modelo en **88.55 PPL** (retención del **52.1% vs FP32 Nativo** y **42.6% vs FP32 Adaptado**), ocupando solo **55.0 MB de memoria VRAM total**.

---

## 3. Atribución Bibliográfica Formal

Se reconoce formalmente en la documentación el trabajo previo directo en PEFT frecuencial y compresión de baja memoria:

1. **FourierFT (Gao et al., ICML 2024)**: *"Parameter-Efficient Fine-Tuning with Discrete Fourier Transform"*. Prior art directo en PEFT espectral 1D.
2. **VeRA (Kopiczko et al., ICLR 2024)**: Vector-based Random Fine-Tuning en el régimen sub-100K parámetros.
3. **QuIP# / QuaRot / SpinQuant**: Rotaciones ortogonales gaussianas para mitigar outliers.

**Contribución Específica de Spec-RAMA**:
Permutación de Pesos 2D via Bipartite TSP + Escalado Frecuencial de Energía de Parseval ($\frac{\alpha}{\sqrt{k_{\text{out}} k_{\text{in}}}}$) + Modulación Multiplicativa-Aditiva RAMA + Recuperación de Bases Cuantizadas NF4 en Cero Latencia.
