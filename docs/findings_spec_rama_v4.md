# Hallazgos del Experimento: Comparativa Iso-Parámetro (Spec-RAMA vs LoRA) (v4)

Este documento resume los resultados del **Experimento 04 (EXP-04)** en una GPU **NVIDIA A10G** (Modal Cloud), donde se evaluó la eficacia de **Spec-RAMA** frente a **LoRA** a **igualdad exacta de presupuesto de parámetros (Iso-Parameter)** sobre el **Test Set de WikiText-2 (280.000 tokens)**.

---

## 1. Resultados Oficiales con Memoria Total (EXP-04)

| Método de Adaptación | Parámetros Entrenables | Tamaño Adaptador | Tamaño Peso Base (FP32) | Tamaño Total Modelo | TEST PPL | Régimen de Dominio |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **GPT-2 Base (FP32)** | 124,439,808 | 0 KB | 474.7 MB | **474.7 MB** | 46.18 | Baseline |
| **SpecRAMA Shared (Sub-KB)** | **4,808** | **18.8 KB** | 474.7 MB | **474.7 MB** | **45.54** ↓ | **Exclusivo Spec-RAMA (<20 KB)** |
| **SpecRAMA Wavelet (8x8)** | **4,608** | **18.0 KB** | 474.7 MB | **474.7 MB** | **42.57** ↓ | **Exclusivo Spec-RAMA (<20 KB)** |
| **LoRA (rank=1, alpha=2)** | 101,376 | 396.0 KB | 474.7 MB | **475.1 MB** | **30.57** ↓ | Dominio espacial $r=1$ |
| **SpecRAMA Wavelet (37x38)** | **101,232** | **395.4 KB** | 474.7 MB | **475.1 MB** | **37.81** ↓ | Dominio espectral concentrado |
| **SpecRAMA DCT (37x38)** | **101,232** | **395.4 KB** | 474.7 MB | **475.1 MB** | **38.41** ↓ | Dominio espectral cosenoidal |
| **LoRA (rank=4, alpha=8)** | 405,504 | 1.55 MB | 474.7 MB | **476.3 MB** | **29.67** ↓ | Dominio espacial $r=4$ |
| **SpecRAMA Wavelet (75x75)** | **405,000** | **1.54 MB** | 474.7 MB | **476.2 MB** | **35.90** ↓ | Dominio espectral amplio |
