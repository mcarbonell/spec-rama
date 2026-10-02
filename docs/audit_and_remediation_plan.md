# Auditoría Integral del Repositorio `spec-rama` y Plan de Remediación

**Autor de la auditoría:** Antigravity AI  
**Fecha:** 2 de Octubre de 2026  
**Repositorio:** `mcarbonell/spec-rama`  
**Autor del algoritmo:** Mario Raúl Carbonell Martínez  
**Objetivo:** Evaluar la madurez del repositorio para la publicación de un paper académico.

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Auditoría del Código Fuente](#2-auditoría-del-código-fuente)
3. [Auditoría Matemática](#3-auditoría-matemática)
4. [Auditoría de los Experimentos](#4-auditoría-de-los-experimentos)
5. [Auditoría de la Documentación](#5-auditoría-de-la-documentación)
6. [Auditoría de Ingeniería de Software](#6-auditoría-de-ingeniería-de-software)
7. [Evaluación de Paper-Readiness](#7-evaluación-de-paper-readiness)
8. [Plan de Remediación Priorizado](#8-plan-de-remediación-priorizado)

---

## 1. Resumen Ejecutivo

### Veredicto General: Prometedor con Gaps Críticos para Publicación

El repositorio implementa una idea algorítmica **genuinamente original y bien diferenciada** que combina tres componentes novedosos:

1. **Permutación Greedy-TSP 2D** de pesos para maximizar suavidad espacial
2. **Adaptación en dominio espectral 2D** (DCT / Walsh-Hadamard / Wavelet Haar)
3. **Modulación RAMA multiplicativa-aditiva** con escalado Parseval

La implementación core es **correcta**, **limpia** y **funcional**. Los resultados experimentales en GPT-2 Small son **reproducibles y competitivos**. Sin embargo, para publicación en una venue de primer nivel (NeurIPS, ICML, ICLR, ACL), existen gaps significativos que deben cerrarse.

### Scorecard de Madurez

| Dimensión | Estado | Nota |
| :--- | :---: | :--- |
| **Corrección Matemática del Algoritmo** | SÓLIDO | Formulación coherente, transformadas correctas |
| **Calidad del Código** | BUENO con gaps | Funcional pero sin tests, sin tipado estricto |
| **Reproducibilidad Experimental** | PARCIAL | Resultados JSON serializados pero semilla única, test set parcial |
| **Rigor Estadístico** | INSUFICIENTE | Sin intervalos de confianza, sin múltiples semillas |
| **Escala de Evaluación** | INSUFICIENTE | Solo GPT-2 Small (124M), solo WikiText-2 PPL |
| **Comparación con Baselines** | PARCIAL | LoRA incluido; faltan FourierFT, VeRA, NOLA |
| **Documentación Técnica** | ABUNDANTE pero dispersa | 12 docs de hallazgos sin consolidación |
| **Paper-Readiness** | NO LISTO | Sin paper draft, sin ablaciones formales, sin escala |

---

## 2. Auditoría del Código Fuente

### 2.1. Estructura del Paquete

```
spec_rama/
├── __init__.py          ← Exports limpios y completos
├── layers.py            ← Core layer (174 LOC)
├── layers_3d.py         ← 3D tensor adaptation (119 LOC)
├── shared_layers.py     ← Cross-layer sharing (138 LOC)
├── permutation.py       ← TSP + Bipartite + PCA (171 LOC)
├── transforms.py        ← DCT, Walsh, Haar DWT/IDWT (134 LOC)
├── quantization.py      ← Hierarchical spectral quantizer (130 LOC)
├── lora_baseline.py     ← LoRA reference implementation (108 LOC)
└── utils.py             ← Injection utilities (74 LOC)
```

**Total: ~1,048 LOC** — Codebase compacto y enfocado.

### 2.2. Hallazgos Positivos

| Hallazgo | Archivo | Detalle |
| :--- | :--- | :--- |
| Soporte Conv1D (GPT-2) correcto | `layers.py:28-38` | Maneja correctamente la transposición de pesos de HuggingFace Conv1D |
| Inicialización a cero de cores | `layers.py:82,87` | Garantiza que el modelo arranca idéntico al base antes de entrenar |
| Merge lossless | `layers.py:166-173` | Fusión in-place que preserva la corrección numérica |
| Permutación bipartita convergente | `permutation.py:76-111` | Criterio de convergencia por TV relativa, elegante |
| DCT-II ortogonal | `transforms.py:5-17` | Implementación correcta de DCT tipo II normalizada |
| Haar DWT/IDWT simétrica | `transforms.py:41-96` | Manejar padding odd es correcto |
| Walsh con QR para N no potencia de 2 | `transforms.py:25-30` | Fallback robusto vía re-ortogonalización QR |

### 2.3. Problemas Detectados

#### P1 — Críticos para Publicación

| # | Problema | Archivo | Línea | Impacto |
| :--- | :--- | :--- | :---: | :--- |
| C1 | **Sin tests unitarios** | — | — | No hay ningún directorio `tests/` ni un solo `test_*.py`. Imposible verificar automáticamente la corrección de transformadas, permutaciones, merge, cuantización, etc. |
| C2 | **Sin `requirements.txt` / `pyproject.toml`** | — | — | `setup.py` lista solo `torch>=2.0` y `numpy`, pero los benchmarks requieren `transformers`, `datasets`, `accelerate`. No hay pinning de versiones. |
| C3 | **Escalado Parseval inconsistente entre componentes** | `layers.py:138,142,147` vs `layers_3d.py:100-101` | — | En `layers.py`, el escalado multiplicativo divide por `sqrt(k_out * k_in) * w_std` pero el aditivo solo por `sqrt(k_out * k_in)`. En `layers_3d.py`, se divide por `num_core_params` (= `k_z * k_y * k_x`) en vez de `sqrt(...)`. No hay justificación documental de esta divergencia. |
| C4 | **`setup.py` atribuye autoría a "Antigravity Research"** | `setup.py:8` | 8 | Para un paper debe ser el autor real: Mario Raúl Carbonell Martínez |

#### P2 — Importantes

| # | Problema | Archivo | Línea | Impacto |
| :--- | :--- | :--- | :---: | :--- |
| I1 | **No hay `unmerge()` en `SpecRAMALinear`** | `layers.py` | 166-173 | `merge()` guarda `original_weight` pero no hay función inversa. Si se necesita re-entrenar tras merge, el estado es irrecuperable. |
| I2 | **`SpecRAMA3DLinearGroup` sin forward method** | `layers_3d.py` | — | La clase computa pesos efectivos pero carece de un `forward()`. No es usable como módulo de red directamente. |
| I3 | **`SharedSpecRAMAModel` no registra sub-layers como `nn.ModuleList`** | `shared_layers.py:110` | 110 | `self.shared_layers` es una lista Python plana (`List[]`), no un `nn.ModuleList`. Los gains `gamma_m/gamma_a` de las subcapas pueden no aparecer en `model.parameters()` si no son accesibles por el módulo tree de PyTorch. **NOTA: Esto funciona porque las subcapas se registran vía `setattr` en el modelo original, pero es frágil.** |
| I4 | **Greedy TSP es O(N^2) en memoria** | `permutation.py:18` | 18 | `torch.cdist` computa la matriz de distancias completa `(N, N)` en memoria. Para capas con `out_features=4096` (LLaMA), esto requiere ~128 MB por capa. No escalará a modelos grandes sin optimización. |
| I5 | **`quantization.py` no soporta Wavelet** | `quantization.py:42-49` | 42-49 | El `HierarchicalSpectralQuantizer` solo soporta DCT y Walsh, pero Wavelet es la transformada ganadora en todos los benchmarks. |
| I6 | **Import de `numpy` innecesario** | `permutation.py:2`, `quantization.py:3` | — | Se importa `numpy` pero no se usa en el cuerpo del módulo. |

#### P3 — Menores / Estilísticos

| # | Problema | Archivo | Detalle |
| :--- | :--- | :--- | :--- |
| M1 | Type hints incompletos | Varios | Funciones como `compute_greedy_tsp_1d` no documentan tipo de retorno en docstring |
| M2 | Hardcoded `float('inf')` | `permutation.py:27` | Podría usar `torch.finfo(dist_matrix.dtype).max` |
| M3 | `__pycache__` en repo | `.gitignore` | Directorio `__pycache__` raíz visible, sugiere que .gitignore no cubre todos los paths |
| M4 | Mezcla español/inglés en docs | `docs/*` | Los findings están en español, el README mezcla ambos. Para un paper, todo debe estar en inglés. |

---

## 3. Auditoría Matemática

### 3.1. Formulación del Algoritmo — Correcta

La pipeline matemática de Spec-RAMA es coherente:

$$W_{\text{eff}} = W_0 \cdot \left(1 + \frac{\alpha_m}{\sqrt{k_\text{out} \cdot k_\text{in}} \cdot \text{std}(W_0)} \cdot \mathcal{T}^{-1}(S_m)\right) + \frac{\alpha_a}{\sqrt{k_\text{out} \cdot k_\text{in}}} \cdot \mathcal{T}^{-1}(S_a)$$

donde:
- $\mathcal{T}^{-1}$ es la transformada inversa (IDWT-2D / IDCT-2D) en coordenadas permutadas
- $S_m, S_a \in \mathbb{R}^{k_\text{out} \times k_\text{in}}$ son los cores entrenables inicializados a cero
- La permutación $\pi_\text{row}, \pi_\text{col}$ maximiza la concentración de energía de baja frecuencia

### 3.2. Propiedades Verificadas

| Propiedad | Estado | Verificación |
| :--- | :---: | :--- |
| **Identidad en inicialización** ($S = 0 \Rightarrow W_\text{eff} = W_0$) | OK | Cores inicializados a cero, transform de cero = cero |
| **Ortogonalidad de DCT-II** | OK | $D \cdot D^T = I$ por construcción con factores $\sqrt{1/N}$, $\sqrt{2/N}$ |
| **Inversibilidad de Haar DWT** | OK | DWT + IDWT = Identity verificado algebraicamente en el código |
| **Lossless merge** | OK | Se computa $W_\text{eff}$ completo y se copia in-place. Sin pérdida numérica. |
| **Parseval energy scaling** | PARCIAL | Correcto para la componente multiplicativa. El aditivo usa $1/\sqrt{k}$ sin dividir por $\text{std}(W_0)$, lo cual es una decisión de diseño razonable (la componente aditiva no es relativa al peso base). |

### 3.3. Observaciones Matemáticas para el Paper

1. **El escalado multiplicativo divide por `std(W_0)`** — Esto normaliza la perturbación relativa al rango dinámico del peso base. Es análogo a la normalización de batch pero en el espacio de pesos. **Merece una justificación formal en el paper.**

2. **La permutación TSP no tiene garantía de optimalidad** — El greedy TSP es una heurística O(N^2) que no garantiza la permutación óptima para maximizar concentración espectral. Sin embargo, el criterio de convergencia por Total Variation (TV) en el bipartite alternante sí converge a un punto fijo local. **Esto debe discutirse como limitación.**

3. **La afirmación de "base universal" (conexión con Kaushik et al.)** es una hipótesis fascinante pero **no demostrada** — El paper propuesto debe distinguir claramente entre:
   - Lo que está **demostrado empíricamente** (concentración de energía, recuperación de PPL)
   - Lo que es **conjetura** (universalidad de la base TSP a través de tareas)

---

## 4. Auditoría de los Experimentos

### 4.1. Inventario Experimental

| Exp | Pregunta | Escala | Semillas | Status |
| :--- | :--- | :--- | :---: | :--- |
| EXP-01 | Sanity check sintético | Dummy data | 1 | Válido como verificación |
| EXP-02 | Head-to-head WikiText-2 | GPT-2, 280K tokens | 1 | Válido |
| EXP-03 | Scaling law de core size | GPT-2, 280K tokens | 1 | Válido, resultado clave |
| EXP-04 | Iso-parameter benchmark | GPT-2, 280K tokens | 1 | Válido |
| EXP-05 a 07 | Cuantización RTN + recovery | GPT-2, 280K tokens | 1 | Superseded por EXP-11 |
| EXP-08 | NF4 breakthrough | GPT-2, 280K tokens | 1 | Superseded por EXP-11 |
| EXP-09 | Heterogeneous quant | GPT-2, 280K tokens | 1 | Válido |
| EXP-10 | 6-arm control | GPT-2, 280K tokens | 1 | Importante pero sin multi-seed |
| **EXP-11** | **Benchmark definitivo NF4** | **GPT-2, 25.6K tokens** | **1** | **Flagship pero con gaps** |
| **EXP-12** | **Sweep de LR en LoRA** | **GPT-2, 25.6K tokens** | **1** | **Necesario pero incompleto** |

### 4.2. Gaps Críticos para Publicación

#### G1 — Sin Múltiples Semillas ni Intervalos de Confianza (CRÍTICO)

**Todos los resultados reportados son de una única semilla (`seed=42`).**

Un reviewer rechazará inmediatamente cualquier comparación sin al menos:
- 3 semillas (preferiblemente 5)
- Media +/- desviación estándar reportada
- Test estadístico (Wilcoxon signed-rank o paired t-test) para las claims de superioridad

#### G2 — Evaluación sobre Test Set Parcial (25,600 tokens) (CRÍTICO)

EXP-11 y EXP-12 evalúan sobre **100 bloques = 25,600 tokens**, que es ~9% del test set completo de WikiText-2 (~287K tokens). Los EXPs 02-04 sí evaluaron sobre los 280K tokens completos, pero los resultados finales reportados en el README son de 25.6K.

Para publicación, se debe evaluar sobre el **test set completo** y verificar que los resultados son consistentes.

#### G3 — Solo GPT-2 Small (124M) (CRÍTICO)

Ningún resultado ha sido obtenido en modelos de escala relevante:
- LLaMA-2/3 (7B/8B)
- Mistral (7B)
- Qwen-2.5 (7B)
- Phi-3 (3.8B)

Un paper de PEFT moderno **requiere** al menos un modelo >1B para ser tomado en serio. Idealmente >7B con benchmarks downstream (MMLU, GSM8K, HumanEval).

#### G4 — Solo Language Modeling PPL, sin Tasks Downstream (CRÍTICO)

La perplejidad sobre WikiText-2 es necesaria pero insuficiente. Se necesitan:
- **GLUE** (SST-2, MRPC, RTE) para clasificación
- **MMLU** para razonamiento general
- **GSM8K** para razonamiento matemático
- O al menos **SuperGLUE** como alternativa

#### G5 — Faltan Baselines Académicos Clave (IMPORTANTE)

| Baseline | Status | Prioridad |
| :--- | :---: | :---: |
| LoRA (Hu et al., 2021) | Implementado | — |
| QLoRA (Dettmers et al., 2023) | Implementado (manual NF4) | — |
| **FourierFT (Gao et al., ICML 2024)** | No implementado | **ALTA** — Competidor directo |
| **VeRA (Kopiczko et al., ICLR 2024)** | No implementado | **ALTA** — Competidor en régimen sub-KB |
| **NOLA (Su et al., 2023)** | No implementado | MEDIA |
| **DoRA (Liu et al., 2024)** | No implementado | MEDIA |
| AdaLoRA | No implementado | BAJA |

#### G6 — Sin Ablación Formal de Componentes (IMPORTANTE)

No existe un experimento que aísle la contribución de cada componente:

| Componente | Pregunta | Status |
| :--- | :--- | :---: |
| Permutación TSP | Cuánto aporta vs. identidad? | No aislado |
| Wavelet vs DCT vs Walsh | Cuál es mejor y por cuánto? | Parcial (EXP-01, EXP-03) |
| Componente multiplicativa | Es necesaria? | No aislado |
| Componente aditiva | Es necesaria? | No aislado |
| Bipartite TSP vs single-pass TSP | Cuánto mejora? | No aislado |
| Escalado Parseval | vs. escalado LoRA clásico? | No aislado |

> [!IMPORTANT]
> La **EXP-16** (ablación causal) propuesta en `proposed_experiments_rate_distortion.md` es exactamente lo necesario. Debe ejecutarse con prioridad máxima.

#### G7 — Cuantización NF4 es Simulada, no Nativa (NOTA)

La cuantización NF4 en EXP-11 es una **simulación software** (`quantize_blockwise_nf`). Los pesos se dequantizan a FP32 para compute. Esto es válido para medir la calidad de recuperación pero:
- No mide ahorro real de VRAM
- No es comparable a la implementación de `bitsandbytes` (que almacena en int4 real)
- Debe documentarse claramente como "simulated NF4 quantization"

### 4.3. Experimentos Bien Diseñados

A pesar de los gaps, varios aspectos son positivos:

- **Protocolo de congelación estricta** implementada en EXP-11/12
- **Serialización JSON** de todos los resultados
- **Evaluación non-overlapping blocks** (sin data leakage)
- **Evolución transparente** — Los 12 docs de findings muestran iteración honesta, incluyendo el error de embeddings descongelados y su corrección
- **Tablas comparativas exhaustivas** con métricas múltiples (PPL, bpt, delta-bpt)

---

## 5. Auditoría de la Documentación

### 5.1. Estado Actual

| Documento | Propósito | Calidad | Idioma |
| :--- | :--- | :---: | :---: |
| `README.md` | Presentación general | Excelente | EN/ES mezclado |
| `findings_v1.md` a `v12.md` | Log de hallazgos iterativo | Abundante pero disperso | ES |
| `proposed_experiments.md` | Roadmap de nuevos EXPs | Excelente | ES |
| `proposed_experiments_rate_distortion.md` | Propuesta de EXP 13-17 | Excelente, muy riguroso | ES |
| `setup.py` | Packaging | Incompleto | — |

### 5.2. Problemas

| # | Problema | Detalle |
| :--- | :--- | :--- |
| D1 | **12 documentos de findings sin consolidación** | Los `findings_v1` a `v12` son un log incremental. Para el paper y el repo público, se necesita un documento consolidado que resuma el estado final. |
| D2 | **Documentación en español** | Los findings están en español. El README mezcla idiomas. Para publicación, todo debe estar en inglés. |
| D3 | **No hay docstring en formato estándar (NumPy/Google)** | Las funciones tienen docstrings pero sin formato estandarizado con Parameters/Returns/Examples. |
| D4 | **No hay CHANGELOG** | Sin historial de cambios versionado. |
| D5 | **No hay CONTRIBUTING.md** | Sin guía de contribución. |
| D6 | **No hay LICENSE** | Sin licencia explícita. CRÍTICO para open source. |
| D7 | **No hay API reference docs** | Sin documentación generada automáticamente (Sphinx/MkDocs). |

---

## 6. Auditoría de Ingeniería de Software

### 6.1. Aspectos Faltantes

| Categoría | Estado | Detalle |
| :--- | :---: | :--- |
| **Tests unitarios** | Ausente | Cero tests. Sin `pytest`, sin CI. |
| **Tests de integración** | Ausente | Ningún test end-to-end automatizado |
| **Type checking** | Ausente | Sin `mypy`, sin `py.typed` marker |
| **Linting** | Ausente | Sin `ruff`, `flake8`, ni `black` |
| **CI/CD** | Ausente | Sin GitHub Actions ni pre-commit hooks |
| **Versionado semántico** | Parcial | `0.1.0` en setup.py pero sin tags ni releases |
| **Pinning de dependencias** | Ausente | Sin `requirements.txt` con versiones pinneadas |
| **Docker** | Ausente | Sin Dockerfile para reproducibilidad |
| **Reproducibilidad** | Parcial | Semillas fijadas pero sin lock de versiones |

### 6.2. Cobertura de Tests Necesaria

Para publicación, se necesitan como mínimo estos tests:

```python
# tests/test_transforms.py
# - test_dct_orthogonality()       # D @ D.T == I
# - test_dct_roundtrip()           # IDCT(DCT(x)) == x
# - test_haar_dwt_roundtrip()      # IDWT(DWT(x)) == x
# - test_walsh_orthogonality()     # H @ H.T == I
# - test_walsh_non_power_of_2()    # QR fallback works

# tests/test_permutation.py
# - test_tsp_reduces_tv()          # TV(W_perm) < TV(W_orig)
# - test_bipartite_convergence()   # TV decreases monotonically
# - test_identity_permutation()    # method="none" returns arange

# tests/test_layers.py
# - test_zero_init_identity()      # Forward matches base layer at init
# - test_merge_preserves_output()  # Merged output == unmerged output
# - test_conv1d_support()          # Works with HF Conv1D

# tests/test_quantization.py
# - test_quantize_dequantize_roundtrip()
```

---

## 7. Evaluación de Paper-Readiness

### 7.1. Contribución Novel — FUERTE

El paper tiene una **contribución genuinamente original y diferenciada**:

1. **Permutación TSP 2D para concentración espectral** — No existe en la literatura de PEFT. FourierFT opera en 1D sin reordenamiento. VeRA usa proyecciones aleatorias.
2. **Adaptación Wavelet 2D multiscala** — Extiende FourierFT a 2D con descomposición multiescala.
3. **Escalado Parseval** — Normalización por energía espectral que estabiliza el entrenamiento.
4. **Merge exacto sin latencia** — Compartido con LoRA pero no trivial en dominio espectral.

### 7.2. Gaps para Publicación — Resumen

| Requisito | Estado | Prioridad |
| :--- | :---: | :---: |
| Múltiples semillas (>=3) | Falta | **P0** |
| Full test set eval | Falta | **P0** |
| Modelo >1B | Falta | **P0** |
| Ablación de componentes | Falta | **P0** |
| Baselines FourierFT + VeRA | Falta | **P1** |
| Tasks downstream (GLUE/MMLU) | Falta | **P1** |
| Tests unitarios | Falta | **P1** |
| Paper draft (LaTeX) | Falta | **P1** |
| Licencia open source | Falta | **P1** |
| Documentación en inglés | Falta | **P2** |

### 7.3. Venue Recomendada

| Venue | Prob. sin remediación | Prob. con remediación completa |
| :--- | :---: | :---: |
| **NeurIPS / ICML Main** | ~5% | 30-40% |
| **ICLR Main** | ~5% | 30-40% |
| **NeurIPS / ICML Workshop** | ~30% | 70-80% |
| **EMNLP / ACL Findings** | ~15% | 50-60% |
| **AAAI** | ~20% | 50-60% |
| **arXiv Technical Report** | ~90% | 95% |

> [!TIP]
> **Estrategia recomendada:** Publicar primero un **arXiv preprint** robusto con los resultados actuales mejorados (multi-seed, full test set, ablación), y simultáneamente ejecutar los experimentos a escala para una submisión a ICML/NeurIPS Workshop o EMNLP Findings.

---

## 8. Plan de Remediación Priorizado

### Fase 0: Foundations (1-2 días) — CRÍTICO

| # | Tarea | Esfuerzo | Detalle |
| :--- | :--- | :---: | :--- |
| R01 | **Añadir LICENSE** (Apache 2.0 o MIT) | 5 min | Crear `LICENSE` en raíz |
| R02 | **Crear `requirements.txt`** con versiones pinneadas | 15 min | `torch>=2.0`, `transformers>=4.38`, `datasets`, `numpy`, `accelerate` |
| R03 | **Corregir autoría en `setup.py`** | 5 min | Cambiar "Antigravity Research" por autor real |
| R04 | **Crear directorio `tests/`** con tests de smoke | 2-3h | Tests de ortogonalidad, roundtrip, zero-init, merge |
| R05 | **Añadir `.gitignore` completo** | 5 min | Asegurar que `__pycache__/`, `*.pyc`, `.eggs/` están excluidos |
| R06 | **Implementar `unmerge()` en `SpecRAMALinear`** | 30 min | Restaurar `original_weight` guardado |

### Fase 1: Rigor Estadístico (2-3 días) — CRÍTICO

| # | Tarea | Esfuerzo | Detalle |
| :--- | :--- | :---: | :--- |
| R07 | **Re-ejecutar EXP-11 con 3 semillas** (`42, 1337, 2026`) | 3-6h GPU | Reportar media +/- std |
| R08 | **Re-ejecutar EXP-12 con 3 semillas** | 3-6h GPU | Sweep completo por semilla |
| R09 | **Evaluación full test set** (~287K tokens) | 1h GPU | Usar `--full-test` flag existente |
| R10 | **Verificar claim "6.0x reduction"** | 30 min | 589,824 / 98,304 = 6.0x — Correcto |
| R11 | **Ejecutar `assert_strictly_frozen_base()`** en todos los EXPs | 1h | Función ya propuesta en `proposed_experiments.md` |

### Fase 2: Ablaciones (3-5 días) — CRÍTICO

| # | Tarea | Esfuerzo | Detalle |
| :--- | :--- | :---: | :--- |
| R12 | **EXP-16: Ablación factorial Permutación x Base** | 1 día GPU | Identidad/TSP/Random x Wavelet/DCT/Random (6 brazos) |
| R13 | **Ablación multiplicativa vs aditiva** | 0.5 día | 4 brazos: solo-M, solo-A, M+A, ninguno |
| R14 | **Ablación de core size** (curva rate-distortion EXP-13) | 1 día | 4x4, 8x8, 16x16, 32x32, 64x64 con 3 semillas |
| R15 | **Ablación bipartite TSP vs single-pass TSP vs PCA** | 0.5 día | 3 métodos de permutación |

### Fase 3: Baselines y Escala (1-2 semanas) — IMPORTANTE

| # | Tarea | Esfuerzo | Detalle |
| :--- | :--- | :---: | :--- |
| R16 | **Implementar baseline FourierFT** | 1-2 días | El competidor más directo |
| R17 | **Implementar baseline VeRA** | 1-2 días | Competidor en régimen ultra-compacto |
| R18 | **Ejecutar en LLaMA-3-8B** (Colab A100 / cloud) | 2-3 días | Evaluar en MMLU, GSM8K |
| R19 | **Ejecutar en Phi-3-mini (3.8B)** | 1-2 días | Alternativa más factible en Colab T4 |
| R20 | **Evaluar en GLUE tasks** (SST-2, MRPC, RTE) | 1-2 días | Classification downstream |

### Fase 4: Paper y Documentación (1-2 semanas) — IMPORTANTE

| # | Tarea | Esfuerzo | Detalle |
| :--- | :--- | :---: | :--- |
| R21 | **Escribir paper draft en LaTeX** | 1 semana | Introduction, Background, Method, Experiments, Analysis, Conclusion |
| R22 | **Consolidar docs en un único findings doc** | 2-3h | Unificar v1-v12 en un summary en inglés |
| R23 | **Traducir README completo a inglés** | 2h | Eliminar secciones en español |
| R24 | **Crear figuras de calidad paper** | 1-2 días | Pareto curves (params vs PPL), ablation bar charts, architecture diagram |
| R25 | **Generar API docs con Sphinx/MkDocs** | 0.5 día | Documentación auto-generada |

### Fase 5: Polish y Release (2-3 días) — DESEABLE

| # | Tarea | Esfuerzo | Detalle |
| :--- | :--- | :---: | :--- |
| R26 | **Optimizar TSP para modelos grandes** | 1-2 días | Implementar approximate nearest neighbor o batched TSP |
| R27 | **Añadir soporte Wavelet al `HierarchicalSpectralQuantizer`** | 0.5 día | Completar el módulo de cuantización |
| R28 | **Crear `pyproject.toml`** moderno | 1h | Reemplazar `setup.py` por packaging moderno |
| R29 | **Añadir GitHub Actions CI** | 2h | Linting + tests en cada PR |
| R30 | **Crear release tag v0.2.0** | 30 min | Con changelog y assets |

---

## Apéndice A: Resumen de Claims y su Soporte Evidencial

| Claim del README/Docs | Soporte Actual | Confianza |
| :--- | :--- | :---: |
| "37.96 PPL con 384 KB adapter" | EXP-11, 1 semilla, 25.6K tokens | Medio |
| "6.0x fewer parameters than LoRA" | 98,304 / 589,824 = 6.0x | Alto |
| "Near-adapted FP32 (<0.08 bpt)" | 5.246 - 5.168 = 0.078 bpt | Medio (1 semilla) |
| "Parseval scaling makes training stable across LR" | Claim implícito, no hay sweep de LR para SpecRAMA | No soportado |
| "Over 92% energy in LL sub-band" | Visualización en img, sin medición cuantitativa reportada | Medio |
| "Zero-latency merge" | Merged PPL = Tuned PPL en EXP-01 a EXP-11 | Alto |
| "Wavelet consistently outperforms DCT" | EXP-01 (2.67 vs 2.74) y EXP-03 (38.35 vs 38.96) | Alto |
| "Beats native FP32 at 3.55 bits" | 45.95 < 46.18 PPL, 1 semilla, 25.6K tokens | Medio |

---

## Apéndice B: Fórmula de Escalado Parseval — Análisis Detallado

La fórmula actual en el código es:

```python
# Multiplicativo (layers.py:142)
scaling_m = alpha_m / (sqrt(k_out * k_in) * w_std)

# Aditivo (layers.py:147)
scaling_a = alpha_a / sqrt(k_out * k_in)
```

**Justificación matemática:**
- Por el Teorema de Parseval, si T es ortogonal: ||S||_F = ||T^-1(S)||_F
- El core S (k_out x k_in) se inyecta en un espacio padded de (N_out x N_in)
- La norma del update reconstruido escala como sqrt(k_out * k_in) * ||S||_rms
- Dividir por sqrt(k_out * k_in) normaliza a unidad RMS independientemente del core size

**Nota:** Esta justificación solo es exacta para DCT (ortogonal). Para Wavelet, la Haar 1-level no es una transformada ortogonal completa sobre todo R^(N x N) sino sobre el subespacio LL. La diferencia es pequeña en la práctica pero debe documentarse.

---

## Apéndice C: Competencia en la Literatura y Posicionamiento

### Trabajos Directamente Relevantes (Deben Citarse y Compararse)

| Paper | Venue | Relevancia | Diferencia clave con Spec-RAMA |
| :--- | :--- | :--- | :--- |
| **FourierFT** (Gao et al.) | ICML 2024 | Muy alta | 1D DFT sin permutación vs. 2D Wavelet+TSP |
| **VeRA** (Kopiczko et al.) | ICLR 2024 | Muy alta | Random frozen projections vs. deterministic TSP |
| **LoRA** (Hu et al.) | ICLR 2022 | Muy alta | Spatial low-rank vs. spectral compact |
| **QLoRA** (Dettmers et al.) | NeurIPS 2023 | Muy alta | Focus on quantization vs. spectral adaptation |
| **DoRA** (Liu et al.) | 2024 | Alta | Magnitude-direction decomposition |
| **NOLA** (Su et al.) | 2023 | Alta | Frozen random bases |
| **Universal Weight Subspace** (Kaushik et al.) | arXiv 2025 | Alta | Theoretical grounding for shared spectral bases |
| **QuIP# / SpinQuant** | 2024 | Media | Orthogonal rotations for quantization |

---

> [!CAUTION]
> **Las claims de "6,000x reduction" que aparecen en `findings_v12.md` son un error tipográfico o de redacción.** El ratio real es **6.0x** (589,824 / 98,304 = 6.0). Esto debe corregirse antes de publicar.
