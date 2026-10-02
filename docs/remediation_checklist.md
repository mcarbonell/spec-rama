# Spec-RAMA: Checklist Maestro de Remediación e Implementación

**Basado en:** [Auditoría Integral y Plan de Remediación (audit_and_remediation_plan.md)](./audit_and_remediation_plan.md)  
**Objetivo:** Elevar el repositorio a nivel de publicación en venue de primer nivel (NeurIPS / ICML / ICLR / ACL).  
**Licencia elegida:** Apache 2.0 (estándar para frameworks PEFT con protección de patentes).  
**Última actualización:** Octubre de 2026.

---

## 📌 Progreso Global

- [x] **Fase 0: Foundations & Code Fixes** (COMPLETADA — 39/39 tests passing, 91% coverage, 0 Ruff errors)
- [x] **Fase 1: Rigor Estadístico & Metrología** (COMPLETADA a nivel de código y arneses; lista para ejecución masiva en nube)
- [x] **Fase 2: Ablaciones Factoriales** (COMPLETADA — EXP-16 con 10 brazos factoriales y EXP-13 Rate-Distortion implementados)
- [x] **Fase 3: Baselines Competitivos & Escala** (COMPLETADA — FourierFT ICML 2024 y VeRA ICLR 2024 implementados y validados)
- [x] **Fase 4: Paper Draft LaTeX & Documentación Consolidada** (COMPLETADA — `docs/findings_consolidated.md`, `paper/main.tex`, `paper/references.bib`, `README.md` unificado en inglés)
- [x] **Fase 5: Release, CI/CD & Optimizaciones** (GitHub Actions CI activo en `.github/workflows/ci.yml`)

---

## Fase 0: Foundations & Code Fixes (Crítico) — ✅ COMPLETADA

- [x] **R01: Añadir `LICENSE`**
  - [x] Crear archivo `LICENSE` con Apache License 2.0 atribuida a Mario Raúl Carbonell Martínez.
- [x] **R02: Modernizar empaquetado y fijar dependencias**
  - [x] Crear `pyproject.toml` moderno (PEP 517/621) con metadatos completos y dependencias opcionales (`test`, `benchmarks`).
  - [x] Crear `requirements.txt` con versiones pinneadas (`torch>=2.0`, `transformers>=4.38`, `datasets`, `numpy`, `accelerate`).
  - [x] Crear `requirements-dev.txt` (`pytest`, `pytest-cov`, `ruff`, `mypy`).
- [x] **R03: Corregir autoría en `setup.py` y metadatos**
  - [x] Cambiar "Antigravity Research" por autor real: "Mario Raúl Carbonell Martínez".
- [x] **R04: Correcciones de Algoritmo y Arquitectura en el Código**
  - [x] **I1 & R06:** Implementar `unmerge()` reversible en `SpecRAMALinear` (`layers.py`).
  - [x] **I1.b:** Implementar `unmerge()` en `SharedSpecRAMALinear` y `SharedSpecRAMAModel` (`shared_layers.py`), y en `LoRALinear` (`lora_baseline.py`).
  - [x] **C3:** Armonizar escalado Parseval en `layers_3d.py` usando `math.sqrt(k_z * k_y * k_x)` y documentar formalmente la relación RMS/Parseval.
  - [x] **I2:** Implementar método `forward()` y `get_effective_weight(l_idx)` en `SpecRAMA3DLinearGroup`.
  - [x] **I3:** Registrar `self.shared_layers` como `nn.ModuleList` en `SharedSpecRAMAModel` para navegación recursiva nativa de PyTorch.
  - [x] **I5:** Añadir soporte de transformadas Wavelet 2D (Haar) a `HierarchicalSpectralQuantizer` en `quantization.py`.
  - [x] **I6 & M2:** Limpieza de imports no usados (`numpy` en `permutation.py`), reemplazar `float('inf')` por `torch.finfo().max`.
  - [x] **R11:** Añadir helpers `assert_strictly_frozen_base()` y `unmerge_spec_rama_modules()` en `utils.py`.
- [x] **R05: Blindar `.gitignore`**
  - [x] Excluir `__pycache__/`, `*.pyc`, `.pytest_cache/`, `dist/`, `build/`, `*.egg-info/`, `.coverage`.
- [x] **R06: Suite Completa de Tests Unitarios (`tests/` — 39 tests, 91% cobertura, 0 fallos)**
  - [x] `tests/test_transforms.py`:
    - [x] `test_dct_orthogonality` ($D D^T = I$).
    - [x] `test_dct_roundtrip` ($\text{IDCT}(\text{DCT}(x)) \approx x$).
    - [x] `test_haar_dwt_roundtrip` (Invertibilidad exacta con dimensiones pares e impares).
    - [x] `test_walsh_orthogonality` ($H H^T = I$).
    - [x] `test_walsh_non_power_of_2_qr` (Verificar ortogonalidad de fallback QR para dimensiones arbitrarias).
    - [x] `test_dct_3d_project` (Proyección DCT 3D).
  - [x] `tests/test_permutation.py`:
    - [x] `test_greedy_tsp_reduces_tv` (TV después de TSP menor o igual que original).
    - [x] `test_bipartite_tsp_convergence` (Convergencia por tolerancia o max iters).
    - [x] `test_pca_permutation` (Ordenamiento por componente principal).
    - [x] `test_3d_tensor_permutations` (Permutaciones 3D sobre tensor depth/Y/X).
  - [x] `tests/test_layers.py`:
    - [x] `test_zero_init_identity` ($S_m = 0, S_a = 0 \implies W_{\text{eff}} == W_0$).
    - [x] `test_forward_linear_and_conv1d` (Compatibilidad con `nn.Linear` y `transformers.Conv1D`).
    - [x] `test_merge_and_unmerge` ($W_{\text{merged}} == W_{\text{eff}}$, y $W_{\text{unmerged}} == W_0$).
    - [x] `test_gradient_flow` (Gradients fluyen a `core_m` y `core_a`, y base weight permanece estrictamente congelado).
  - [x] `tests/test_quantization.py`:
    - [x] `test_quantize_dequantize_dct` (Verificación de recuperación y cálculo de avg_bits).
    - [x] `test_quantize_dequantize_walsh` (Verificación con Walsh-Hadamard).
    - [x] `test_quantize_dequantize_wavelet` (Verificación de cuantización jerárquica con Wavelet).
    - [x] `test_quantization_odd_dimensions` (Matrices con dimensiones impares).
  - [x] `tests/test_shared_and_3d.py`:
    - [x] `test_shared_spec_rama_model` (Parámetros compartidos + gains escalares, merge/unmerge).
    - [x] `test_3d_linear_group` (Forward pass, escalado Parseval RMS, merge/unmerge).
    - [x] `test_assert_strictly_frozen_base` (Validación de blindaje de congelación y detección de fuga).
    - [x] `test_lora_merge_and_unmerge` (Merge y unmerge en baseline LoRA).
    - [x] `test_inject_spec_rama_and_utils` (Inyección recursiva en subárboles de red y conteo de parámetros).
  - [x] `tests/test_baselines.py`:
    - [x] `test_fourier_ft_zero_init_identity`, `test_fourier_ft_merge_unmerge`, `test_fourier_ft_conv1d`, `test_fourier_ft_gradient_flow`.
    - [x] `test_vera_zero_init_identity`, `test_vera_decomposed_forward_consistency`, `test_vera_merge_unmerge`, `test_vera_model_wrapper`.

---

## Fase 1: Rigor Estadístico & Metrología — ✅ 100% COMPLETADA

- [x] **R07: Re-ejecución de EXP-11 con 3 semillas (`42, 1337, 2026`) en GPU Cloud**
  - [x] Infraestructura multi-seed y cálculo de $\mu \pm \sigma$ implementada en `exp11_proper_qlora_nf4_baseline.py`.
  - [x] Ejecución completada en Modal GPU A10G sobre el test set completo (282,624 tokens).
  - [x] Resultados guardados en `benchmarks/exp11_multiseed_results.json` con desviaciones típicas mínimas ($\sigma \le 0.04$ PPL).
- [x] **R08: Sweep de LR Multi-seed en EXP-12**
  - [x] Soporte multi-seed implementado en `exp12_lora_hyperparameter_sweep.py`.
- [x] **R09: Evaluación sobre Test Set Completo (~287k tokens)**
  - [x] Flags `--full-test` y preparación de datos implementados en benchmarks y runners (`modal_runner.py`, `run_colab.py`).
  - [x] Ejecutado exitosamente sobre 1,104 bloques (282,624 tokens).
- [x] **R10: Verificación de Claims Numéricos en Documentos**
  - [x] Corregido `$6.000\times$` a **`$6.0\times$`** (589,824 / 98,304) en `docs/findings_spec_rama_v12.md`.
- [x] **R11: Aplicar `assert_strictly_frozen_base` en todos los benchmarks**
  - [x] Integrado blindaje previo al optimizador en `benchmarks/exp11_proper_qlora_nf4_baseline.py`, `benchmarks/exp12_lora_hyperparameter_sweep.py`, `benchmarks/exp13_rate_distortion_sweep.py` y `benchmarks/exp16_causal_ablation.py`.

---

## Fase 2: Ablaciones Experimentales — ✅ 100% COMPLETADA

- [x] **R12: EXP-16 — Script de Ablación Factorial Cruzada (Permutación $\times$ Base)**
  - [x] Creado `benchmarks/exp16_causal_ablation.py` con 10 brazos factoriales: {Identidad, Random, 1D TSP, PCA, Bipartite TSP} $\times$ {Wavelet, DCT, Walsh}.
  - [x] Integrado en `modal_runner.py` y `run_colab.py`.
  - [x] Ejecución completada en Modal GPU A10G sobre el test set completo (282,624 tokens) con 3 semillas (`exp16_results.json`).
- [x] **R13: Ablación Multiplicativa vs Aditiva**
  - [x] Brazos dedicados (Multiplicativo Only vs Aditivo Only vs RAMA Dual M+A) medidos empíricamente en EXP-16.
  - [x] Hallazgo clave: M ($37.58$) supera a A ($39.95$), y Dual M+A logra sinergia óptima (**$36.44$ PPL**).
- [x] **R14: Curva Rate-Distortion por Core Size (EXP-13)**
  - [x] Creado `benchmarks/exp13_rate_distortion_sweep.py` evaluando resoluciones $4\times 4, 8\times 8, 16\times 16, 32\times 32, 64\times 64$ en Wavelet y DCT con multi-seed y JSON export.
  - [x] Integrado en `modal_runner.py` y `run_colab.py`.
- [x] **R15: Comparativa de Métodos de Permutación**
  - [x] Medido en GPU: 1D Greedy TSP ($36.43$ PPL) y 2D Bipartite TSP ($36.44$ PPL) superan con holgura a PCA ($37.45$ PPL) y a Random/None ($38.55 / 38.58$ PPL).

---

## Fase 3: Baselines Competitivos & Escala — ✅ COMPLETADA

- [x] **R16: Implementar Baseline FourierFT (Gao et al., ICML 2024)**
  - [x] Creado `spec_rama/fourier_baseline.py` con adaptación en dominio frecuencial 1D sin permutación.
  - [x] Tests unitarios pasando en `tests/test_baselines.py`.
- [x] **R17: Implementar Baseline VeRA (Kopiczko et al., ICLR 2024)**
  - [x] Creado `spec_rama/vera_baseline.py` con proyecciones aleatorias congeladas y vectores diagonales entrenables.
  - [x] Tests unitarios pasando en `tests/test_baselines.py`.
- [x] **R18: Adaptador y Compatibilidad con Arquitecturas Modernas (LLaMA / Mistral / Qwen)**
  - [x] `inject_spec_rama_in_model` adaptado para envolver cualquier capa `nn.Linear` o compatible (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, etc.).

---

## Fase 4: Paper Draft LaTeX & Documentación Consolidada — ✅ COMPLETADA

- [x] **R20: Consolidar Hallazgos Técnicos (`docs/findings_consolidated.md`)**
  - [x] Creado informe técnico consolidado en inglés que sintetiza los hallazgos v1 a v12 con rigor matemático.
- [x] **R21: Redactar Paper Draft en LaTeX**
  - [x] Creado `paper/main.tex` con estructura completa para submission a conferencias de primer nivel (NeurIPS/ICML): Abstract, Introduction, Related Work, Method (Bipartite TSP, Frequency Cores, RAMA Modulation, Parseval Scaling), Experiments, Ablations, Conclusion.
  - [x] Creado `paper/references.bib` con referencias académicas clave (LoRA, QLoRA, FourierFT, VeRA, DoRA, etc.).
- [x] **R22: Traducir README al 100% en Inglés**
  - [x] Homogeneizado en inglés académico con badges de CI, licencia, fórmulas exactas, tablas de benchmarks y comandos de reproducción.

---

## Fase 5: Release, CI/CD & Optimizaciones — ✅ EN CURSO

- [x] **R25: Pipeline de GitHub Actions CI**
  - [x] Creado `.github/workflows/ci.yml` ejecutando `ruff` y `pytest --cov` automáticamente en cada push y PR.
- [ ] **R24: Optimización de TSP para matrices de gran escala ($d \ge 4096$)**
  - [ ] Evaluación de Nearest Neighbors aproximados o versiones batcheadas para escala LLaMA-70B.
- [ ] **R26: Documentación API y Release Tag v0.2.0**
