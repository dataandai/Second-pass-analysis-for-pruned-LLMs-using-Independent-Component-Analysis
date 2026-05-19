# ICA-Based Second-Pass Pruning Scan for LLMs

> GitHub rendering note: displayed equations use fenced `math` blocks rather than dollar-delimited display math. This is the most reliable format for README rendering on GitHub.


This repository/notebook implements a minimal research prototype for scanning a pretrained or already-pruned causal language model for **additional pruning opportunities** using **Independent Component Analysis (ICA)** on layer activations.

The method is intentionally simple:

- no SAE,
- no sparse autoencoder,
- no LoRA,
- no fine-tuning,
- no dashboard,
- no claim that the discovered components are human-interpretable concepts.

The goal is practical compression diagnostics: identify weight regions that appear weakly connected to statistically salient activation components and may therefore be candidates for a second pruning pass.

---

## 1. Hypothesis

A model that has already been compressed by magnitude pruning, SVD/low-rank pruning, Wanda, or SparseGPT may still contain residual redundancy. The remaining redundancy may not be visible as simply:

- small individual weights,
- low-energy singular directions,
- low local activation magnitude,
- or layer-output reconstruction error.

The hypothesis is:

> Layer activations may still contain statistically separable source-like components. If these components are estimated with ICA, they can be projected back into the corresponding weight matrix to build a **weight-level protection map**. Weights that are small and weakly connected to protected ICA directions are plausible candidates for additional pruning.

The proposed scan is:


```math
\text{activation samples}
\rightarrow
\text{ICA components}
\rightarrow
\text{weight protection map}
\rightarrow
\text{extra pruning mask proposal}
```


This is best understood as a **second-pass scan**, not as a replacement for existing pruning engines.

A central design choice is to use ICA rather than an orthogonal basis search. Orthogonal methods such as PCA/SVD restrict the basis vectors to be mutually perpendicular. ICA imposes a stricter and different condition: it looks for components whose activations are as statistically independent as possible. In practice, this means ICA is not satisfied with directions that merely explain variance; it tries to separate activation sources that have distinct non-Gaussian usage patterns.

What we gain from this stricter condition is a more selective protection signal. If a direction is preserved by ICA, it is not only large in energy, but also carries separable activation structure. For second-pass pruning, this can help avoid pruning weights that support rare or mixed activation sources that may be invisible to magnitude, SVD, Wanda-style activation norms, or local reconstruction alone.

---

## 2. Why use this after SparseGPT or Wanda?

### 2.1 Wanda

Wanda, short for *Pruning by Weights and Activations*, removes weights using a local score based on weight magnitude and input activation magnitude.

For a linear layer with weight matrix:


```math
W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```


Wanda uses a score of the form:


```math
\mathrm{WandaScore}_{ij}
=
|W_{ij}| \cdot \|X_j\|_2
```


where:

- $W_{ij}$ is the weight connecting input dimension $j$ to output dimension $i$,
- $X_j$ is the collected activation vector for input channel $j$,
- $\|X_j\|_2$ is the magnitude of that input activation channel across calibration samples.

Weights with the smallest scores are pruned, typically on a per-output basis.

This is efficient and effective because it uses only a forward pass and does not require retraining or weight updates. However, it is still local: it asks which weights look small after scaling by input activation strength.

A second-pass ICA scan asks a different question:

> Does this input channel participate in statistically salient activation components, or does it mostly support weak / redundant components?

Instead of using only a raw activation norm, ICA estimates component-level protection:


```math
\mathrm{input\_protection}_j
=
p^{\mathrm{in}}_j
=
\sum_{k=1}^{K}
\alpha^{\mathrm{in}}_k
\left|a^{\mathrm{in}}_{k,j}\right|
```


where:

- $K$ is the number of ICA components,
- $\alpha^{\mathrm{in}}_k$ is the estimated importance of input-side component $k$,
- $a^{\mathrm{in}}_{k,j}$ is entry $j$ of the **mixing direction** of component $k$ in the original input space (i.e., `ica.mixing_.T @ pca.components_` row $k$, entry $j$).

Then a Wanda-like ICA score is:


```math
\mathrm{ICAScore}_{ij}
=
|W_{ij}| \cdot p^{\mathrm{in}}_j
```


This preserves the spirit of Wanda but replaces the scalar input-channel norm with an ICA-derived protection score.

### 2.2 SparseGPT

SparseGPT formulates pruning as a layer-wise reconstruction problem. For a linear module:


```math
y = Wx
```


it seeks a sparse approximation $\widehat{W}$ such that:


```math
WX \approx \widehat{W}X
```


SparseGPT uses approximate second-order information to prune and compensate weights while preserving the local layer output.

This is much stronger than pure magnitude pruning, but it remains primarily a **layer-local output preservation** objective. It does not explicitly ask whether the preserved output directions correspond to statistically separable activation sources, nor does it explicitly identify which weight regions support those directions.

An ICA scan can be used after SparseGPT as a diagnostic layer:

1. run the already-pruned model on calibration data,
2. collect the surviving activations,
3. estimate ICA components,
4. project component protection back to weights,
5. propose a small extra pruning mask only where the ICA protection score is low.

The intended role is therefore:


```math
\text{SparseGPT/Wanda}
=
\text{primary pruning engine}
```



```math
\text{ICA scan}
=
\text{second-pass residual redundancy detector}
```


---

## 3. Mathematical formulation

Consider a target linear module:


```math
y = Wx
```


with:


```math
W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```



```math
x \in \mathbb{R}^{d_{\mathrm{in}}}
```



```math
y \in \mathbb{R}^{d_{\mathrm{out}}}
```


During calibration, we collect token-level input activation samples:


```math
X_{\mathrm{in}}
\in
\mathbb{R}^{n \times d_{\mathrm{in}}}
```


and optionally output activation samples:


```math
X_{\mathrm{out}}
\in
\mathbb{R}^{n \times d_{\mathrm{out}}}
```


where $n$ is the number of sampled token vectors.

### 3.1 ICA decomposition

ICA assumes that observed activations are mixtures of statistically independent latent sources.

The matrix form used throughout this document is:


```math
X \approx S A^\top
```


where:

- $X \in \mathbb{R}^{n \times d}$ is the observed activation matrix ($n$ samples, $d$ features),
- $S \in \mathbb{R}^{n \times K}$ is the estimated source activation matrix,
- $A \in \mathbb{R}^{d \times K}$ is the **mixing matrix** (column $k$ is the mixing direction of component $k$).

**Mixing matrix vs. unmixing matrix.**
`sklearn.decomposition.FastICA` exposes two related matrices:

| Attribute | Shape | Meaning |
|---|---|---|
| `components_` | $[K, d]$ | **Unmixing** matrix $W$: extracts sources via $S = X W^\top$ |
| `mixing_` | $[d, K]$ | **Mixing** matrix $A$: reconstructs data via $X \approx S A^\top$ |

For computing which original dimensions *participate* in each component — the "loading" in the notation below — this prototype uses the **mixing matrix** $A$.
Column $k$ of $A$ (i.e., row $k$ of $A^\top$) gives how strongly component $k$ mixes into each feature dimension.
The unmixing matrix $W$ answers the dual question: which direction to project the input onto in order to *extract* each component.
The implementation projects the mixing matrix back to original activation space via:

```
mixing_original[k] = ica.mixing_.T[k] @ pca.components_   # shape [original_dim]
```

Let the input-side mixing direction of component $k$ be:


```math
a^{\mathrm{in}}_k
\in
\mathbb{R}^{d_{\mathrm{in}}}
```


and the output-side mixing direction of component $k$ be:


```math
a^{\mathrm{out}}_k
\in
\mathbb{R}^{d_{\mathrm{out}}}
```


### 3.2 Component importance

The simplest component importance proxy is mean absolute source activation:


```math
\alpha_k
=
\mathbb{E}_{t}
\left[
\left|S_{t,k}\right|
\right]
```


where:

- $S_{t,k}$ is the estimated source activation of component $k$ for token/sample $t$,
- $\alpha_k$ is a cheap, non-causal importance proxy.

This is not a causal importance measure. Later versions can replace it with ablation-based loss sensitivity:


```math
\alpha_k
=
\Delta \mathcal{L}_k
```


where $\Delta \mathcal{L}_k$ is the loss increase after suppressing component $k$.

### 3.3 Input-side protection

Input feature/channel protection is defined as:


```math
p^{\mathrm{in}}_j
=
\sum_{k=1}^{K}
\alpha^{\mathrm{in}}_k
\left|a^{\mathrm{in}}_{k,j}\right|
```


where $a^{\mathrm{in}}_{k,j}$ is entry $j$ of the input-side mixing direction $a^{\mathrm{in}}_k$ (i.e., `mixing_original[k, j]` in the code).

Interpretation: input dimension $j$ is protected if it participates strongly (large mixing weight) in high-importance input-side ICA components.

### 3.4 Output-side protection

If output activations are also analyzed, define:


```math
p^{\mathrm{out}}_i
=
\sum_{k=1}^{K}
\alpha^{\mathrm{out}}_k
\left|a^{\mathrm{out}}_{k,i}\right|
```


where $a^{\mathrm{out}}_{k,i}$ is entry $i$ of the output-side mixing direction $a^{\mathrm{out}}_k$.

Interpretation: output dimension $i$ is protected if it participates strongly in high-importance output-side ICA components.

### 3.5 Weight-level score

The input-only score is:


```math
\mathrm{score}_{ij}
=
|W_{ij}| \cdot p^{\mathrm{in}}_j
```


The input-output score is:


```math
\mathrm{score}_{ij}
=
|W_{ij}| \cdot p^{\mathrm{out}}_i \cdot p^{\mathrm{in}}_j
```


Low score means that:

1. the weight magnitude is small, and
2. the input/output dimensions it connects are weakly protected by ICA components.

Those weights are proposed as additional pruning candidates.

---

## 4. Why ICA is stricter than an orthogonal basis search

SVD and PCA search for orthogonal directions. For a weight matrix, SVD gives:


```math
W = U \Sigma V^\top
```


where the columns of $U$ and $V$ are orthonormal. Low-rank pruning keeps the top singular directions:


```math
W_r = U_r \Sigma_r V_r^\top
```


This is an energy-preserving criterion: keep the orthogonal directions that explain the largest amount of squared mass. It is useful when the remaining redundancy is genuinely low-rank.

ICA asks for a stricter kind of separation. Instead of only requiring directions to be orthogonal, ICA estimates latent sources $S_1, \ldots, S_K$ such that their joint distribution factorizes as much as possible:


```math
p(S_1, \ldots, S_K)
\approx
\prod_{k=1}^{K} p(S_k)
```


Equivalently, ICA tries to reduce statistical dependence among components. One way to express the objective is to minimize mutual information:


```math
I(S_1, \ldots, S_K)
=
\sum_{k=1}^{K} H(S_k)
-
H(S_1, \ldots, S_K)
```


where $H$ denotes entropy.

Practical FastICA implementations approximate this by maximizing non-Gaussianity after whitening. Whitening may make the coordinates uncorrelated, but ICA then rotates the whitened space to find components that are more independent, not merely orthogonal.

The distinction can be summarized as:

| Method | Constraint | What it prefers | What it may miss |
|---|---|---|---|
| PCA/SVD | orthogonality | high-energy directions | low-energy but structured sources |
| Wanda-style scoring | magnitude × activation norm | locally strong channels | mixed source structure |
| SparseGPT | local output reconstruction | weights needed to preserve local layer output | source-level separability |
| ICA scan | approximate statistical independence | separable non-Gaussian activation sources | causal importance unless ablation is added |

The gain from this stricter condition is not philosophical interpretability. The gain is a potentially better second-pass pruning signal. If a component is statistically separable, then the weights supporting it may deserve protection even when their raw magnitude, singular energy, or channel activation norm is modest. Conversely, weights that are small and weakly connected to independent activation sources are stronger pruning candidates.

This is why ICA is useful after SVD, Wanda, or SparseGPT: those methods already remove obvious redundancy. ICA then asks whether the surviving activations still contain source-like structure that can be mapped back to weights.

---

## 5. Practical pipeline

The notebook implements the following steps.

### Step 1: Load model

Uses:

- `transformers.AutoModelForCausalLM`
- `transformers.AutoTokenizer`

The model is placed in `eval()` mode. No fine-tuning is performed.

### Step 2: Load calibration text

Supported inputs:

- Hugging Face dataset, such as `wikitext`
- plain text file, one line per sample

The text is tokenized to fixed sequence length.

### Step 3: Find target linear modules

The scan targets `torch.nn.Linear` modules with names such as:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`
- `fc1`, `fc2`, `c_fc`, `c_proj`

Embedding layers and `lm_head` are skipped by default.

### Step 4: Collect activations

Forward hooks collect input and output activations for each target module. To keep memory bounded, token vectors are subsampled and stored on CPU as `float32`.

### Step 5: Run ICA

For each module:

1. center activations,
2. optionally reduce dimension using PCA for numerical stabilization,
3. run `FastICA`,
4. compute source activation magnitudes,
5. compute component protection vectors.

PCA here is only a numerical preprocessing step before ICA. It is not used as the pruning criterion.

### Step 6: Score weights

For each target module, compute a score matrix with the same shape as the weight matrix.

Low-scoring nonzero weights are selected as proposed extra pruning candidates.

### Step 7: Save outputs

The output directory contains:

- `config.json`
- `summary.csv`
- `module_scores/*.pt`
- `proposed_masks/*.pt`
- optionally `pruned_model/` if mask application is enabled

---

## 6. Usage

This repository contains a Colab-compatible Jupyter notebook. To configure a run, edit the `ScanConfig` dataclass in the configuration cell:

```python
cfg = ScanConfig(
    model_name_or_path="facebook/opt-125m",
    dataset_name="wikitext",
    num_samples=64,
    seq_len=256,
    n_components=32,
    extra_sparsity=0.05,
    output_dir="outputs/opt125m_ica_scan",
    random_seed=42,
)
```

> **Note:** A standalone CLI script (`run_ica_prune_scan.py`) is not included in this repository. All configuration is done through the `ScanConfig` cell in the notebook. If you need a script interface, the notebook cells can be exported to a `.py` file and a standard `argparse` wrapper added around `ScanConfig`.

---

## 7. Outputs and interpretation

### `summary.csv`

Each row corresponds to one scanned linear module and includes:

- module name,
- weight shape,
- current sparsity,
- proposed extra sparsity,
- estimated total sparsity after applying the mask,
- score mean and standard deviation,
- number of proposed new zeros.

### Score matrix

For module weight matrix $W$, the score matrix has the same shape:


```math
\mathrm{score}
\in
\mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```


Low score means more prunable under the ICA-based criterion.

### Mask

The mask is a Boolean tensor with the same shape as the weight matrix. `True` entries indicate weights proposed for zeroing.

Existing zeros are ignored when selecting new pruning candidates. The `extra_sparsity` parameter applies only to weights that are currently nonzero.

---

## 8. Evaluation protocol

A minimal evaluation should compare:

1. original model,
2. already-pruned model,
3. already-pruned model plus random extra pruning,
4. already-pruned model plus magnitude extra pruning,
5. already-pruned model plus ICA-guided extra pruning.

At equal additional sparsity, measure:

- perplexity,
- downstream task accuracy,
- latency or memory impact if the sparsity pattern is hardware-supported,
- recovery after short fine-tuning, if allowed.

The key claim to test is:


```math
\Delta \mathrm{PPL}_{\mathrm{ICA}}
<
\Delta \mathrm{PPL}_{\mathrm{magnitude}}
```


at the same extra sparsity level.

A stronger claim would be:


```math
\mathrm{Sparsity}_{\mathrm{ICA}}(\epsilon)
>
\mathrm{Sparsity}_{\mathrm{baseline}}(\epsilon)
```


where $\epsilon$ is a fixed acceptable perplexity degradation threshold.

---

## 9. Limitations

This method is deliberately modest. It has several limitations:

1. **ICA components are not causal by default.**  
   Mean source activation is only a proxy. Component ablation would be a stronger importance measure.

2. **The score is local.**  
   It is computed per linear module and does not explicitly optimize the full model loss.

3. **Residual-stream interactions are indirect.**  
   The method sees them through calibration activations but does not model cross-layer circuits explicitly.

4. **FastICA can be unstable in high dimensions.**  
   The notebook uses PCA preprocessing for numerical stability, but this introduces a hyperparameter.

5. **Unstructured sparsity may not accelerate inference.**  
   To obtain real speedups, the mask may need to be converted into structured or semi-structured sparsity patterns.

6. **It is not a replacement for Wanda or SparseGPT.**  
   It is designed as a second-pass diagnostic or constraint generator.

---

## 10. Possible extensions

The prototype can be extended without changing the core idea:

- replace activation-energy component importance with component ablation loss,
- compare input-only vs input-output scoring,
- add per-output pruning budgets similar to Wanda,
- constrain the mask to 2:4 or 4:8 semi-structured sparsity,
- use the ICA score as a protection penalty inside a SparseGPT-like reconstruction objective,
- evaluate layer groups separately: attention projections vs MLP projections,
- add a small recovery fine-tuning phase after mask application.

A SparseGPT-like constrained objective could be:


```math
\min_{\widehat{W}}
\left\|WX - \widehat{W}X\right\|_F^2
+
\lambda
\sum_{i,j}
M_{ij} \cdot P_{ij}
```


where:

- $M_{ij}=1$ if weight $(i,j)$ is removed,
- $P_{ij}$ is the ICA protection score,
- $\lambda$ controls how strongly protected weights are discouraged from being removed.

This would keep SparseGPT as the pruning engine while using ICA as a protection prior.

---

## 11. References

- Frantar, E. and Alistarh, D. **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.** arXiv:2301.00774, 2023.
- Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. **A Simple and Effective Pruning Approach for Large Language Models.** arXiv:2306.11695, 2023.
- Hyvärinen, A. and Oja, E. **Independent Component Analysis: Algorithms and Applications.** Neural Networks, 2000.

---

## 12. Scope statement

This project is about **practical LLM compression diagnostics**. It does not attempt to prove that ICA components are human-readable concepts. It only tests whether statistically separated activation components provide a useful extra signal for second-pass pruning after standard compression methods have already been applied.
