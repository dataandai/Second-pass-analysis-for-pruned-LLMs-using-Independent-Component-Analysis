# ICA-Based Second-Pass Pruning Scan for LLMs

**A minimal diagnostic prototype for testing whether ICA-derived activation structure can provide a useful second-pass pruning signal after standard LLM compression methods.**

This repository explores a simple engineering question:

> After a model has already been pruned by a method such as magnitude pruning, Wanda, SparseGPT, or low-rank/SVD pruning, can Independent Component Analysis (ICA) on layer activations help identify a small additional set of weights that are relatively safe to prune?

The goal is not to replace existing pruning engines.

The goal is to test whether ICA can provide a useful **protection prior** or **extra mask proposal** for second-pass compression.

---

## Status

**Status:** minimal research prototype

**Primary use:** exploratory pruning diagnostics

**Core idea:** use ICA on calibration activations to build per-channel protection scores, project those scores back to weights, and propose a small additional pruning mask over already-surviving weights.

This repository is intentionally small and practical:

- no sparse autoencoder
- no LoRA
- no fine-tuning requirement
- no interpretability dashboard
- no claim that ICA components are human-readable concepts
- no claim that ICA is better than Wanda or SparseGPT

The prototype should be judged by evaluation, not by how sophisticated the decomposition sounds.

---

## What this is

This repository is a **second-pass pruning scan**.

It assumes that a model may already have been compressed by one of the following:

- magnitude pruning
- SVD / low-rank pruning
- Wanda
- SparseGPT
- another one-shot pruning method

Those methods remove the obvious redundancy according to their own criteria.

The ICA scan asks a follow-up question:

> Among the weights that survived the first pruning pass, are there still weights that are small and weakly connected to statistically separable activation structure?

If yes, those weights become candidates for an additional pruning mask.

---

## What this is not

This repository does **not** claim that:

- ICA components are circuits
- ICA components are human-interpretable concepts
- mean ICA source activation is causal importance
- the proposed mask is guaranteed to improve compression
- unstructured extra sparsity will automatically improve inference speed
- ICA replaces Wanda, SparseGPT, or reconstruction-based pruning

The scan is a diagnostic layer.

Its output is a pruning proposal that must be evaluated against baselines.

---

## One-sentence summary

The method collects calibration activations, fits ICA components, converts component participation into input/output protection scores, and proposes extra pruning only where weight magnitude and ICA protection are both low.

```math
\text{calibration activations}
\rightarrow
\text{ICA components}
\rightarrow
\text{channel protection scores}
\rightarrow
\text{weight scores}
\rightarrow
\text{extra pruning mask proposal}
```

---

## Why use this after Wanda or SparseGPT?

### Wanda

Wanda scores weights using weight magnitude and input activation magnitude.

For a linear layer:

```math
W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```

a Wanda-style score is:

```math
\mathrm{WandaScore}_{ij} = |W_{ij}| \cdot \|X_j\|_2
```

where \(X_j\) is the collected activation vector for input channel \(j\).

This is simple, efficient, and effective.

However, it is still a local magnitude-and-activation-norm criterion.

The ICA scan asks a different question:

> Does this channel participate in statistically separable activation components, or does it mostly support weakly protected structure?

The scan keeps the spirit of a cheap local score, but replaces or supplements raw channel norm with ICA-derived protection.

---

### SparseGPT

SparseGPT treats pruning as a layer-wise reconstruction problem.

For a linear module:

```math
y = Wx
```

it seeks a sparse approximation:

```math
WX \approx \widehat{W}X
```

SparseGPT is much stronger than pure magnitude pruning because it uses approximate second-order information and weight compensation to preserve local layer outputs.

The ICA scan is not a replacement for that.

A better way to think about the relationship is:

```math
\text{SparseGPT / Wanda} = \text{primary pruning engine}
```

```math
\text{ICA scan} = \text{second-pass diagnostic or protection prior}
```

The ICA scan can be run after a primary pruning method to inspect the surviving activations and propose a small extra mask.

---

## Hypothesis

A model that has already been pruned may still contain residual redundancy that is not fully captured by:

- small individual weights
- low singular value directions
- low input activation magnitude
- local layer-output reconstruction loss

The working hypothesis is:

> If surviving layer activations contain statistically separable source-like structure, then the weights supporting those structures may deserve protection. Conversely, small surviving weights that are weakly connected to protected ICA directions may be plausible candidates for a second pruning pass.

This is a hypothesis to test, not a conclusion.

---

## Method overview

For each target linear module:

1. collect calibration activations,
2. optionally reduce dimensionality with PCA for numerical stability,
3. run FastICA,
4. estimate component activity,
5. convert component loadings into input and/or output channel protection,
6. compute a weight-level pruning score,
7. propose a small extra pruning mask over currently nonzero weights.

The proposed mask should then be evaluated against random and magnitude-based extra pruning at the same additional sparsity level.

---

## Mathematical formulation

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

During calibration, collect token-level input activation samples:

```math
X_{\mathrm{in}} \in \mathbb{R}^{n \times d_{\mathrm{in}}}
```

and optionally output activation samples:

```math
X_{\mathrm{out}} \in \mathbb{R}^{n \times d_{\mathrm{out}}}
```

where \(n\) is the number of sampled token vectors.

---

## ICA decomposition

ICA assumes that observed activations are mixtures of statistically independent latent sources.

A common form is:

```math
X \approx SA^\top
```

where:

- \(X\) is the observed activation matrix,
- \(S\) is the estimated source activation matrix,
- \(A\) is the mixing matrix.

In implementation, `sklearn.decomposition.FastICA` returns estimated sources and component/mixing directions. For this prototype, the philosophical interpretation of the sources is not the point. They are used only as statistically separated activation directions.

Let an input-side ICA direction be:

```math
c^{\mathrm{in}}_k \in \mathbb{R}^{d_{\mathrm{in}}}
```

and an output-side ICA direction be:

```math
c^{\mathrm{out}}_k \in \mathbb{R}^{d_{\mathrm{out}}}
```

---

## Component importance

The simplest component importance proxy is mean absolute source activation:

```math
\alpha_k = \mathbb{E}_t \left[ |S_{t,k}| \right]
```

where:

- \(S_{t,k}\) is the source activation of component \(k\) for token/sample \(t\),
- \(\alpha_k\) is a cheap non-causal importance proxy.

This is not causal importance.

A stronger later version should replace or validate it with component ablation:

```math
\alpha_k = \Delta \mathcal{L}_k
```

where \(\Delta \mathcal{L}_k\) is the loss increase after suppressing component \(k\).

---

## Input-side protection

Input channel protection is defined as:

```math
p^{\mathrm{in}}_j =
\sum_{k=1}^{K}
\alpha^{\mathrm{in}}_k
\left|c^{\mathrm{in}}_{k,j}\right|
```

Interpretation:

> Input dimension \(j\) is more protected if it participates strongly in high-activity ICA directions.

This is not proof that the channel is important. It is only a diagnostic protection score.

---

## Output-side protection

If output activations are also analyzed, define:

```math
p^{\mathrm{out}}_i =
\sum_{k=1}^{K}
\alpha^{\mathrm{out}}_k
\left|c^{\mathrm{out}}_{k,i}\right|
```

Interpretation:

> Output dimension \(i\) is more protected if it participates strongly in high-activity output-side ICA directions.

---

## Weight-level score

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

Low score means:

1. the weight magnitude is small, and
2. the input/output dimensions connected by the weight are weakly protected by ICA-derived scores.

Those weights are proposed as additional pruning candidates.

Existing zeros are ignored. The scan only proposes additional pruning among weights that survived the first pruning pass.

---

## ICA versus PCA/SVD

PCA and SVD look for orthogonal directions that explain variance or energy.

For a weight matrix:

```math
W = U\Sigma V^\top
```

low-rank pruning keeps the leading singular directions:

```math
W_r = U_r\Sigma_rV_r^\top
```

This is useful when redundancy is well described by low-rank structure.

ICA uses a different and stronger assumption.

Instead of only requiring uncorrelated or orthogonal directions, ICA tries to find components whose source activations are as statistically independent as possible:

```math
p(S_1, \ldots, S_K)
\approx
\prod_{k=1}^{K} p(S_k)
```

A practical objective is related to reducing statistical dependence among components, often approximated in FastICA by whitening followed by maximizing non-Gaussianity.

The point is not that ICA is automatically better than PCA/SVD.

The point is that ICA may expose a different kind of activation structure: separable non-Gaussian usage patterns that are not necessarily the highest-energy orthogonal directions.

---

## Method comparison

| Method | Main signal | What it prefers | What it may miss |
|---|---|---|---|
| Magnitude pruning | weight size | small individual weights | activation-dependent importance |
| PCA/SVD | orthogonal energy directions | high-energy low-rank structure | low-energy but structured activation sources |
| Wanda | weight magnitude × input activation norm | locally strong activated channels | mixed or source-like activation structure |
| SparseGPT | local output reconstruction | weights needed to preserve layer outputs | explicit source-level protection |
| ICA scan | approximate statistical independence in activations | separable non-Gaussian activation directions | causal importance unless ablation is added |

The ICA scan is most useful if it improves second-pass pruning decisions beyond simple random or magnitude-based extra pruning.

---

## Practical pipeline

The notebook implements the following workflow.

### Step 1: Load model

Uses:

- `transformers.AutoModelForCausalLM`
- `transformers.AutoTokenizer`

The model is placed in `eval()` mode.

No fine-tuning is required for the scan.

### Step 2: Load calibration text

Supported inputs:

- a Hugging Face dataset such as `wikitext`,
- a plain text file with one sample per line.

Text is tokenized to a fixed sequence length.

### Step 3: Find target linear modules

The scan targets `torch.nn.Linear` modules with names such as:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`
- `fc1`
- `fc2`
- `c_fc`
- `c_proj`

Embedding layers and `lm_head` are skipped by default.

### Step 4: Collect activations

Forward hooks collect input and output activations for each target module.

To keep memory bounded:

- token vectors are subsampled,
- tensors are moved to CPU,
- activations are stored as `float32`.

### Step 5: Run ICA

For each module:

1. center activations,
2. optionally apply PCA preprocessing for numerical stability,
3. run `FastICA`,
4. compute source activation magnitudes,
5. compute component protection vectors.

PCA here is only a preprocessing step before ICA. It is not the pruning criterion.

### Step 6: Score weights

For each target module, compute a score matrix with the same shape as the module weight matrix.

Low-scoring currently nonzero weights are selected as proposed extra pruning candidates.

### Step 7: Save outputs

The output directory may contain:

- `config.json`
- `summary.csv`
- `module_scores/*.pt`
- `proposed_masks/*.pt`
- optionally `pruned_model/` if mask application is enabled

---

## Usage

Example command-line usage:

```bash
python run_ica_prune_scan.py \
  --model_name_or_path facebook/opt-125m \
  --dataset_name wikitext \
  --num_samples 64 \
  --seq_len 256 \
  --n_components 32 \
  --extra_sparsity 0.05 \
  --output_dir outputs/opt125m_ica_scan
```

In the Colab notebook, edit the `ScanConfig` cell instead of passing command-line arguments.

---

## Outputs

### `summary.csv`

Each row corresponds to one scanned linear module and includes fields such as:

- module name
- weight shape
- current sparsity
- proposed extra sparsity
- estimated total sparsity after applying the mask
- score mean
- score standard deviation
- number of proposed new zeros

### Score matrices

For module weight matrix \(W\), the score matrix has the same shape:

```math
\mathrm{score}
\in
\mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}
```

Lower score means more prunable under the ICA-derived criterion.

### Masks

Each proposed mask is a Boolean tensor with the same shape as the corresponding weight matrix.

`True` entries indicate weights proposed for zeroing.

The `extra_sparsity` parameter applies only to weights that are currently nonzero.

---

## Evaluation protocol

A minimal evaluation should compare:

1. original dense model,
2. already-pruned model,
3. already-pruned model plus random extra pruning,
4. already-pruned model plus magnitude extra pruning,
5. already-pruned model plus ICA-guided extra pruning.

At equal additional sparsity, measure:

- perplexity,
- downstream task accuracy,
- latency or memory impact if the sparsity pattern is hardware-supported,
- recovery after short fine-tuning, if fine-tuning is allowed.

The first key claim to test is:

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

where \(\epsilon\) is a fixed acceptable perplexity degradation threshold.

---

## Recommended results table

Before making strong claims, report something like:

| Model | First-pass pruning | Base sparsity | Extra sparsity | Random ΔPPL | Magnitude ΔPPL | ICA ΔPPL |
|---|---:|---:|---:|---:|---:|---:|
| OPT-125M | Wanda | 50% | +2% | TBD | TBD | TBD |
| OPT-125M | SparseGPT | 50% | +2% | TBD | TBD | TBD |
| OPT-350M | Wanda | 50% | +2% | TBD | TBD | TBD |

The method is only interesting if ICA-guided extra pruning is consistently less damaging than simple baselines at the same extra sparsity.

---

## Interpretation discipline

Use cautious wording.

| Avoid saying | Prefer saying |
|---|---|
| ICA finds circuits | ICA estimates statistically separated activation directions |
| ICA discovers concepts | ICA produces activation components under independence assumptions |
| protected weights are important | protected weights are less attractive pruning candidates under this score |
| ICA is better than Wanda | ICA-guided extra pruning performed better than the baseline in this experiment |
| component importance | component activity proxy |
| second-pass pruning method | second-pass pruning diagnostic or mask proposal |

This keeps the project honest and easier to evaluate.

---

## Limitations

### ICA components are not causal by default

Mean absolute source activation is only a proxy. Component ablation or loss sensitivity would be a stronger importance measure.

### The score is local

The score is computed per linear module. It does not directly optimize full-model loss.

### Residual-stream interactions are indirect

The method observes residual effects through calibration activations, but it does not explicitly model cross-layer circuits.

### FastICA can be unstable in high dimensions

PCA preprocessing improves numerical stability but introduces additional hyperparameters.

Recommended robustness checks:

- multiple random seeds,
- different calibration subsets,
- different numbers of ICA components,
- with and without output-side protection,
- comparison against random and magnitude baselines.

### Unstructured sparsity may not speed up inference

Extra unstructured zeros may reduce parameter count but not wall-clock latency.

For actual speedups, the mask may need to be converted into hardware-supported sparsity patterns such as 2:4 or block sparsity.

### This is not a replacement for Wanda or SparseGPT

The intended role is second-pass scanning, diagnostics, or protection-prior generation.

---

## Possible extensions

The prototype can be extended in several directions:

- replace mean activation importance with component ablation loss,
- compare input-only and input-output scoring,
- add per-output pruning budgets similar to Wanda,
- constrain masks to 2:4 or 4:8 semi-structured sparsity,
- use ICA scores as a protection penalty inside a SparseGPT-like objective,
- evaluate attention projections and MLP projections separately,
- add short recovery fine-tuning after mask application,
- add multi-seed and multi-calibration robustness reports.

---

## ICA as a SparseGPT protection prior

A stronger version of the idea is not to let ICA choose the pruning mask directly.

Instead, ICA can provide a penalty inside a SparseGPT-like reconstruction objective:

```math
\min_{\widehat{W}}
\left\|
WX - \widehat{W}X
\right\|_F^2
+
\lambda
\sum_{i,j}
M_{ij} P_{ij}
```

where:

- \(M_{ij}=1\) if weight \((i,j)\) is removed,
- \(P_{ij}\) is the ICA-derived protection score,
- \(\lambda\) controls how strongly protected weights are discouraged from being removed.

This keeps reconstruction as the primary objective while using ICA as a protection prior.

That is likely a more robust long-term direction than direct score-thresholding alone.

---

## References and related work

- Frantar, E. and Alistarh, D. **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.** arXiv:2301.00774, 2023.
- Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. **A Simple and Effective Pruning Approach for Large Language Models.** arXiv:2306.11695, 2023.
- Hyvärinen, A. and Oja, E. **Independent Component Analysis: Algorithms and Applications.** Neural Networks, 2000.

---

## Scope statement

This project is about practical LLM compression diagnostics.

It does not attempt to prove that ICA components are human-readable concepts or causal mechanisms.

It tests a narrower engineering hypothesis:

> ICA-derived activation structure may provide a useful extra protection signal for second-pass pruning after standard compression methods have already removed the obvious redundancy.

The repository is useful if it makes that hypothesis easy to test, reproduce, and falsify.
