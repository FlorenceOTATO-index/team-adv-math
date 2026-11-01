# Mathematical reference: PCA, NMF and the CAE on Fashion-MNIST

This note supports the final report and oral presentation. Each statement is tagged:

* **[Theorem]**: a proven mathematical fact, with a reference or a short proof.
* **[Derived]**: follows directly from the definitions or code in this repository.
* **[Observed]**: a measured result from this project's files.
* **[Hypothesis]**: an interpretation of observations that this project has not tested causally.

Notation: training matrix $X \in [0,1]^{n \times p}$ with $n = 54{,}000$ images and
$p = 784$ pixels. Row $x_i$ is one flattened image, and $d$ is the latent dimension.
"MSE" always means the mean over images **and** pixels, $\frac{1}{np}\lVert X - \hat X\rVert_F^2$,
exactly as `evaluate.py` computes it.

---

## 1. PCA

**Centering and covariance.**
$\mu = \frac1n \sum_i x_i$, $X_c = X - \mathbf 1 \mu^\top$, $S = \frac1n X_c^\top X_c$.
(sklearn divides by $n-1$ when it reports variances. This rescales eigenvalues and
does not change directions.)

**SVD.** $X_c = U \Sigma V^\top$ with singular values $\sigma_1 \ge \sigma_2 \ge \dots$.
Then $S = V \,\mathrm{diag}(\sigma_j^2 / n)\, V^\top$, so the principal directions are
the columns of $V$ and the eigenvalues of $S$ are $\lambda_j = \sigma_j^2/n$.

**Encoding and reconstruction.** With $V_d$ the first $d$ columns of $V$:
$z = V_d^\top (x - \mu) \in \mathbb R^d$ and $\hat x = \mu + V_d z = \mu + P(x-\mu)$, where
$P = V_d V_d^\top$ satisfies $P^2 = P = P^\top$. This is an **orthogonal projection** onto a
$d$-dimensional affine subspace. The rank-$d$ reconstruction of the whole training
set is $\hat X_c = U_d \Sigma_d V_d^\top$.

**[Theorem] Eckart–Young–Mirsky (1936).** For every matrix $B$ with $\mathrm{rank}(B) \le d$,
$\lVert X_c - B\rVert_F^2 \ge \lVert X_c - U_d\Sigma_d V_d^\top\rVert_F^2 = \sum_{j>d} \sigma_j^2$.
Combined with the fact that the best affine offset is the mean (ESL §14.5.1),
centered PCA minimises training squared error over **all affine rank-$d$
reconstructions**:

$$\text{MSE}^{\text{PCA}}_{\text{train}}(d) = \frac{1}{np}\sum_{j>d}\sigma_j^2
= \frac{\operatorname{tr} S}{p}\Big(1 - \text{cumulative explained variance}(d)\Big).$$

**[Observed] Numerical check** on the 54K training partition, float64, script in Appendix A.
The predicted and empirical values agree to about $10^{-10}$:

| d | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| EYM-predicted PCA train MSE | 0.046298 | 0.036752 | 0.026699 | 0.020435 | 0.015132 | 0.010338 | 0.006269 |
| best uncentered rank-$d$ (lower bound for NMF) | 0.048871 | 0.037054 | 0.026859 | 0.020497 | 0.015149 | 0.010343 | 0.006271 |

$\operatorname{tr}S/p = 0.0870$ is the MSE of always predicting the mean image.

**Scope of PCA's optimality.** The theorem guarantees optimality only:

1. for **squared** error;
2. on the **data it was fitted to**. Test MSE is close for $n = 54$K but is not covered by the theorem;
3. among **linear/affine** reconstructions. A nonlinear decoder, such as the CAE's, is outside its scope, so the theorem neither forbids nor predicts that a CAE does better;
4. for **reconstruction**. Maximum variance is not the same as maximum class separation (compare Fisher LDA), so PCA's optimality says nothing about k-NN accuracy or silhouette.

## 2. NMF

**Model.** $X \approx WH$ with $W \in \mathbb R_{\ge 0}^{n\times d}$ (codes) and
$H \in \mathbb R_{\ge 0}^{d \times p}$ (basis images, sklearn's `components_`). The project uses
the Frobenius objective with no regularisation (sklearn defaults, `init="nndsvda"`, coordinate-descent solver):

$$\min_{W \ge 0,\, H \ge 0}\ \tfrac12 \lVert X - WH \rVert_F^2 .$$

**[Theorem] Non-convexity and non-uniqueness.** The objective is bilinear in $(W, H)$ and is not jointly
convex. It is convex in $W$ for fixed $H$, and the reverse: each block is a non-negative least-squares problem.
For any positive diagonal $D$ and permutation $\Pi$, $(W D\Pi,\ \Pi^\top D^{-1} H)$ gives the same $WH$.
Solutions are therefore never unique up to scaling, and the answer depends on the initialisation and the solver.

**Additive parts.** Each image is approximated as $\hat x = \sum_k w_k h_k$ with $w_k, h_k \ge 0$.
Components can only add, never cancel. This tends to produce localised "parts"
(Lee & Seung 1999), but that is an empirical tendency, **not a theorem**.

**[Theorem] Convergence.** Exact block coordinate descent and the multiplicative
updates never increase the objective (Lee & Seung 2001). Iterates approach
stationary points under mild conditions, but there is no guarantee of a global optimum.
**[Observed]** `results/nmf_convergence.csv` shows that d = 64 and d = 128 stopped at
`max_iter = 1000` without meeting the tolerance. Those rows describe non-converged solutions.

**[Theorem] Sparsity is not guaranteed.** Non-negativity allows exact zeros but
does not force them. Without an explicit $\ell_1$ penalty (`alpha_W`, `alpha_H`, `l1_ratio`)
codes and bases can be dense. Uniqueness and sparsity results need extra
geometric conditions such as separability (Donoho & Stodden 2003).

**[Derived] Encoding new images is an optimisation.** sklearn computes
$z = \arg\min_{w\ge 0}\lVert x - wH\rVert^2$ iteratively, whereas PCA uses one matrix
product. This is consistent with NMF's larger `encode_time_ms_per_img`.

**[Theorem] NMF cannot beat PCA on training reconstruction at equal d.**
Every NMF reconstruction $WH$ has rank $\le d$, so by Eckart–Young its error is
at least the best uncentered rank-$d$ error. Setting the offset to 0 is one
allowed choice of affine reconstruction, so that error is in turn at least
centered PCA's. Hence $\text{MSE}^{\text{NMF}}_{\text{train}} \ge \text{MSE}^{\text{rank-}d}_{\text{train}} \ge \text{MSE}^{\text{PCA}}_{\text{train}}$.
The table above shows the middle quantity. **[Observed]** On the test set, too, NMF's MSE is
above PCA's at every d in `results/results.csv`. The theorem covers training data only.

## 3. Convolutional autoencoder

**Convolution and shapes.** A 2-D convolution layer (cross-correlation, as in PyTorch) computes

$$y[c', i, j] = b[c'] + \sum_{c}\sum_{u,v=0}^{k-1} W[c', c, u, v]\; x[c,\ s i + u - q,\ s j + v - q],$$

with stride $s$, padding $q$ and zeros outside the image. The output size is
$\lfloor (N + 2q - k)/s \rfloor + 1$, so with $k=3, s=2, q=1$ the sizes go $28 \to 14 \to 7$.
A transposed convolution is the adjoint of that linear map, plus a bias. Its output size is
$(N-1)s - 2q + k + \text{output\_padding}$, so the sizes go $7 \to 14 \to 28$.

**Model.** Encoder $f_\theta:\mathbb R^{784}\to\mathbb R^d$ and decoder $g_\phi:\mathbb R^d \to (0,1)^{784}$:

$$f = L_e \circ \text{flatten} \circ \text{ReLU} \circ C_2 \circ \text{ReLU} \circ C_1, \qquad
g = \text{sigmoid} \circ T_2 \circ \text{ReLU} \circ T_1 \circ \text{reshape} \circ L_d .$$

$L_e$ and $L_d$ are the affine bottleneck layers ($1568 \to d$ and $d \to 1568$).
The full shape table is in `docs/person4.md`.

**Bottleneck.** **[Derived]** All reconstructions lie in $g_\phi(\mathbb R^d)$, the image
of a $d$-dimensional space under a continuous, piecewise-smooth map. It is a
(possibly curved) $d$-dimensional surface in pixel space, whereas PCA's is a
flat affine subspace.
**[Theorem]** A *linear* autoencoder trained with squared error has global minima
whose reconstruction is exactly the PCA projection, and it has no spurious local minima
(Baldi & Hornik 1989). Our CAE is nonlinear and cannot exactly represent the PCA
map: for instance, its output is confined to $(0,1)$. **No theorem says it must beat
PCA, or match it.** Whether it does is an empirical question.

**Objective.**
$\mathcal L(\theta,\phi) = \frac{1}{np}\sum_i \lVert x_i - g_\phi(f_\theta(x_i))\rVert^2$.
This is the same quantity as `evaluate()`'s MSE, and it is non-convex in $(\theta, \phi)$.

**[Derived] Parameter count.**
`Conv2d(c_in, c_out, k)` has $c_{out}c_{in}k^2 + c_{out}$ parameters,
`ConvTranspose2d(c_in, c_out, k)` has $c_{in}c_{out}k^2 + c_{out}$, and `Linear(a, b)` has $ab + b$:

$$\underbrace{160}_{C_1} + \underbrace{4640}_{C_2} + \underbrace{1569d}_{L_e}
+ \underbrace{1568d + 1568}_{L_d} + \underbrace{4624}_{T_1} + \underbrace{145}_{T_2}
= 11137 + 3137\,d .$$

At d = 32 this is 111,521. PCA stores $784(d+1)$ numbers and NMF stores $784d$. Each
CAE latent dimension costs $2 \times 1568 + 1 = 3137$ parameters, because both
bottleneck matrices act on a 1568-wide feature map, twice the pixel count. So
for large d the CAE has about 4 times as many parameters as PCA, and about 7.4 times as many at d = 2.

**[Derived] Compute per image**, counted in multiply-accumulates and independent of hardware.
The four convolutions cost $28{,}224 + 225{,}792 + 225{,}792 + 28{,}224 = 508{,}032$ MACs,
and each bottleneck layer costs $1568d$.
Encoding takes $254{,}016 + 1568d$ MACs, which is 304,192 at d = 32. PCA's encoding takes $784d$, which is 25,088.
So the CAE costs about 12 times as much as PCA per encoded image at d = 32, before
any hardware effects.

**Training.** We use mini-batch gradients computed by back-propagation and Adam updates:
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, and
$\theta_t = \theta_{t-1} - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ with bias-corrected $\hat m_t, \hat v_t$.
**[Theorem]** For non-convex objectives such methods are at best guaranteed to approach stationary points,
and they come with no global-optimality guarantee. Results therefore depend on the
seed through initialisation and batch order. That is why the CAE is run with
five seeds and reported as mean ± standard deviation. Early stopping on
validation MSE chooses the training length without looking at the test set.

**[Theorem] Low reconstruction loss does not determine class separability.**
Both bottleneck layers are affine. For any invertible $A \in \mathbb R^{d\times d}$,
replacing $(W_e, b_e, W_d)$ by $(A W_e,\ A b_e,\ W_d A^{-1})$ leaves every reconstruction,
and therefore the loss, exactly unchanged. It maps the latent codes to $z' = Az$.
k-NN with Euclidean distance is invariant only to similarity transforms (rotation, reflection,
translation, uniform scaling), and silhouette is invariant to the same transforms. A non-orthogonal $A$,
such as stretching one axis, can change the nearest neighbours and the silhouette score while the loss
stays identical. So the training objective leaves the latent geometry that k-NN
and silhouette measure undetermined. A further reason is that the objective
never sees labels: it spends capacity on whatever pixel variation reduces
squared error most, such as brightness, size or texture, whether or not that
variation distinguishes classes.

## 4. Evaluation metrics

* **k-NN accuracy.** Each test code is labelled by majority vote among its $k = 9$ nearest training codes in Euclidean distance. k was chosen on validation data from PCA at d = 32. k-NN is invariant to similarity transforms of the latent space. It is not invariant to rescaling individual axes, which matters for NMF's scaling ambiguity and the CAE's free $A$ above. **[Theorem]** For exact PCA with $d = p$, $z = V^\top(x-\mu)$ is an isometry. PCA k-NN then equals raw-pixel k-NN.
* **Euclidean distance in latent space.** **[Derived]** For PCA, $\lVert z_i - z_j\rVert = \lVert P(x_i - x_j)\rVert \le \lVert x_i - x_j\rVert$. Projection never increases pixel distances; it discards the components outside the subspace. For the CAE, distances are those of a learned nonlinear embedding and have no fixed relation to pixel distances.
* **Silhouette.** $s(i) = \dfrac{b(i)-a(i)}{\max(a(i), b(i))}$. Here $a(i)$ is the mean distance from point $i$ to the other points of its own class, and $b(i)$ is the smallest mean distance to the points of another class. The score is the average over a 5,000-point test subsample whose random seed equals the run seed. Silhouette rewards compact, well-separated, roughly convex classes. k-NN needs only *local* label purity. Elongated or multi-modal classes, such as shirts versus T-shirts, can give a low silhouette with a high k-NN accuracy.
* **Pixel-wise MSE.** $\frac{1}{np}\lVert X_{test} - \mathrm{clip}_{[0,1]}(\hat X)\rVert_F^2$. **[Theorem]** Clipping never increases the error, because for $x \in [0,1]$, $|\mathrm{clip}(a) - x| \le |a - x|$, so PCA and NMF can only gain from it. For the CAE, the sigmoid output is already in $(0,1)$ and clipping does nothing.
* **Training cost.** `fit_time_s` is wall-clock time. In terms of operations, full-SVD PCA costs $O(np^2 + p^3)$ once, independent of d in our runs. NMF costs $O(npd)$ per iteration, times the number of iterations. The CAE costs about $3 \times (508{,}032 + 3136d)$ MACs per image per epoch, times 54,000 images, times the epochs until early stopping. The factor 3 covers forward plus backward. **The CAE was timed on a GPU and PCA/NMF on CPUs.** Wall-clock times therefore combine algorithmic cost with hardware and must not be read as a hardware-normalised comparison.
* **Encoding cost.** `encode_time_ms_per_img` covers 1,000-image chunks. For the CAE it includes the transfer to and from the GPU.
* **Model complexity vs d.** PCA has $784(d+1)$ parameters, NMF $784d$ and the CAE $11137 + 3137d$. All three are linear in d with slopes 784, 784 and 3137.

**Why the three metrics can disagree.** They measure different properties with different invariances:

| Metric | Depends on | Invariant to |
|---|---|---|
| MSE | encoder **and** decoder | any invertible reparametrisation of the CAE latent space |
| k-NN accuracy | local neighbourhoods of the codes | similarity transforms only |
| Silhouette | global compactness of each class | similarity transforms; also depends on the 5K subsample |

A method can therefore lower MSE by modelling class-irrelevant detail. It can
raise k-NN accuracy while silhouette stays low because classes are separable
locally but not compact. It can also change silhouette without changing MSE.

---

## 5. Observed results and interpretation

Sources: `results/results.csv` (merged by `scripts/merge_results.py`),
`results/cae_summary.csv` and `results/cae_vs_pca_mcnemar.csv`. All values are on the
official 10K test set and come from the frozen protocol (`results/cae_protocol.json`).

**How to read the CAE numbers.** Each CAE cell is the mean ± sample standard deviation
(n − 1 denominator) over the five training seeds, all scored on the **same** test set. What the spread contains
depends on the metric:

* **MSE and k-NN accuracy.** Scoring a trained model is deterministic here, so their
  spread comes from training randomness: initialisation, mini-batch order and the
  early-stopping epoch these lead to.
* **Silhouette.** Training randomness plus the seed-dependent 5,000-image test
  subsample that `evaluate()` scores.
* **Fit and encode times.** Fit time varies with the number of epochs, which depends on
  the seed. Both times also include system-level measurement variability, such as
  shared-server load, GPU clocks and scheduling. Encoding does the same work for every
  seed at a given d, so its spread is measurement variability.

None of these spreads includes test-set sampling variability. None is a standard error
over independent test sets, and dividing by √5 does not give one. PCA and NMF are single
deterministic runs at seed 0.

| d | MSE: CAE | MSE: PCA | MSE: NMF | k-NN: CAE | k-NN: PCA | k-NN: NMF | silhouette: CAE | silhouette: PCA | silhouette: NMF |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.03103 ± 0.00041 | 0.04565 | 0.04919 | 0.6517 ± 0.0117 | 0.5251 | 0.4828 | 0.0685 ± 0.0126 | −0.0323 | −0.0474 |
| 4 | 0.02105 ± 0.00006 | 0.03593 | 0.03810 | 0.7658 ± 0.0036 | 0.7078 | 0.6568 | 0.0912 ± 0.0039 | 0.0444 | 0.0109 |
| 8 | 0.01467 ± 0.00013 | 0.02595 | 0.02880 | 0.8370 ± 0.0010 | 0.7923 | 0.7538 | 0.1032 ± 0.0074 | 0.0773 | 0.0180 |
| 16 | 0.01049 ± 0.00010 | 0.01982 | 0.02266 | 0.8632 ± 0.0008 | 0.8357 | 0.7798 | 0.0890 ± 0.0040 | 0.0776 | 0.0087 |
| 32 | 0.00751 ± 0.00014 | 0.01450 | 0.01766 | 0.8741 ± 0.0005 | 0.8515 | 0.7829 | 0.0782 ± 0.0035 | 0.0679 | 0.0267 |
| 64 | 0.00528 ± 0.00006 | 0.00984 | 0.01274 | 0.8757 ± 0.0012 | 0.8583 | 0.7579 | 0.0646 ± 0.0031 | 0.0595 | −0.0126 |
| 128 | 0.00339 ± 0.00006 | 0.00591 | 0.00825 | 0.8695 ± 0.0002 | 0.8619 | 0.7543 | 0.0550 ± 0.0025 | 0.0528 | −0.0290 |

Raw pixels (d = 784, single run): k-NN 0.8478, silhouette 0.0452.

**Cost.** CAE times are mean ± SD over the five seeds. That spread includes system-level measurement variability as well as the seed-dependent number of epochs. PCA and NMF times are single measurements.
See §5.4 before comparing them.

| d | fit s: CAE | fit s: PCA | fit s: NMF | encode µs/img: CAE | encode µs/img: PCA | encode µs/img: NMF | params: CAE | params: PCA | params: NMF |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 40.7 ± 10.1 | 1.60 | 1.5 | 0.95 ± 0.04 | 2.33 | 2.22 | 17,411 | 2,352 | 1,568 |
| 4 | 32.0 ± 5.2 | 1.81 | 2.8 | 0.97 ± 0.04 | 0.71 | 2.80 | 23,685 | 3,920 | 3,136 |
| 8 | 57.7 ± 6.6 | 1.58 | 4.3 | 0.97 ± 0.02 | 0.36 | 4.92 | 36,233 | 7,056 | 6,272 |
| 16 | 54.6 ± 14.0 | 1.60 | 10.0 | 0.99 ± 0.02 | 0.35 | 10.02 | 61,329 | 13,328 | 12,544 |
| 32 | 49.9 ± 6.8 | 1.69 | 15.0 | 1.01 ± 0.02 | 4.95 | 45.63 | 111,521 | 25,872 | 25,088 |
| 64 | 59.0 ± 15.0 | 1.68 | 101.1 | 1.16 ± 0.05 | 0.69 | 193.30 | 211,905 | 50,960 | 50,176 |
| 128 | 46.1 ± 7.4 | 1.55 | 377.2 | 1.31 ± 0.13 | 1.06 | 509.86 | 412,673 | 101,136 | 100,352 |

### 5.1 What was measured

* **[Observed] Reconstruction.** The CAE's mean test MSE is 52–68 % of PCA's at every d and below NMF's at every d. The seed standard deviation is at most 0.0004, far smaller than these gaps. Within the d grid, the CAE at d = 8 has about PCA's d = 32 error (0.0147 vs 0.0145), and the CAE at d = 16 about PCA's d = 64 error (0.0105 vs 0.0098). The grid is coarse, so "about four times fewer dimensions" is a rough reading of two comparisons, not a fitted ratio. This does not contradict §1, because Eckart–Young bounds only affine reconstructions.
* **[Observed] k-NN accuracy compared with PCA.** Every one of the 35 CAE runs has higher k-NN accuracy than PCA at the same d. The per-d summary of the paired test below comes from `results/cae_vs_pca_mcnemar.csv`. "CAE only" counts test images the CAE classifies correctly and PCA does not; "PCA only" is the reverse. Ranges are over the five seeds.

  | d | CAE − PCA, points: mean [min, max] | CAE only | PCA only | largest raw p | largest Holm-adjusted p |
  |---|---|---|---|---|---|
  | 2 | 12.66 [11.33, 14.30] | 2153–2261 | 831–1020 | 8.7e-92 | 2.7e-90 |
  | 4 | 5.80 [5.31, 6.24] | 1315–1378 | 741–784 | 2.9e-31 | 6.0e-30 |
  | 8 | 4.47 [4.36, 4.61] | 860–878 | 407–427 | 1.5e-34 | 3.4e-33 |
  | 16 | 2.75 [2.68, 2.86] | 572–589 | 301–321 | 4.8e-19 | 8.2e-18 |
  | 32 | 2.26 [2.20, 2.34] | 449–468 | 229–241 | 2.3e-17 | 2.6e-16 |
  | 64 | 1.74 [1.56, 1.88] | 395–428 | 219–253 | 6.2e-10 | 3.7e-09 |
  | 128 | 0.76 [0.72, 0.78] | 308–332 | 233–256 | 2.9e-03 | 6.1e-03 |

  The raw p-values are exact two-sided McNemar tests. The adjusted values use Holm's step-down correction across all 35 comparisons. All 35 are significant at 0.05 both before and after adjustment. At d = 128 the adjusted p-values are about 0.006: the difference is small (0.7–0.8 points) but appears in every seed.
* **[Observed] Comparison with raw pixels.** The CAE's mean accuracy exceeds the raw-pixel baseline (0.8478) from d = 16, PCA's from d = 32. NMF's accuracy is below PCA's at every d. These comparisons were not tested for significance.
* **[Observed] Metrics move in different directions within the CAE.**
  * From d = 64 to d = 128 the mean MSE falls by 36 %, but k-NN accuracy falls from 0.8757 to 0.8695. All five d = 128 runs are below all five d = 64 runs. No paired significance test was run for this comparison.
  * Silhouette peaks at d = 8 for the CAE and at d = 8–16 for PCA, then declines while MSE keeps improving.
  * This is consistent with §3, which proves that reconstruction loss cannot determine latent geometry. It does not explain *why* these particular changes occur.
* **[Observed] Silhouette is low for every method.**
  * All values are at most 0.103. By the usual reading of silhouette values, no representation forms compact, well-separated classes, even though k-NN reaches 0.88.
  * Every CAE seed has a higher silhouette than PCA at every d. At d ≥ 64, however, the gap (0.002–0.005) is about the size of the CAE's seed standard deviation.
  * Each CAE seed's silhouette uses a different 5,000-image subsample, while PCA's uses the seed-0 subsample. No significance test was run for silhouette.
* **[Observed] Cost, in the measured environments** (see §5.4 for limits).
  * Mean CAE fit time was 32–59 s. PCA's was 1.6–1.8 s and NMF's 1.5–377 s.
  * CAE time per epoch was flat at 0.23–0.24 s for every d. In this environment, CAE fit time is therefore governed by how many epochs early stopping takes (102–334, varying by seed) rather than by d.
  * At d ≥ 64, NMF's recorded fit time exceeds the CAE's. NMF also did not converge at those d (§2).
  * Mean CAE encoding time was 0.95–1.31 µs per image, including GPU transfers. The single PCA measurements range from 0.35 to 4.95 µs, and PCA at d = 32 measured fourteen times PCA at d = 16, which shows how noisy single timings are.
* **[Derived] Hardware-independent cost.** The CAE has 4.1–7.4 times as many parameters as PCA. At d = 32 it needs about 12 times as many multiply-accumulates per encoded image (§3).

### 5.2 Answer to the research question, under this protocol

These statements summarise the measurements above. The judgments about when the
cost is worth it are interpretations.

* **Reconstruction.** The CAE's mean test MSE is 32–48 % lower than PCA's at every d tested. In two grid comparisons it matches PCA's error with a quarter of the latent dimensions.
* **Classification utility (k-NN).** The measured gain over PCA is largest at small d: +4.5 to +12.7 points for d ≤ 8. It is +2.3 to +2.8 points at d = 16–32 and +0.8 to +1.7 points at d = 64–128. All of these remain significant after Holm correction on this test set.
* **Cost.**
  * In the measured environments, mean CAE training took 18–37 times PCA's recorded fit time. The CAE ran on one GPU; PCA ran on a CPU machine the repository does not record. That ratio belongs to those two environments and is not a hardware-normalised comparison (§5.4).
  * The hardware-independent figures are 4.1–7.4 times as many parameters and about 12 times as many encoding operations at d = 32.
  * CAE training on a CPU was not measured.
* **Interpretation.** On these measurements the case for the CAE is strongest when the representation must be very compact or when reconstruction fidelity matters. At d ≥ 64, whether a 0.8–1.7 point k-NN gain justifies the extra cost is a judgment for the application.

### 5.3 Hypotheses (consistent with the data, not tested here)

None of these was tested. They are candidate explanations, not findings.

* **H1.** The CAE's large advantage at small d *may* come from its nonlinear encoder and decoder, which can follow a curved low-dimensional surface. PCA's flat 2-D subspace explains only 47 % of the variance. Other differences, such as training procedure and early stopping on validation data, have not been ruled out. A *linear* autoencoder trained with the same procedure, which should recover PCA's subspace (Baldi & Hornik), would separate these explanations.
* **H2.** Between d = 64 and d = 128, the extra capacity *may* be spent on within-class detail that enlarges within-class distances and slightly lowers k-NN accuracy. The printed text visible at d = 128 in `figures/cae_test_reconstructions_s0.png` illustrates such detail for one image and one seed. It is not evidence for the mechanism.
* **H3.** Silhouette *may* fall with d because each extra dimension adds within-class variance to Euclidean distances faster than it adds separation between classes.
* **Not examined.** Whether adding a ReLU after the decoder Linear changes any conclusion. How much timing varies between repeated measurements.

### 5.4 Statistical and measurement caveats

* **Sampling error.** On a 10K test set, one accuracy value has a binomial standard error of about 0.34 points. The paired McNemar test is more sensitive than comparing two accuracies, because it uses only the images on which the two classifiers disagree.
* **What each McNemar test covers.** Each test compares one trained CAE with PCA on this one test set. It conditions on both trained models, so it says nothing about other test sets or other training runs.
* **The five seeds are not independent experiments.** They share the test set and the PCA predictions, so the 35 p-values are dependent. Holm's correction controls the family-wise error rate under arbitrary dependence, which makes it valid here. The p-values must not be combined as if the seeds were independent test datasets.
* **Seed standard deviations.** For MSE and k-NN accuracy they reflect training randomness. For silhouette they also include the seed-dependent subsample. For timings they also include system-level measurement variability. None reflects test-set sampling (see "How to read the CAE numbers").
* **Single runs.** PCA and NMF were each run once, which is appropriate for deterministic methods.
* **Timing limitations.**
  * Fit and encode times are single wall-clock measurements per run.
  * CAE times were recorded on an NVIDIA RTX 5000 Ada in a shared server. Other users' jobs were running on other GPUs, and the recorded CPU load average was about 12–16 on 40 cores.
  * PCA and NMF ran on CPUs, because scikit-learn has no GPU path, on a machine the repository does not record.
  * CAE fit time includes the per-epoch validation pass that early stopping needs. CAE encode time includes host-to-GPU transfers.
  * The training-time ratios (18–37 times PCA) and the d ≥ 64 crossover with NMF therefore belong to these particular environments. They are **not** hardware-normalised and would change on other hardware.
  * Neither the CAE on a CPU nor PCA on a GPU was measured.
  * Parameter and operation counts (§3) are the hardware-independent comparison.

---

## Appendix A: Eckart–Young numerical check

```python
import numpy as np
from data import load_data
X = load_data()[0].astype(np.float64)            # 54K training images only
n, p = X.shape
s = np.linalg.svd(X - X.mean(0), compute_uv=False)
for d in [2, 4, 8, 16, 32, 64, 128]:
    print(d, (s[d:] ** 2).sum() / (n * p))       # equals sklearn PCA's training MSE
```

## Appendix B: paired McNemar test of CAE vs PCA k-NN (post-hoc, test set)

This was run after the official evaluation; it informs interpretation only. Its output
is `results/cae_vs_pca_mcnemar.csv`. The script also asserts that every recomputed
accuracy equals the value in `results/results.csv`.

```python
import numpy as np
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.multitest import multipletests
import config
from data import load_data
X_train, y_train, _, _, X_test, y_test = load_data()
knn = lambda A, B: KNeighborsClassifier(config.KNN_K, n_jobs=-1).fit(A, y_train).predict(B)
p_raw = []
for d in config.D_GRID:
    pca = PCA(n_components=d, svd_solver="full").fit(X_train)       # reproduces results_pca.csv
    ok_p = knn(pca.transform(X_train), pca.transform(X_test)) == y_test
    for s in config.SEEDS:
        z = np.load(f"latents/CAE_d{d}_s{s}.npz")
        ok_c = knn(z["Z_train"], z["Z_test"]) == y_test
        n11, n10 = np.sum(ok_c & ok_p), np.sum(ok_c & ~ok_p)        # both right / CAE only
        n01, n00 = np.sum(~ok_c & ok_p), np.sum(~ok_c & ~ok_p)      # PCA only / both wrong
        p_raw.append(binomtest(n10, n10 + n01, 0.5).pvalue)          # exact two-sided McNemar
p_holm = multipletests(p_raw, method="holm")[1]                     # 35 comparisons
```

The exact p-values were also recomputed in rational arithmetic as
$2\,P(X \le \min(n_{10}, n_{01}))$ with $X \sim \text{Bin}(n_{10}+n_{01}, \tfrac12)$, and they
agree with SciPy. The Holm values were also computed directly and agree with statsmodels.

## References

* Eckart, C. & Young, G. (1936). The approximation of one matrix by another of lower rank. *Psychometrika* 1, 211–218.
* Baldi, P. & Hornik, K. (1989). Neural networks and principal component analysis: learning from examples without local minima. *Neural Networks* 2, 53–58.
* Lee, D. & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature* 401, 788–791; (2001) Algorithms for non-negative matrix factorization. *NIPS 13*.
* Donoho, D. & Stodden, V. (2003). When does non-negative matrix factorization give a correct decomposition into parts? *NIPS 16*.
* Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, §14.5.
* Rousseeuw, P. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.* 20, 53–65.
