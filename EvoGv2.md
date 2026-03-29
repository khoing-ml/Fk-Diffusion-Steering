# Archive-Based Evolution Guidance for Diffusion Inference-Time Scaling

## Abstract

We study inference-time scaling for diffusion and flow models under a reward model $r(x_0, c)$, where the objective is to sample from a reward-tilted distribution

$$
p_{\text{target}}(x_0 \mid c)
\propto
p_\theta(x_0 \mid c)\exp(\lambda r(x_0, c)).
$$

Existing particle-based approaches such as Feynman--Kac steering can improve reward but often suffer from genealogical collapse, reduced diversity, and poor exploration. We propose **Archive-Based Evolution Guidance (AEG)**, a delayed guidance framework that first collects a diverse archive of high-reward and low-reward samples and then constructs a smooth guidance field from the archive instead of relying only on the current particle population. The key idea is to approximate the score correction of the reward-tilted marginal at intermediate timesteps, rather than aggressively resampling trajectories. This yields a more stable tradeoff between reward optimization and sample diversity.

---

## 1. Problem Formulation

Let $c$ denote the conditioning prompt and let $p_\theta(x_0 \mid c)$ be the base generative distribution induced by a pretrained diffusion or flow model. Let $r(x_0, c)$ be a reward function, such as a user preference reward model.

Our target distribution is

$$
p_{\text{target}}(x_0 \mid c)
=
\frac{1}{Z(c)}
p_\theta(x_0 \mid c)\exp(\lambda r(x_0, c)),
$$

where $\lambda \ge 0$ controls the strength of reward optimization.

The inference-time scaling problem is:

> How can we steer the reverse-time generation process so that final samples are drawn from a distribution close to $p_{\text{target}}(x_0 \mid c)$, while preserving diversity and avoiding particle collapse?

---

## 2. A Better View of the Intermediate Target

The main object of interest at intermediate time $t$ is not simply a binary "good" region, but the marginal of the reward-tilted process.

Let $p_t^\theta(x_t \mid c)$ denote the base diffusion marginal at time $t$. Define

$$
h_t(x_t, c)
:=
\mathbb{E}\left[\exp(\lambda r(x_0, c)) \mid x_t, c\right].
$$

Then the exact reward-tilted marginal at time $t$ is

$$
\pi_t(x_t \mid c)
\propto
p_t^\theta(x_t \mid c)\, h_t(x_t, c).
$$

Therefore the correct guided score is

$$
\nabla_{x_t}\log \pi_t(x_t \mid c)
=
\nabla_{x_t}\log p_t^\theta(x_t \mid c)
+
\nabla_{x_t}\log h_t(x_t, c).
$$

If the base model score is

$$
s_\theta(x_t, t \mid c)
\approx
\nabla_{x_t}\log p_t^\theta(x_t \mid c),
$$

then the ideal guided score is

$$
s_{\text{guided}}(x_t, t \mid c)
=
s_\theta(x_t, t \mid c)
+
\nabla_{x_t}\log h_t(x_t, c).
$$

This reframes the problem:

> Inference-time scaling reduces to estimating the correction term $\nabla_{x_t}\log h_t(x_t, c)$ accurately and stably.

---

## 3. Why Existing Particle Methods Fail

### 3.1 Feynman--Kac Steering

A standard approach is to use sequential Monte Carlo or Feynman--Kac steering: sample trajectories, reweight them according to predicted future reward, and resample particles over time.

This can be effective for pushing reward upward, but it has several structural weaknesses:

1. **Genealogical collapse**  
   As the number of reverse steps grows, repeated resampling causes particles to share the same ancestors. In the extreme, the population collapses onto a small number of high-weight lineages.

2. **Loss of diversity**  
   The sampler increasingly concentrates on the currently best-looking trajectories, even when multiple good modes exist.

3. **Poor exploration**  
   If all current trajectories appear weak early on, the method has little ability to recover by exploring new promising regions later.

4. **High variance at intermediate timesteps**  
   Reward predictions from noisy intermediate states can be unreliable, making aggressive reweighting unstable.

The core issue is that Feynman--Kac methods entangle **exploration**, **selection**, and **guidance** into a single resampling process.

---

## 4. Proposed Method: Archive-Based Evolution Guidance

We propose to separate these roles.

Instead of estimating the guidance field from only the current batch of particles, we maintain a persistent archive of good and bad samples collected across denoising steps and use the archive to construct a delayed, smoother, and more diverse guidance signal.

### 4.1 Core intuition

At selected timesteps:

1. Run the base denoising update.
2. Decode or predict the corresponding $\hat{x}_0$ for each particle.
3. Score the predicted samples with the reward model.
4. Add high-reward and low-reward samples to a bounded archive.
5. Once the archive is sufficiently mature, construct a guidance field from the archive.
6. Apply the guidance field sparsely and mostly in the middle of the reverse trajectory.

This avoids early collapse and lets the sampler gather information before applying strong steering.

---

## 5. Good/Bad Regions as a Practical Approximation

For a practical first implementation, we define binary labels:

$$
y =
\begin{cases}
1, & \text{good sample}, \\
0, & \text{bad sample}.
\end{cases}
$$

At time $t$, define

$$
q_t(x \mid c) = p(x_t = x \mid y = 1, c),
\qquad
b_t(x \mid c) = p(x_t = x \mid y = 0, c).
$$

This gives two empirical regions:

- **good region**: partial states likely to lead to high-reward final samples,
- **bad region**: partial states likely to lead to low-reward final samples.

However, a hard threshold should **not** be the primary theoretical object. It is best viewed as a practical approximation to the softer reward-tilted target.

### 5.1 Recommended labeling strategy

A robust first choice is:

- label top 25% by reward as **good**,
- label bottom 25% as **bad**,
- ignore the middle 50%.

This avoids unstable thresholding and produces cleaner anchors.

### 5.2 Why soft weighting is better

A more principled alternative is to use soft weights

$$
w_i \propto \exp(\eta \tilde{r}_i),
$$

where $\tilde{r}_i$ is a normalized reward. This yields a smooth estimate of the reward tilt and avoids arbitrary binary decisions.

Still, the good/bad split is a useful first version because it supports simple contrastive guidance.

---

## 6. Key Identity: We Do Not Need to Estimate $\nabla \log q_t$ From Scratch

Using Bayes' rule,

$$
q_t(x \mid c)
=
p_t^\theta(x \mid c)\frac{P(y=1 \mid x_t=x, c)}{P(y=1 \mid c)}.
$$

Therefore,

$$
\nabla_x \log q_t(x \mid c)
=
\nabla_x \log p_t^\theta(x \mid c)
+
\nabla_x \log P(y=1 \mid x_t=x, c).
$$

Similarly,

$$
\nabla_x \log \frac{q_t(x \mid c)}{b_t(x \mid c)}
=
\nabla_x \log \frac{P(y=1 \mid x_t=x, c)}{P(y=0 \mid x_t=x, c)}.
$$

This is important: the practical goal is not to estimate the full score of $q_t$ from scratch, but rather to estimate a **time-conditional goodness signal** or **density ratio** on partial states.

That is a much more tractable problem.

---

## 7. Guidance Construction

We now describe two versions of the guidance field.

### 7.1 Contrastive density-ratio guidance

Let $f_t(x_t)$ be a feature representation of the current partial state. This may be:

- the latent $x_t$ itself,
- an intermediate U-Net feature,
- or a learned projection of the denoiser hidden state.

Let the good archive contain features $\{g_i\}_{i=1}^{M_g}$ and the bad archive contain features $\{b_j\}_{j=1}^{M_b}$, each associated with reward scores.

Define the contrastive log-density score

$$
\ell_t(x_t)
=
\log \sum_{i=1}^{M_g}\alpha_i K(f_t(x_t), g_i)
-
\log \sum_{j=1}^{M_b}\beta_j K(f_t(x_t), b_j),
$$

where:

- $K(\cdot,\cdot)$ is a similarity kernel,
- $\alpha_i$ are optional reward-based good weights,
- $\beta_j$ are optional bad weights.

A common choice is an RBF kernel

$$
K(u, v) = \exp\left(-\frac{\|u-v\|^2}{h^2}\right).
$$

Then define the guidance field

$$
g_t(x_t) := \nabla_{x_t}\ell_t(x_t).
$$

Finally, guide the base score by

$$
\tilde{s}_\theta(x_t, t \mid c)
=
s_\theta(x_t, t \mid c)
+
\gamma_t g_t(x_t),
$$

where $\gamma_t$ is a timestep-dependent strength.

### 7.2 Soft reward-weighted archive guidance

A more direct approximation to the reward-tilted marginal is

$$
h_t(x_t, c)
\approx
\sum_{i=1}^{M} w_i K(f_t(x_t), a_i),
\qquad
w_i \propto \exp(\lambda r_i),
$$

where $a_i$ are archived anchors.

Then use

$$
g_t(x_t)
=
\nabla_{x_t}\log h_t(x_t, c),
$$

and again guide with

$$
\tilde{s}_\theta
=
s_\theta + \gamma_t g_t.
$$

This version is closer to the ideal $\nabla \log h_t$, while the good/bad contrastive version is easier to implement initially.

---

## 8. Why Not Use Raw SVGD Directly?

A natural idea is to use a Stein variational vector field to push current particles toward the good region and away from the bad region.

Formally, SVGD for a target density $p$ and current distribution $q$ uses

$$
\phi_{q,p}^*(\cdot)
=
\mathbb{E}_{x\sim q}\left[
k(x,\cdot)\nabla_x \log p(x)
+
\nabla_x k(x,\cdot)
\right].
$$

However, in high-dimensional diffusion latent spaces, raw SVGD has several problems:

1. the kernel geometry is fragile,
2. the repulsion term can dominate unpredictably,
3. estimating $\nabla \log p$ is itself difficult,
4. a small online particle set gives a noisy field.

Therefore, instead of directly applying SVGD to the current batch, we propose to use the archive to construct a **contrastive score field** or **density-ratio field**, which plays a similar role but is more stable and easier to control.

In short:

- **do not** rely only on repulsion from bad samples,
- **do** combine attraction to good samples and repulsion from bad samples,
- or preferably estimate the reward-tilted score correction directly.

---

## 9. Archive Design

### 9.1 Two archives

Maintain two bounded archives:

- **Good archive**: high-reward samples,
- **Bad archive**: low-reward samples.

Each archive stores:

- predicted $\hat{x}_0$,
- corresponding partial-state feature $f_t(x_t)$,
- reward score,
- timestep,
- optional metadata such as prompt ID and sample ID.

### 9.2 Requirements

Each archive should:

1. keep at most a fixed number of anchors,
2. preserve multiple modes,
3. remove exact duplicates caused by repeated denoising or resampling,
4. prevent a single visual cluster from dominating.

### 9.3 Diversity preservation

We recommend:

- near-duplicate suppression using feature-space distance,
- cluster-aware pruning,
- or farthest-first retention.

This is essential. Without diversity control, the archive itself can collapse and the guidance will inherit that collapse.

---

## 10. When Should Guidance Start?

Guidance should be **delayed**.

At very early timesteps:

- $x_t$ is too noisy,
- $\hat{x}_0$ predictions are unreliable,
- reward scores are unstable.

At very late timesteps:

- sample semantics are already mostly fixed,
- strong guidance can damage fidelity or introduce artifacts.

### Recommended schedule

- **Burn-in phase**: no guidance or very weak guidance.
- **Middle phase**: strongest guidance.
- **Late phase**: taper guidance down again.

A practical rule:

- no guidance during the first 30-40% of denoising,
- strongest guidance during the middle 30-40%,
- reduced guidance during the final 10-20%.

Also, guidance should usually be applied **every few steps**, not every step.

---

## 11. Gating and Stability

If a current sample already lies in a strong good region, the method should not continue pushing it aggressively.

We recommend confidence-based gating:

$$
\hat{g}_t(x_t)
=
\sigma(m - \ell_t(x_t))\, g_t(x_t),
$$

where:

- $\ell_t(x_t)$ is the good-vs-bad logit,
- $m$ is a margin,
- $\sigma$ is a sigmoid.

This means:

- if a sample is ambiguous or bad, guidance remains active,
- if a sample is already confidently good, guidance fades out.

We also recommend norm clipping:

$$
\hat{g}_t
\leftarrow
\hat{g}_t
\cdot
\min\left(1,\frac{\tau}{\|\hat{g}_t\|+\varepsilon}\right).
$$

This is especially important at low noise.

---

## 12. Full Algorithm Sketch

### Archive-Based Evolution Guidance (AEG)

**Inputs**

- base diffusion model score $s_\theta(x_t, t \mid c)$,
- reward model $r(x_0, c)$,
- prompt $c$,
- number of particles $N$,
- denoising steps $T$,
- archive capacity $M_g, M_b$,
- guidance schedule $\gamma_t$,
- burn-in threshold,
- archive update rule.

**Initialize**

- sample $x_T^{(1)}, \dots, x_T^{(N)}$ from the base prior,
- initialize good archive $\mathcal{A}_g = \emptyset$,
- initialize bad archive $\mathcal{A}_b = \emptyset$.

**For** $t = T, T-1, \dots, 1$:

1. **Base denoising step**  
   Update particles using the base reverse sampler.

2. **Predict clean samples**  
   Compute $\hat{x}_0^{(n)}$ for each particle.

3. **Reward evaluation**  
   Score each predicted sample:
   $$
   r_n = r(\hat{x}_0^{(n)}, c).
   $$

4. **Archive update**  
   - add top-quantile samples to $\mathcal{A}_g$,
   - add bottom-quantile samples to $\mathcal{A}_b$,
   - deduplicate and prune to maintain diversity.

5. **Check maturity**  
   If archives are sufficiently populated and diverse, enable guidance.

6. **Construct guidance field**  
   For each particle, compute either:
   - contrastive log-density ratio guidance, or
   - soft reward-weighted archive guidance.

7. **Apply guidance**  
   Replace the base score with
   $$
   \tilde{s}_\theta
   =
   s_\theta + \gamma_t \hat{g}_t.
   $$

8. **Continue denoising**

**Return**

- final samples $\{x_0^{(n)}\}$.

---

## 13. Recommended First Experimental Version

A good first prototype is intentionally simple.

### Configuration

- Use a prompt-level archive shared across particles in one generation.
- Good archive = top 25% of reward scores.
- Bad archive = bottom 25%.
- Ignore the middle 50%.
- Use latent-space features or a U-Net hidden feature.
- Use burn-in before guidance starts.
- Guide every 3 steps.
- Apply guidance only in the middle denoising region.
- Use norm clipping and gating.

### Compare the following variants

1. **Base sampler**  
   No reward guidance.

2. **Online batch-only guidance**  
   Guidance estimated from only the current particle batch.

3. **Delayed archive guidance**  
   Guidance built from the accumulated archive.

4. **Delayed archive guidance + bad-sample repulsion**  
   Contrastive version using both good attraction and bad repulsion.

This comparison isolates whether the archive itself improves the reward-diversity tradeoff.

---

## 14. Diagnostics and Evaluation

The following metrics should be tracked throughout denoising.

### Reward-related

- reward mean,
- reward max,
- reward quantiles,
- final reward histogram.

### Diversity-related

- pairwise feature diversity,
- number of unique archive anchors,
- cluster occupancy of archive,
- mode collapse indicators.

### Guidance-related

- norm of guidance field over time,
- ratio of guidance norm to base score norm,
- number of steps where guidance is active,
- average good-vs-bad logit of current particles.

### Quality-related

- visual fidelity,
- prompt alignment,
- artifact rate,
- preference-model score vs human evaluation.

### Failure analysis

- does reward improve at the expense of fidelity?
- do samples collapse to one visual template?
- does guidance become unstable at late timesteps?
- does the archive become dominated by near duplicates?

---

## 15. Main Hypothesis

The central hypothesis is:

> A delayed archive-based guidance field can approximate the reward-tilted score correction more smoothly than online resampling-based methods, yielding higher reward than the base sampler while preserving significantly more diversity than Feynman--Kac steering.

---

## 16. Expected Advantages

Compared with sequential Monte Carlo style guidance, AEG offers:

1. **less collapse**  
   because it does not repeatedly resample ancestors;

2. **better exploration**  
   because particles are allowed to evolve before strong selection pressure is applied;

3. **more stable guidance**  
   because the archive aggregates information across steps rather than relying on a small online batch;

4. **better diversity-reward tradeoff**  
   because archive maintenance can explicitly preserve multiple modes;

5. **modular design**  
   because reward evaluation, archive construction, and guidance are separated.

---

## 17. Limitations and Open Questions

Several questions remain open.

### 17.1 Which feature space should define the archive?

Possible choices:

- latent $x_t$,
- predicted $\hat{x}_0$,
- intermediate U-Net feature,
- learned compact representation.

This choice may dominate performance.

### 17.2 How should the kernel bandwidth be chosen?

The guidance quality is sensitive to bandwidth and feature scaling.

### 17.3 Should good/bad archives be timestep-specific?

A single global archive is simple, but timestep-conditioned archives may better reflect denoising dynamics.

### 17.4 How much bad repulsion is useful?

Too much repulsion may push samples away from both bad and good manifolds.

### 17.5 Can the archive be amortized?

A learned estimator of $P(y=1 \mid x_t, c)$ or $h_t(x_t, c)$ could eventually replace hand-built kernel guidance.

---

## 18. One-Sentence Summary

Archive-Based Evolution Guidance replaces collapse-prone particle resampling with a delayed, diversity-preserving estimate of the reward-tilted score correction, constructed from an evolving archive of high-reward and low-reward partial trajectories.

---

## 19. Practical Recommendation

For the first implementation, the best path is:

- do **not** start with raw SVGD in high-dimensional space,
- do **not** use aggressive early-time guidance,
- do **not** rely on a hard threshold alone,

instead:

- collect a diverse archive,
- use mid-step delayed contrastive guidance,
- preserve multiple modes explicitly,
- and evaluate reward and diversity jointly.

That version is the most likely to work and to reveal the real behavior of the method.

---

## 20. Optional Naming Variants

Possible names for the method:

- **Archive-Based Evolution Guidance (AEG)**
- **Evolution Archive Guidance (EAG)**
- **Contrastive Archive Guidance (CAG)**
- **Reward-Tilted Archive Guidance (RTAG)**

My preferred name is **Archive-Based Evolution Guidance**, since it matches the intuition of collecting a population history and evolving a guidance field from it.