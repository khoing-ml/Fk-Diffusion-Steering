# Evolving Guidance

Created: March 27, 2026 7:21 PM
Tags: Ideas
Text: Draft about how to mitigate the limitations of Mean Field particles methods for test-time scaling in diffusion/flow models.

## **1. Problem Formulation**

- Given an user preference reward model $r(x_0)$ and the goal is to sample from a distribution that tilts the distribution model’s generation $p_\theta(x_0)$ towards:

$$
p_{target}(x_0|c)=\frac{1}{Z}p_\theta(x_0|c)exp(\lambda r(x_0,c))
$$

## **2. Current Approaches**

### 2.1. Feynman-Kac Steering

$$
\begin{aligned}
&\textbf{Input: } \text{Diffusion model } p_\theta(\mathbf{x}_{0:T}\mid \mathbf{c}),\ \text{reward } r(\mathbf{x}_0,\mathbf{c}), \\
&\quad \text{proposals } \tau(\mathbf{x}_t \mid \mathbf{x}_{t+1}, \mathbf{c}),\ \text{potentials } G_t, \\
&\quad \text{intermediate rewards } r_\phi(\mathbf{x}_t,\mathbf{c}),\ \text{number of particles } K. \\[1em]

&\textbf{Sample } \mathbf{x}_T^i \sim \tau(\mathbf{x}_T \mid \mathbf{c}) \text{ for } i \in [K] \\
&\textbf{Score } G_T^i = G_T(\mathbf{x}_T^i,\mathbf{c}) \text{ for } i \in [K] \\[1em]

&\textbf{for } t \in \{T,\dots,1\} \textbf{ do} \\
&\quad \textbf{Resample: } \text{Sample } a_t^i \sim \text{Multinomial}(\mathbf{x}_t^1, \dots, \mathbf{x}_t^K; G_t^1, \dots, G_t^K) \\
&\quad \text{and let } \mathbf{x}_t^i = \mathbf{x}_t^{a_t^i} \text{ for } i \in [K] \\[0.5em]

&\quad \textbf{Propose: } \text{Sample } \mathbf{x}_{t-1}^i \sim \tau(\mathbf{x}_{t-1} \mid \mathbf{x}_t^i,\dots,\mathbf{x}_T^i,\mathbf{c}) \text{ for } i \in [K] \\[0.5em]

&\quad \textbf{Re-weight: } \text{Compute weight } G_{t-1}^i \text{ for } i \in [K]: \\
&\qquad G_{t-1}^i = \frac{p_\theta(\mathbf{x}_{t-1}^i \mid \mathbf{x}_t^i,\mathbf{c})}{\tau(\mathbf{x}_{t-1}^i \mid \mathbf{x}_{t:T}^i,\mathbf{c})} \, G_{t-1}(\mathbf{x}_{T:t-1}^i,\mathbf{c}) \\
&\textbf{end for} \\[1em]

&\textbf{Output: } \text{return samples } \{\mathbf{x}_0^i\}
\end{aligned}
$$

#### Issues:

- FKS degrades the diversity of model’s samples. It can be shown that when the number of inference steps $T \to \infty$, all particles converge to the particle whose future reward is the best
- Hinders the exploration of diffusion model $\to$ It cannot find the good samples when every initialized particle is determined to be failed in the future

$\implies$Can we find a method that utilizes the information (e.g., rewards) from the generated samples and guide the model to better (high-density) region of the reward-tilted distribution? 

## 3. Evolution Steering

- Given prompt $c$, base diffusion marginal at time t is $s^\theta(x,t|c) \approx \nabla_x{log\,p_t(x|c)}$. We sample $N$ trajectories $(x_{t}^{(n)})_{n=1,..,N}$ $(t = T \to 0)$, and score $x_{0}^{(n)}$ as:

$$
y_n= 
\begin{cases}
0,  & \text{if } r(x_0^{(n)})<threshold & (bad) \\
1,  & \text{else } & (good)
\end{cases}
$$

<aside>
💡

What should be the **threshold** here?

- Second max of each trajectory batch,
- Or something else
</aside>

- We assume that after sampling from the base model, the timestep samples also belong to 2 regions (good and bad). Denote their corresponding probabilities by:

$$
q_t(x|c)=p(x_t=x|y=1;c) \to G_t={(x_t^{(i)})_{i=1}^{N_g}}\\
b_t(x|c)=p(x_t=x|y=0;c)\to B_t=(y_t^{(j)})_{j=1}^{N_b}
$$

1. Given the collected samples, can we create a vector field in order to steer the new generated samples from $b_t(x|c)\to q_t(x|c)$?

$$
b_t(x_t|c) >> q_t(x_t|c) \to b_t(\phi(x_t)|c) << q_t(\phi(x_t)|c) 
$$

- Suppose the new generated samples at timestep $t$ is $x_t \sim b_t(x|c)$.  An approach to steer $x_t \to q_t(x|c)$ is to use **Stein Variational Vector Field**

#### Stein Variational Vector Field

- In **Stein variational gradient descent** (SVGD), the **Stein variational vector field** means the **optimal infinitesimal transport direction** that moves a current distribution $q$ closer to a target distribution $p$
- Formally, SVGD chooses a small transform  $T(x)=x+\epsilon \phi(x)$ and picks $\phi(x)$ to gives the steepest first-order decrease of $KL(q||p)$ within a function class, typically a vector-valued RKHS:

$$
\phi_{q, p}^*(\bullet)=E_{x \sim q}[k(x,\bullet)\nabla_x log\,p(x)+\nabla_xk(x,\bullet)]
$$

- With particles $x_1,x_2,...,x_n \sim q$, SVGD uses the empirical version:

$$
\hat{\phi^*}(x)= \frac{1}{n}\sum_{j=1}^{n}[k(x_j,x)\nabla_{x_j}log\,p(x_j)+\nabla_{x_j}k(x_j, x)]
$$

then updates $x_i=x_i+\epsilon \hat{\phi^*}(x_i)$

- Default kernel choice = RBF Kernel: $k(x, x')=exp(-\frac{||x-x'||^2}{\sigma})$

<aside>
💡

The only problem is how to evaluate $\nabla_xlog\,q_t(x|c)?$  

- Can we set $q=Law(x_t|c)$ and $p=b_t(x|c)$ so that we obtain $\hat{\phi}^*$ is a vector field that pushes $x_t$ to $b_t$ then $-\hat{\phi}^*$ pushes $x_t$  away from $b_t$?
</aside>

### Approximate score of good probability

Main issues: How do we evaluate the score of good probability $q_t(x|c)$ for new samples $x_1,...,x_n$

$$
\nabla_{x_t}log\,p(x_t=x|y=1;c) \\
\approx\sum_{i=1}^{N_G}w_i(x)\nabla_{x_t}log\,p(x_t=x|x_0=x_0^{(i)})\\
where: \; w_i(x) \propto p(x_t=x|x_0=x_0^{(i)}),\\ x_0^{(i)} \in G_0
$$

<aside>
💡

What if the new generated sample already belongs to $q_t(x|c)$?

</aside>

1.  And which timestep we should steer to prevent the quality of samples?
- High noise (high $t$) >< Low noise (low $t$)

## 4. Implementation Plan

### Goal

Build an evolution-guidance variant that **collects a diverse set of good and bad samples before applying strong guidance**, instead of estimating the guidance direction from only the current particle batch.

### Core idea

At each selected denoising step:

1. Run the base diffusion update and decode the predicted $x_0$ samples.
2. Score the particle population with the reward model.
3. Add the current high-reward and low-reward $x_0$ predictions to a bounded archive.
4. Keep the archive diverse by deduplicating anchors and pruning dense clusters.
5. Delay SVGD guidance until the archive contains enough useful anchors.
6. Once the archive is mature, guide the current particles using the archived good and bad sets rather than only the current batch.

### Archive design

- Maintain two archives:
  - **Good archive**: top-quantile reward samples
  - **Bad archive**: bottom-quantile reward samples
- Each archive should:
  - keep at most a fixed number of anchors
  - prefer high-quality anchors
  - preserve diversity across modes
  - remove exact duplicates created by resampling

### Guidance rule

Use a delayed, contrastive guidance signal:

$$
\phi_t(x) \approx \text{SVGD}(q_t^{good}) - \beta \, \text{SVGD}(q_t^{bad})
$$

or, in the score-based version,

$$
s_t(x) \approx \nabla_x \log q_t^{good}(x) - \beta \nabla_x \log q_t^{bad}(x)
$$

where the good and bad conditional densities are approximated from the archived $x_0$ anchors.

### Practical schedule

- **Warm-up / burn-in**: collect archive samples first, with weak or no guidance.
- **Start guidance only when**:
  - the number of unique good anchors is large enough
  - optionally the number of bad anchors is large enough
  - the archive covers multiple modes
- **Apply guidance at a controllable frequency**, rather than at every step.
- **Keep resampling sparse or disabled** during early experiments to reduce collapse.

### First experiment version

- Use a global prompt-level archive of good and bad $x_0$ predictions.
- Keep top 25% as good and bottom 25% as bad.
- Use a burn-in period before guidance starts.
- Guide every few steps, not every step.
- Compare:
  - current online guidance
  - delayed archive guidance
  - archive guidance with bad-sample repulsion

### Diagnostics to track

- number of unique good anchors over time
- number of unique bad anchors over time
- reward histogram at each step
- pairwise diversity of the archive
- whether guidance improves reward without collapsing to one visual mode
