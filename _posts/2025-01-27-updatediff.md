---
layout: distill
title: Diffusion Models
description: an overview of the mathematics of Diffusion Models.
tags: 
giscus_comments: true
date: 2025-01-26
featured: False
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true
thumbnail: assets/img/noisycane.PNG

authors:
  - name: Matteo Lippi

bibliography: diffusion.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: 0. Introduction and Intuitions
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: 1. Defining the Forward Process
  - name: 2. Defining the Reverse Process
    subsections:
      - name: "Note: Variational approach and equivalence with Score Matching"
    #   - name: Example Child Subsection 2
  - name: 3. Score Matching
    subsections:
      - name: Denoising Score Matching
      - name: Implicit Score Matching  
  - name: 4. Score Based Models
  - name: 5. Continuous Time View
  - name: 6. Equivalence to Ordinary Differential Equation (ODE) formulation
  - name: 7. Conditional Generation and Inverse Problems
  - name: 8. Sources
---

## 0. Introduction and Intuitions

One of the main goals of generative modelling is, given samples $$x_i \sim p_{data}(x)$$ from an unknown data distribution, to generate new samples $$x_{new}$$. This is usually a complicated task, because: 
1. $$p_{data}(x)$$ is not known.
2. $$p_{data}(x)$$ may have a very complicated form that makes sampling from it difficult.

Diffusion models approach this problem by first sampling $$z_{new} \sim p_{easy}(x)$$ from an easy-to-sample distribution, and then defining a way to transform $$z_{new}$$ into a sample $$x_{new}$$ of $$p_{data}(x)$$. This is similar to what is done in other generative models, such as Variational Autoencoders or Normalising Flows.  
A key insight of Diffusion Models is that transforming a sample between two probability distributions is easy if the transition is small. Because of this reason, to transform as sample from $$p_{easy}(x)$$ into a sample from a potentially very different $$p_{data}(x)$$, it is very useful to employ multiple transition steps. Essentially, being able to define an interpolation between two probability distributions is very useful for sampling.  
It turns out that the connection between $$p_{data}(x)$$ and $$p_{easy}(x)$$ can be very naturally defined in terms of noising. If that is the case, going from $$z_{new}$$ to $$x_{new}$$ is intimately linked with a certain untractable quantity, the **Stein Score** of the data distribution. Approximating it is where Deep Learning and neural networks come into play.

Historically, the idea of a Diffusion Model was reached from different perspectives, which sometimes makes the literature hard to navigate.
- *Sohl-Dickstein et al.* <d-cite key="og_diff"></d-cite> is the first paper on the topic, which used a Variational Inference approach.
- *Song and Ermon* <d-cite key="og_song"></d-cite> introduced the connection with Denoising Score Matching, seeing a Diffusion Model through the lenses of Score Based Models.
- *Ho et al.* <d-cite key="Ho"></d-cite> tied together the first two views.
- *Song et al.* <d-cite key="sde"></d-cite> generalised the existing Diffusion Model framework to continuous time processes.

## 1. Defining the Forward Process

We define $$ p(\mathbf{x}_0,\mathbf{x}_1, ... ,\mathbf{x}_T) $$ as a joint probability distribution over $$(\mathbb{R}^d)^T$$ , where the marginal $$p(\mathbf{x}_0)$$ identifies the unknown data distribution and the marginal $$p(\mathbf{x}_T)$$ identifies the easy-to-sample terminal distribution (usually Gaussian).
As any joint probability distribution, we can express it using a Markov model factorisation. By applying the product rule, starting from $$p(\mathbf{x}_0)$$ we obtain the following **forward representation**:
$$\begin{equation} p(\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_T) = p(\mathbf{x}_0) \prod_{t=1}^{T} p(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \end{equation} $$
This factorisation can be visualised using a Probabilistic Graphical Model as follows:
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diff_fwd.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Graphical model for the forward diffusion process. Adapted from <d-cite key="diff_pgm"></d-cite>.
</div>

How do we choose the transition kernel $$p(\mathbf{x}_i \mid \mathbf{x}_{i-1})$$ ? As we are looking for an easy to sample end distribution $$p(\mathbf{x}_T)$$, a reasonable choice could be:
$$ \begin{equation} p(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t ; \sqrt{1 -\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}) \label{forward} \end{equation} $$

where $$\beta_t$$ is a given noise schedule. 
This parametrisation ensures that $$p(\mathbf{x}_t)$$ becomes approximately $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$. Will see later that convergence to a Gaussian is a characteristic feature of all process of this type (Ornstein–Uhlenbeck process, see more in Section 5).

From a sampling perspective, this is equivalent to defining the following forward Markov Chain:

$$ \begin{equation}
\mathbf{x}_t = \sqrt{1 -\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \mathbf{z}_{t-1} \quad \quad \mathbf{z}_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{1}), \quad \forall t
\end{equation}
$$
 
Having the transition kernel designed as such, we have access to a close form expression of $$p(\mathbf{x}_t \mid \mathbf{x}_{0})$$ from which we can easily sample (see below). As will be clear later, this is a crucial property of the model. Allowing immediate access to a noised sample $$\mathbf{x}_t$$ for any $$\mathbf{x}_0$$ means that we do not have to go through the entire Markov Chain, guaranteeing large computational savings.  
To see how this is true, first define:

$$
\begin{equation*}
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i
\end{equation*}
$$

then observe that:

$$ \begin{align*}
\mathbf{x}_t 
&= \sqrt{1 -\beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \mathbf{z}_{t-1} \\
&= \sqrt{\alpha_t} \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t} \mathbf{z}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \mathbf{z}_{t-2} + \sqrt{1 - \alpha_t} \mathbf{z}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \tilde{\mathbf{z}}_{t-2} \\
&= \sqrt{\bar{\alpha}_t} \mathbf{x}_{0} + \sqrt{1 - \bar{\alpha}_t} \tilde{\mathbf{z}}_{0}
\end{align*}
$$

where we used the identity that the sum of two Gaussian random variables $$\mathbf{z}_{1} \sim \mathcal{N}(\mathbf{0}, \sigma_1^2 \mathbf{I})$$ and $$\mathbf{z}_{2} \sim \mathcal{N}(\mathbf{0}, \sigma_2^2 \mathbf{I})$$ gives $$\tilde{\mathbf{z}} \sim \mathcal{N}(\mathbf{0}, \sigma_1^2 +  \sigma_2^2 \mathbf{1})$$.
## 2. Defining the Reverse Process

Equivalently to the forward representation, we can also factorise $$p(\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_T)$$ starting from the end $$\mathbf{x}_T$$ of the forward Markov Chain and proceeding in the **reverse** direction.
$$\begin{equation}
p(\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_T) = p(\mathbf{x}_T) \prod_{t=T}^{1} p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)
\end{equation}
$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diff_rev.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Graphical model for the reverse diffusion process. Adapted from <d-cite key="diff_pgm"></d-cite>.
</div>

The main question that a diffusion model tries to answer is: given a forward representation like the one in $$\eqref{forward}$$ what is the reverse transition kernel $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ ?
Intuitively, this is a hard problem. As we are adding noise in the forward transitions $$p(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$, exact knowledge of the reverse $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ implies being able to perfectly denoise a sample.  
At the same time, even approximate knowledge of $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ would enable us to generate a sample from any arbitrary data distribution $$p(\mathbf{x}_{0})$$ by first starting from the easy to sample $$p(\mathbf{x}_T)$$, and then "hopping" backwards on the reverse Markov Chain until $$p(\mathbf{x}_{0})$$ is reached, in a process known as **Ancestral Sampling** in the Probabilistic Graphical Models literature.

To find an approximation for the reverse transition probability, we first express it as a function of the forward kernel using Bayes' Theorem:
$$
\begin{equation}
p(\mathbf{x}_{t-1}|\mathbf{x}_t) = \frac{p(\mathbf{x}_{t-1}) p(\mathbf{x}_t \mid \mathbf{x}_{t-1})}{p(\mathbf{x}_t)}
\label{bayes}
\end{equation}
$$

while $$p(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$ is known, this formula suffers from the stereoptypical problem of Bayesian inference: $$p(\mathbf{x}_t)$$ and $$p(\mathbf{x}_{t-1})$$ are intractable, as they would require marginalisation over the rest of the other variables.  
Without getting discouraged, in the following we will try to build an intuition for what type of terms to expect in $$p(\mathbf{x}_{t-1}|\mathbf{x}_t)$$.  
We plug in the functional form of $$p(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$ and rearrange as follows:

$$
\begin{align*}
p(\mathbf{x}_{t-1}|\mathbf{x}_t) 
&= \frac{1}{Z} \exp\left(-\frac{\|\mathbf{x}_t - \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}\|^2}{2 \beta_t}\right) \frac{p(\mathbf{x}_{t-1})}{p(\mathbf{x}_t)} \\
&= \frac{1}{Z} \exp\left(-\frac{\|\mathbf{x}_t - \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}\|^2}{2 \beta_t}\right) \exp \left(\log p(\mathbf{x}_{t-1}) - \log p(\mathbf{x}_t) \right)
\end{align*}
$$

In the limit of many noising steps $$T >> 1$$, both the steps difference  $$\|\mathbf{x}_t - \mathbf{x}_{t-1}\|^2$$ and the noise variance $$\beta_t$$ are small. This allows to Taylor expand the difference in the second exponential as follows:

$$ \exp \left(\log p(\mathbf{x}_{t-1}) - \log p(\mathbf{x}_t) \right) = \exp \left( \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \cdot (\mathbf{x}_{t-1} - \mathbf{x}_t) + \mathcal{O}(\beta_t^2) \right)$$

Plugging this back in the expression for $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ yields:

$$
\begin{align*}
p(\mathbf{x}_{t-1}|\mathbf{x}_t) 
&= \frac{1}{Z} \exp\left(-\frac{\|\mathbf{x}_t - \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}\|^2}{2 \beta_t}\right) \exp \left(\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \cdot (\mathbf{x}_{t-1} - \mathbf{x}_t) + \mathcal{O}(\beta_t^2) \right) \\
&= \frac{1}{Z} \exp\left(-\frac{\|\mathbf{x}_t - \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1}\|^2 + 2\beta_t \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \cdot (\mathbf{x}_{t-1} - \mathbf{x}_t) + \mathcal{O}(\beta_t^3)}{2 \beta_t}\right)
\end{align*}
$$

At the numerator we can recognise a quadratic form of $$\mathbf{x}_{t-1}$$, meaning that the whole expression can be approximated as a Gaussian!  
More specifically, we can further manipulate the above expression (assuming that $$\| \log p(\mathbf{x}_{t-1}) - \log p(\mathbf{x}_t) \| = \mathcal{O}(\beta_t^2)$$, completing the square and rearranging) to obtain:

$$ 
\begin{equation}
p(\mathbf{x}_{t-1}|\mathbf{x}_t) 
\approx \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{1 + \beta_t} \, \mathbf{x}_{t} + \beta_t \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t), \beta_t \mathbf{I})
\label{reverse}
\end{equation}
$$

This is an exciting result, as it means that by sampling from the above Gaussian one can effectively reverse the Markov Chain defined in $$\eqref{forward}$$.  
The only problematic term that makes this sampling not trivial is the unknown $$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$$ appearing in the mean of the Gaussian. $$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$$ is known as **Stein Score** in the literature, and intuitively represents the vector field pointing in the direction of highest increase of the log likelihood in sample space $$\mathbb{R}^d$$ (if i am at $$\mathbf{x}_t$$, where should i go to get a more likely sample?).  
But how can we find an estimator for the score of any given marginal $$p(\mathbf{x}_t)$$? To solve this problem we need to apply some techniques from the field of Score Matching.

### Note: Variational approach and equivalence with Score Matching

Using the same starting point as the above derivation (Equation $$\eqref{bayes}$$), we could have also proceeded with a variational approach <d-cite key="og_diff"></d-cite>. Specifically, knowing that $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$ is approximately Gaussian, we could simply parametrise it using a neural network:

$$N(x_{t-1} \mid \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t))
$$

and train the network by maximising an Evidence Lower Bound (ELBO), which is effectively a lower bound on the untractable log likelihood $$\log p(\mathbf{x}_{t-1})$$.  
We can view this setup as learning the "decoder" part of a Variational Autoencoder (VAE), where $$\mathbf{x}_t$$ takes the role of the latent variable $$\mathbf{z}$$. The only difference being that in this case the "encoder" model $$p(\mathbf{x}_{t}|\mathbf{x}_{t-1})$$ is not learnable, as it is just adding noise.
Generalising this intuition to the entire Markov Chain, we can define a Hierarchical VAE which has observable variable $$\mathbf{x}_0$$ and latent variables $$\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_T = \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T$$, and proceed to optimise the ELBO.  
The advantage of this formulation is that, instead of learning an approximation of the score $$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)$$, we are learning a lower bound on the log likelihood, which can be very useful when comparing with other generative models (e.g. if our diffusion model has a higher ELBO then another model's log likelihood, it means that our diffusion model must have higher log likelihood).

A remarkable result by <d-cite key="Ho"></d-cite> shows that the loss function used by the variational approach is equivalent to Score Matching, specifically Denoising Score Matching, which will be presented below.

## 3. Score Matching

We aim to use a function approximator $$\mathbf{s}_\theta(\mathbf{x}): \mathbb{R}^d \to \mathbb{R}^d$$ , which in practice is represented by a Neural Network, to learn the score $$\nabla_\mathbf{x} \log p(\mathbf{x})$$ of the data distribution $$p(\mathbf{x}_0)$$ (or any given noisy marginal $$p(\mathbf{x}_t)$$ of it). This is achieved by training $$\mathbf{s}_\theta(\mathbf{x})$$ to minimise a Mean Squared Error loss given by:

$$\begin{equation}\mathbb{E}_{p(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right\|^2 \right]
\label{ESM}
\end{equation}$$

which is also referred to as Fisher Divergence or **Explicit Score Matching**.  
In its standard form, $$\eqref{ESM}$$ is not very useful because, although the term in brackets can be easily estimated via Monte Carlo, the first term cannot be accessed directly, since only samples are available from $$p(\mathbf{x})$$.  
To make this loss function tractable, two approaches can be considered:

### Denoising Score Matching

Denoising Score Matching (DSM) <d-cite key="vincent11"></d-cite> works by defining the Explicit Score Matching objective for $$q(\tilde{\mathbf{x}})$$, a noisy version of $$p(\mathbf{x})$$ obtained by applying a noising kernel $$q(\tilde{\mathbf{x}} \mid \mathbf{x})$$ through a convolution:

$$q(\tilde{\mathbf{x}}) = \int q(\tilde{\mathbf{x}} \mid \mathbf{x}) p(\mathbf{x}) \, d\mathbf{x}$$

In our specific case, the noisy distribution $$q(\tilde{\mathbf{x}}) = p(\mathbf{x}_t)$$, which is exactly what we are interested in. At the same time, this framework can (and was intended to) be applied more generally to approximate the score $$\nabla_{\mathbf{x}} \log p(\mathbf{x})$$ of any data distribution through a noisy estimate $$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}})$$.

Using clever algebraic manipulations that isolate the constant terms (no dependence on $$\theta$$) and the untractable ones, we obtain (for full derivation see Appendix of <d-cite key="vincent11"></d-cite>):

$$\begin{equation}
\mathbb{E}_{q(\tilde{\mathbf{x}})} \left[ \left\|   
\mathbf{s}_\theta(\tilde{\mathbf{x}}) -
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}) \right\|^2 \right] = \mathbb{E}_{p(\mathbf{x}) q(\tilde{\mathbf{x}} | \mathbf{x})} \left[ \left\|   
\mathbf{s}_\theta(\tilde{\mathbf{x}}) -
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x}) \right\|^2 \right] + \textit{const}
\label{DSM}
\end{equation}$$

The same result can also be obtained using Tweedie's formula (see for example [this blog post](https://alexxthiery.github.io/posts/reverse_and_tweedie/reverse_and_tweedie.html)).

To see a clear connection with denoising, notice how, in the case of a Gaussian noise kernel $$\mathcal{N}(\tilde{\mathbf{x}} \mid \mathbf{x}, \sigma^2 \mathbf{I})$$: 

$$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}} | \mathbf{x}) = -\frac{(\tilde{\mathbf{x}} - \mathbf{x})}{\sigma^2}$$

$$\tilde{\mathbf{x}} = \mathbf{x} + \sigma \mathbf{z} , \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

substituting both of the above in the right hand side of $$\eqref{DSM}$$ yields the following loss function:

$$\begin{equation}
\mathbb{E}_{p(\mathbf{x}) \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \left\|   
\mathbf{s}_\theta(\mathbf{x} + \sigma \mathbf{z})  + \frac{\mathbf{z}}{\sigma} \right\|^2 \right]
\end{equation}$$

which clearly shows how the score approximator is effectively learning to predict the noise that was applied to its input.

To gain further intuition on this connection between learning the score and denoising, one can also look at the Figure below, where a collection of samples $$\mathbf{x}_i \sim p(\mathbf{x})$$ (empirical data density) is represented by Dirac deltas, and the noisy distribution $$q(\tilde{\mathbf{x}})$$ is represented by the continuous line.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/denoising_deltas.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A collection of samples $\mathbf{x}_i \sim p(\mathbf{x})$ (empirical data density), and the noisy distribution $q(\tilde{\mathbf{x}})$ obtained from them. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 13.
</div>

the two following observations can be made with respect to the Figure:
- Learning the score $$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}})$$ means learning a vector field that, if followed, brings to points $$\tilde{\mathbf{x}}_i$$ where $$q(\tilde{\mathbf{x}})$$ has a maximum. 
- At the same time, $$q(\tilde{\mathbf{x}})$$ is constructed by smoothing Diract deltas centered on the samples $$\mathbf{x}_i$$ (noising process), and maximum points of $$q(\tilde{\mathbf{x}})$$ correspond to known samples of $$p(\mathbf{x})$$:

$$\tilde{\mathbf{x}}_i = \mathbf{x}_i$$

which implies that the two above mentioned tasks are essentially the same.

### Implicit Score Matching

An alternative approach is proposed in <d-cite key="hyv05"></d-cite>.  
Expanding the square in $$\eqref{ESM}$$ and realising the last term is constant (can not be optimised over) yields:

$$\begin{equation}\mathbb{E}_{p(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right\|^2 \right] = \mathbb{E}_{p(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) \right\|^2 \right] + 2 \mathbb{E}_{p(\mathbf{x})} \left[ \mathbf{s}_\theta(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right] + const
\label{ism}
\end{equation}$$

Considering the second term, we first apply the log trick $$\nabla p_t(\mathbf{x}) = \nabla \log p_t(\mathbf{x}) \cdot p_t(\mathbf{x})$$, and then the multi dimensional integration by part formula (product rule of divergence), assuming that $$p(\mathbf{x}) \mathbf{s}_\theta(\mathbf{x}) \rightarrow 0$$  at the border of its domain.

$$\begin{align*}
\mathbb{E}_{p(\mathbf{x})} \left[ \mathbf{s}_\theta(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right] 
&=  \int p(\mathbf{x}) \left[ \mathbf{s}_\theta(\mathbf{x}) \cdot \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right] d \mathbf{x} \\
&=  \int  \mathbf{s}_\theta(\mathbf{x}) \cdot \nabla_{\mathbf{x}} p(\mathbf{x})  d \mathbf{x} \\
&= 0 + \int  p(\mathbf{x}) \nabla_{\mathbf{x}} \cdot \mathbf{s}_\theta(\mathbf{x})  d \mathbf{x} \\
&= \mathbb{E}_{p(\mathbf{x})} \big[\nabla_{\mathbf{x}} \cdot \mathbf{s}_\theta(\mathbf{x}) \big]
\end{align*}$$

Plugging this back in Equation $$\eqref{ism}$$, we are left with yet another tractable formulation of the Score Matching objective, known under the name of Implicit Score Matching (ISM).

$$\begin{equation}\mathbb{E}_{p(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \right\|^2 \right] = \mathbb{E}_{p(\mathbf{x})} \left[ \left\| \mathbf{s}_\theta(\mathbf{x}) \right\|^2 \right] + 2 \mathbb{E}_{p(\mathbf{x})} \left[ \nabla_{\mathbf{x}} \cdot  \mathbf{s}_\theta(\mathbf{x}) \right] + \textit{const}
\end{equation}$$

A series of observations can be made:
- The main advantage of ISM over DSM is that it trains $$\mathbf{s}_\theta(\mathbf{x})$$ to approximate the exact score, not the score $$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}})$$ of a noisy version of the original data distribution. Even in the case where $$\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}})$$ was of interest, ISM doesnt not require any knowledge of the transition kernel $$q(\tilde{\mathbf{x}} \mid \mathbf{x})$$.
- A major disadvantage of ISM is the computation of the score's divergence $$\nabla_{\mathbf{x}} \cdot  \mathbf{s}_\theta(\mathbf{x})$$. This is what makes ISM not that convenient in the practical case of dealing with high dimensional data $$\mathbf{x} \in \mathbb{R}^d$$ , where $$d>>1$$. Specifically, computing $$\nabla_{\mathbf{x}} \cdot  \mathbf{s}_\theta(\mathbf{x})$$ with automatic differentiation would require $$d$$ distinct backpropagations (one for each output dimension of the network $$\mathbf{s}_\theta(\mathbf{x})$$). To have a practical intuition for this, see the Figure below.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/score_grad.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Computing the score's divergence requires $d$ backpropagations. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 14.
</div>

To alleviate such computational burden, **Sliced Score Matching**<d-cite key="sliced_SM"></d-cite> applies some intuitions from randomised linear algebra in order to obtain a computationally lighter estimator:

$$\begin{equation}
\mathbb{E}_{p_{\mathbf{v}},p(\mathbf{x})} \left[ \left\| \mathbf{v}^\intercal \mathbf{s}_\theta(\mathbf{x}) \right\|^2  + 2 \mathbf{v}^\intercal \nabla_{\mathbf{x}} \cdot  \mathbf{s}_\theta(\mathbf{x}) \mathbf{v} \right] + \textit{const}
\end{equation}$$

which can be practically viewed both as projecting both $$\mathbf{s}_\theta(\mathbf{x})$$ and $$\nabla_{\mathbf{x}} \log p(\mathbf{x})$$ onto a random direction $$\mathbf{v} \sim p_{\mathbf{v}}$$ before computing ESM, or alternatively as applying Hutchinson’s trace estimation trick<d-cite key="sliced_SM"></d-cite> to $$\nabla_{\mathbf{x}} \cdot  \mathbf{s}_\theta(\mathbf{x}) = tr[\mathbf{J}(\mathbf{s}_\theta(\mathbf{x}))]$$.

A final note is that all the above approaches rely on having a **continuous** probability density function for which the gradient (and consequently the score) exists.

## 4. Score Based Models

Before moving onto the continuous formulation of Diffusion Models, it is worth reviewing another equivalent formulation through the lenses of Score Based Modelling, which historically was the one uncovering the importance of Denoising Score Matching.

Score based models are initially born from the observation that working with the score helps in solving some of the untractability problems of [Energy Based Models](https://en.wikipedia.org/wiki/Energy-based_model) (see Chapter 24 of <d-cite key="pml2Book"></d-cite> for an excellent overview).  
Specifically, given an Energy Based Model of the form:

$$
\begin{equation}
p(\mathbf{x}) = \frac{e^{-E(\mathbf{x})}}{Z}, \quad Z = \int e^{-E_(\mathbf{x})} d\mathbf{x}
\end{equation}
$$

working with the score allows to ignore the computationally intractable normalisation constant $$Z$$:

$$
\begin{align*}
\nabla_{\mathbf{x}} \log p(\mathbf{x}) 
&= -\nabla_{\mathbf{x}} E(\mathbf{x}) - \nabla_{\mathbf{x}} \log Z \\
&= -\nabla_{\mathbf{x}} E(\mathbf{x}) - 0
\end{align*}
$$

The key insight is that learning the score, $ \mathbf{s}_{\theta}(\mathbf{x})$, does not necessarily mean we are learning an Energy-Based Model (where the score is a conservative field). Instead, we are learning a more general generative model described by the score, which might not correspond to any energy function $E(\mathbf{x})$. This broader perspective defines the category of Score-Based Models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/venn2.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Score-based models in the landscape of generative models. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 13.
</div>

Once the score $$\nabla_\mathbf{x} \log p(\mathbf{x})$$ is approximated (see Score Matching Chapter), generation can be achieved by the use of the **Unadjusted Langevin Algorithm**.  
The Unadjusted Langevin Algorithm is a Markov Chain Monte Carlo (MCMC) sampling method that consists in running the following Markov Chain:

$$\begin{equation}
\mathbf{x}_{i+1} = \mathbf{x}_i + \eta \nabla_\mathbf{x} \log p(\mathbf{x}_i) + \sqrt{2\eta} \boldsymbol{z}_i \quad \quad \mathbf{z}_{i} \sim \mathcal{N}(\mathbf{0}, \mathbf{1})
\end{equation}$$

where $$p(\mathbf{x}_i)$$ is proven to converge to $$p(\mathbf{x})$$ for $$i \rightarrow + \infty$$ under reasonable assumptions.  
To gain an intuition on what exactly the Unadjusted Langevin Algorithm is achieving, it can be helpful to view it as a noisy version of simple Gradient Ascent on $$p(\mathbf{x})$$.

When trying to implement the above sampling method, two problems become apparent:
1. Langevin MCMC has a very slow mixing time. This can be intuitively understood by considering a mixture of Gaussians. If the noise term is not carefully tuned, the Markov Chain could be stuck on one of the two modes.  
2. The second and most relevant problem to Machine Learning applications is that the true $$p(\mathbf{x})$$ is inaccessible and needs to be approximated from its samples $$\mathbf{x} \sim p(\mathbf{x})$$ (the dataset). Especially considering the [Manifold Hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis), a $$p(\mathbf{x})$$ describing real world data often does not have support on the entire $$\mathbb{R}^d$$ (some $$\mathbf{x}$$ have zero probability, they are "impossible"). This property of $$p(\mathbf{x})$$, coupled with the fact that a dataset may have scarce data regions, makes the estimation of $$\nabla_\mathbf{x} \log p(\mathbf{x})$$ very inaccurate everywhere but in areas of $$\mathbb{R}^d$$ that are particularly data dense (see top row of the Figure below). Clearly a wrong score estimation leads to incorrect sampling from the Langevin MCMC procedure.

All of the above problems are addressed by learning the score of $$q(\tilde{\mathbf{x}})$$, a noised version of $$p(\mathbf{x})$$, which coincidentally can be obtained through Denoising Score Matching. Intuitively, this is understood by considering the noising process as a "smoothing" operator, which makes $$q(\tilde{\mathbf{x}})$$ easier to work with.  
More specifically, the samples now cover more uniformly $$\mathbb{R}^d$$, and mixing between modes is accelerated as it is "easier" for the Markov Chain to switch between them.  
The only problem becomes that we now have an accurate estimate of the score of $$q(\tilde{\mathbf{x}})$$, which is not $$p(\mathbf{x})$$. A trade-off can be identified where, the more noise is added, the easier a precise estimation becomes, but the further away from the real $$p(\mathbf{x})$$ our estimation $$q(\tilde{\mathbf{x}})$$ is.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/tradeoff_noising.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Illustrating the trade-off between estimation accuracy and data quality. Red encodes error. The bottom row is obtained by applying a noising kernel $q(\tilde{\mathbf{x}} \mid \mathbf{x})$ to the probability distribution of the top row. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 14.
</div>

The paper <d-cite key="og_song"></d-cite> had the brilliant intuition of leveraging this trade-off to define a sampling procedure called **Annealed Langevin Algorithm**.  
The Annealed Langevin Algorithm starts by defining a family $$\{ q_i(\tilde{\mathbf{x}}) \}$$ of noisy versions of $$p(\mathbf{x})$$, each one obtained by applying a noising kernel  $$q(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \, \mathbf{x}, \sigma_i \mathbf{I})$$ with $$\sigma_i$$ of different strength. Sampling from $$q_i(\tilde{\mathbf{x}})$$ can be equivalently seen as starting from $$\mathbf{x_0} \sim p(\mathbf{x})$$ and iteratively applying the following Markov Chain $$i$$ times:

$$
\begin{equation}
 \mathbf{x}_i =  \mathbf{x}_{i-1} + \sqrt{\sigma_i - \sigma_{i-1}} \mathbf{z}_{i-1} \quad \quad \mathbf{z}_{i-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{1}), \quad \sigma_0 = 0
\label{forward_song}
\end{equation}
$$

During training time, the scores $$\nabla_{\tilde{\mathbf{x}}} \log q_i(\tilde{\mathbf{x}})$$ are estimated for all $$i$$ using Denoising Score Matching. In practice, this is achieved by the use of a Noise Conditional Score Network, a score approximator $$\mathbf{s}_\theta(\mathbf{x}, \sigma_i)$$ that also takes as an input the noise level $$\sigma_i$$.  
During the sampling phase, the original Langevin MCMC procedure is initially run for $$K$$ steps using $$\mathbf{s}_\theta(\mathbf{x}, \sigma_{i_{MAX}})$$, the score estimate of the $$q_i(\tilde{\mathbf{x}})$$ with the most noise. The obtained samples are then used to initiate a new Langevin MCMC procedure with $$\mathbf{s}_\theta(\mathbf{x}, \sigma_{i_{MAX-1}})$$, a slightly lower noise level. Iterating this process while slowly decreasing the amount of added noise $$\sigma_i$$ all the way to $$\sigma_1$$, yields samples from the wanted $$p(\mathbf{x})$$. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/adjusted_langevin.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Samples from three consecutive noise levels of the Annealed Langevin MCMC. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 14.
</div>

Running the Annealed Langevin Algorithm with $$K=1$$:

$$
\begin{equation}
\tilde{\mathbf{x}}_{t} = \tilde{\mathbf{x}}_{t-1} + \eta \nabla_\tilde{\mathbf{x}} \log q_{t-1}(\tilde{\mathbf{x}}_{t-1}) + \sqrt{2\eta} \boldsymbol{z}_t \quad \quad \mathbf{z}_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{1})
\label{reverse_song}
\end{equation}
$$

is very similar to the process presented in Section 2, as they both rely on a Markov Chain having an expression that includes a score evaluation. To better understand the similarities between the two, we can leverage the unifying view of Continuous Time Diffusion Models.

## 5. Continuous Time View

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/sde_fwd.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Continuous view of the forward diffusion process. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 16.
</div>

The main contribution of <d-cite key="sde"></d-cite> is the expansion of the discrete diffusion framework presented above to the continuous setting.

As a high number of steps has proved to be helpful in going from $$p(\mathbf{x}_0)$$ to $$p(\mathbf{x}_T)$$, a natural question is if it could be possible to bring the step number to infinity.  
This can be done by controlling the noising dynamics through a Stochastic Differential Equation (SDE), which can be written (using Ito notation) as:

$$
\begin{equation}
d \mathbf{x}_t = f(\mathbf{x}_t, t)dt + g(t)d\mathbf{w}_t
\label{ito}
\end{equation}
$$

where $$\mathbf{w}_t$$ represents a standard [Wiener Process](https://en.wikipedia.org/wiki/Wiener_process), also called Brownian Motion.

Specifically, the Markov Chains defining the forward noising processes (both in Section 1 and 4) corresponds to the discretisation of two well-known SDEs: the Ornstein–Uhlenbeck SDE, and the Brownian Motion SDE. The discretisations can be obtained using a stochastic generalisation of the Euler method called [Euler–Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method).

- The Ornstein–Uhlenbeck (OU) SDE:

  $$
  \begin{equation}
  d \mathbf{x}_t = - \frac{\beta(t)}{2} \mathbf{x}_tdt + \sqrt{\beta(t)} d\mathbf{w}_t
  \end{equation}
  $$

  which, when discretised, yields Equation $$\eqref{forward}$$ from Section 1 (where the square root term in $$\eqref{forward}$$ is Taylor expanded).
  An important property of the OU SDE is that its solution converges exponentially fast to having $$\mathcal{Law}(\mathbf{x}_t) = \mathcal{N}(\mathbf{0}, c \mathbf{I})$$ with fixed variance $$c$$. Because of this reason it is also referred to as Variance Preserving (VP) SDE.
- The Brownian Motion SDE:

  $$
  \begin{equation}
  d \mathbf{x}_t =  \sqrt{\beta(t)} d\mathbf{w}_t
  \end{equation}
  $$
  
  which, when discretised, yields Equation $$\eqref{forward_song}$$ from Section 4.
	Its solution also has Gaussian law, but with the variance going to infinity. Because of this reason it is also referred to as Variance Exploding (VE) SDE.

To obtain samples we need to define a reverse process going from $$p(\mathbf{x}_T)$$ to $$p(\mathbf{x}_0)$$.  
A remarkable result in SDE theory, which can be traced back to <d-cite key="anderson"></d-cite> <d-cite key="altrosde"></d-cite>, tells us that for any SDE of the form $$\eqref{ito}$$, a **Reverse SDE** can be defined as:

$$
\begin{equation}
 d \mathbf{x}_t = \big[ f(\mathbf{x}_t, t) -  g(t)^2\nabla_{\mathbf{x}} \log p(\mathbf{x}_t) \big]dt+ g(t)d\tilde{\mathbf{w}_t}
\end{equation}
$$

where $$dt$$ is a **negative** time increment, and $$\tilde{\mathbf{w}}_t$$ is the time reversed Brownian Motion.  
Intuition about the above reverse SDE can be gained by looking at the two drift contributions: $$f(\mathbf{x}_t, t)dt$$ is the same drift found in the forward equation, but applied in the opposite direction, and $$\nabla_{\mathbf{x}} \log p(\mathbf{x}_t)dt$$ is an additional drift term that reverses the diffusion process (concentrates the probability mass instead of spreading it).

Applying this formula to the two SDEs defined above yields:

$$
\begin{equation}
d \mathbf{x}_t = \big[ - \frac{\beta(t)}{2} \mathbf{x}_t - \beta(t)\nabla_{\mathbf{x}} \log p(\mathbf{x}_t) \big]dt+ \sqrt{\beta(t)}d\tilde{\mathbf{w}_t}
\end{equation}
$$

$$
\begin{equation}
d \mathbf{x}_t = \big[ -\beta(t) \nabla_{\mathbf{x}} \log p(\mathbf{x}_t) \big]dt+ \sqrt{\beta(t)}d\tilde{\mathbf{w}_t}
\end{equation}
$$

which bear a lot of similarities to Equations $$\eqref{reverse}$$ and $$\eqref{reverse_song}$$, justifying the SDE view of Diffusion Models as a natural generalisation of the previous results.

The main advantages of a continuous framework are as follows:

- there is a lot of flexibility in the definition of the reverse process, intended as a solution of the reverse SDE. Specifically, parameters like discretisation step size and     discretisation method can be tuned to have better control over the accuracy versus speed trade-off. An interesting approach to solving the reverse SDE is given by the Predictor-Corrector method.  
The **Predictor-Corrector method** consists in leveraging information on the score to improve the accuracy of regular SDE solvers. It works by first performing a Predictor step, which approximates the value of $$\mathbf{x}_{T-\Delta t}$$ using an SDE solver (e.g. Euler-Maruyama method). Subsequently, during the Corrector step, a Langevin MCMC procedure is run using score information $$\mathbf{s}_\theta(\mathbf{x}_{T- \Delta t}, T-\Delta t) \approx \nabla_{\mathbf{x}_{T- \Delta t}} \log p(\mathbf{x}_{T- \Delta t})$$, improving the original sample $$\mathbf{x}_{T-\Delta t}$$. Iterating these two steps allows to solve the reverse SDE achieving greater precision than a purely SDE solver approach, while also speeding up sampling compared to a purely Langevin MCMC approach like the one presented in Section 4.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/predictor_corrector.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Predictor-Corrector method. White arrows indicate the SDE solver, while red arrows indicate the Langevin MCMC correction. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 16.
</div>

- As the reverse SDE result holds for **any stochastic process**, we effectively have access to a more general framework that can be applied to any noising process of choice. This being said, it is important to consider that, to have a model that is computationally feasible to train, a closed form of $$p(\mathbf{x}_t \mid \mathbf{x}_0)$$ is essential to avoid simulating the entire forward process up to $$\mathbf{x}_t$$ with Euler-Maruyama (or any other SDE solver)
- Choosing a OU SDE, we have theoretical guarantees that the process converges to a Gaussian distribution exponentially fast.
- As can be seen in the following Section, the continuous view allows for an ODE interpretation of the mapping between $$p(\mathbf{x_0})$$ and $$p(\mathbf{x}_T)$$, which has several advantages.

## 6. Equivalence to Ordinary Differential Equation (ODE) formulation

Another remarkable contribution of <d-cite key="sde"></d-cite> is the observation that the forward/reverse SDEs can be reformulate in a deterministic way by describing the evolution of probability density.  
This result can be achieved by writing the [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation) (also known as Kolmogorov Forward Equation) associated with the forward/reverse SDE, and performing some simple algebraic manipulations to obtain another SDE without noise term (an ODE!), which takes the name of Probability Flow ODE in <d-cite key="sde"></d-cite>.  
The crucial point of this derivation is that, as both equations satisfy the same Fokker-Planck Equation, their marginals $$p(\mathbf{x}_t)$$ are the same.

We start off by considering a given SDE

$$ d \mathbf{x} = f(\mathbf{x}, t)dt + g(t)d \mathbf{w}$$

and writing its related Fokker-Planck equation:

$$
\begin{align*}
\frac{\partial p_t(\mathbf{x})}{\partial t} &= -\nabla \cdot \big( f(\mathbf{x}, t) p_t(\mathbf{x}) \big) + \frac{1}{2} g(t)^2 \nabla^2 p_t(\mathbf{x}) \\
&= - \nabla \cdot \big[ f(\mathbf{x}, t) p_t(\mathbf{x}) - \frac{1}{2} g(t)^2 \nabla p_t(\mathbf{x}) \big]
\end{align*}
$$

Using the log trick $$\nabla p_t(\mathbf{x}) = \nabla \log p_t(\mathbf{x}) \cdot p_t(\mathbf{x})$$, substituting in the diffusion term and factoring $$p_t(\mathbf{x})$$ out yields :

$$
\begin{equation}
\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot \big( h(\mathbf{x}, t) p_t(\mathbf{x}) \big)
\end{equation}
$$

where $$h(\mathbf{x}, t) =  f(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(\mathbf{x})$$.  

This is an example of Liouville equation, a special case of the Fokker-Planck equation describing a process that is only driven by a deterministic term. Reverting from the Fokker-Planck to the SDE view of the process we get: 

$$\begin{align}
d \mathbf{x} &= h(\mathbf{x}, t)dt \nonumber \\
			&= \big[ f(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(\mathbf{x}) \big]dt \\
\end{align}$$

For the more general case (SDEs with noise coefficients given by the matrix $$\mathbf{G}(\mathbf{x}, t)$$) please see <d-cite key="sde"></d-cite> Appendix A.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/ode_sde.PNG" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    ODE and SDE formulations of a Diffusion Model. The two processes are different but have the same marginal distributions. Adapted from <a href="https://deepgenerativemodels.github.io/syllabus.html" target="_blank">CS236</a> Lecture 16.
</div>

It is important to note that the SDE and ODE formulations describe two **different** processes. As both these processes solve the same Fokker-Planck equation, their marginal densities are the same, which is ultimately what we are interested in modelling.

The ability to have an ODE formulation has several key advantages:
- The mapping between points $$\mathbf{x}_0$$ and $$\mathbf{x}_T$$ has now become fully deterministic, meaning that the above equations describe a Continuous Normalising Flow!
	- As likelihoods are tractable in Normalising Flows, we can have **access to the model likelihoods**. Even thought this is great, training through score matching is still preferred as likelihood computations are very slow compared to it (would need to solve the ODE on the full time length $$0$$ to $$T$$ for every datapoint)
- Regular ODE solvers can be used to perform the sampling process. This is an advantageous property as existing literature on ODE solvers can be leveraged to control the trade-off between speed and accuracy
- Considering that the Forward ODE does not have any learnable parameters, and that there is an invertible deterministic mapping between $$p(\mathbf{x}_0)$$ and $$p(\mathbf{x}_T)$$, each data point $$\mathbf{x}_0$$ has a latent encoding $$\mathbf{x}_T$$ which is uniquely identifiable. This means that, as long as the dataset size, model capacity and optimisation accuracy are enough, the latent encoding depends only on $$p(\mathbf{x}_0)$$, and is agnostic of model/optimiser/dataset choice. This is a remarkable property that is not shared by many other generative models (think about the fact that in a VAE the encoding is a learned process).
- Having a latent encoding also enables interpolation in latent space, similarly to other generative models.

## 7. Conditional Generation and Inverse Problems

So far we have looked at generating a sample from a probability distribution $$p(\mathbf{x})$$, but in real world applications we are often interested in sampling from the conditional distribution $$p(\mathbf{x} \mid \mathbf{c})$$. An example is _text-to-image_ models, which generate an image $$\mathbf{x}$$ given a text prompt $$\mathbf{c}$$.

This is particularly interesting as it formalises a wide range of inverse problems, which are commonly found in science and engineering, where the goal is to infer the underlying cause $$\mathbf{x}$$ from some observed low dimensional (potentially noisy) data $$\mathbf{c}$$.

The easiest way to achieve a conditional diffusion model is to perform training on a labelled dataset of elements $$(\mathbf{x}, \mathbf{c})$$, and approximate the true conditional score $$\nabla_{\mathbf{x}} \log p(\mathbf{x|\mathbf{c}})$$ with a learned score function of the form $$\mathbf{s}_\theta(\mathbf{x}, t, \mathbf{c})$$, where the input $$\mathbf{c}$$ is usually given by an embedding of the condition.  
There are a couple of problems with this naive approach. The first one is that we need to retrain the entire model for different types of conditions $$\mathbf{c}_1$$ and  $$\mathbf{c}_2$$, and we are not leveraging the shared structure of $$p(\mathbf{x})$$.  
The second and more subtle problem is that this approach does not allow for a way to trade off sample variance with sample quality, which is a feature that has great usefulness in many applications.

To propose a solution to both problems, <d-cite key="class_guided"></d-cite> presents the idea of **classifier guidance**.  
At its core, classifier guidance consists in observing that, by applying Bayes' theorem to the conditional distribution
$$p(\mathbf{x} \mid \mathbf{c}) = \frac{p(\mathbf{x}) p(\mathbf{c} \mid \mathbf{x})}{p(\mathbf{c})}$$
and by taking the score 
$$\nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \nabla_{\mathbf{x}} \log p(\mathbf{c} \mid \mathbf{x}) - \nabla_{\mathbf{x}} \log p(\mathbf{c})$$
the marginal $$p(\mathbf{c})$$ gets canceled out.  
This leaves us with:

$$
\begin{equation}
\nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \nabla_{\mathbf{x}} \log p(\mathbf{c} \mid \mathbf{x})
\end{equation}$$

The intuition in the above formula is that the second term "pushes" the sampling process towards areas of high probability $$p(\mathbf{c} \mid \mathbf{x})$$, effectively introducing an additional drift term in the reverse SDE. Classifier guidance can be further enhanced into:

$$
\begin{equation}
\nabla_{\mathbf{x}} \log p_w(\mathbf{x} \mid \mathbf{c}) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + (w+1) \nabla_{\mathbf{x}} \log p(\mathbf{c} \mid \mathbf{x})
\label{clas_guid}
\end{equation}$$

where $$(w+1)$$ is an additional factor which controls the strength of the guidance. By varying the value of $$w$$, control over the above mentioned trade-off between sample variance and sample quality can be achieved.  
A big advantage of this approach is that, given a pretrained classifier, the second term can be easily calculated by taking the gradient of its output $$\mathbf{c}$$ with respect to the input $$\mathbf{x}$$, avoiding the need to retrain the diffusion model from scratch. In practice, as we need the scores at any noise level $$t$$, the classifier must also be trained on noisy versions of the dataset, meaning that a regular classifier would not work "out of the box".  
One of the drawbacks of classifier guidance lies in its high reliance on the classifier gradients, which are generally known to not exactly rely on all the core features of a given class. Similarly to an adversarial attack, the sampling process could prioritise directions that maximise $$p(\mathbf{c}|\mathbf{x})$$ but do not correspond to very representative samples from the class $$\mathbf{c}$$.  
An alternative to classifier guidance that tackles some of these problems is given by **classifier-free guidance** <d-cite key="class_free"></d-cite>, whose core idea is to substitute the classifier's guidance by the difference between two diffusion models, a conditional one and an unconditional one.  
Starting from equation $$\eqref{clas_guid}$$ and applying Bayes' Theorem to the classifier term yields:

$$\begin{align}
\nabla_{\mathbf{x}} \log p_w(\mathbf{x}|\mathbf{c}) 
&= \nabla_{\mathbf{x}} \log p(\mathbf{x}) + (w+1) \big[ \nabla_{\mathbf{x}} \log p(\mathbf{x}|\mathbf{c}) - \nabla_{\mathbf{x}} \log p(\mathbf{x}) \big] \nonumber\\
&= (w+1) \nabla_{\mathbf{x}} \log p(\mathbf{x}|\mathbf{c}) - w \nabla_{\mathbf{x}} \log p(\mathbf{x})
\end{align}
$$

This alternative view of equation $$\eqref{clas_guid}$$ shows that the additional weighting factor is equivalent to having a score that simultaneously pushes generation towards areas that have high $$p_w(\mathbf{x}|\mathbf{c})$$, and away from areas that would otherwise be likely samples from $$p_w(\mathbf{x})$$.  
The same effect of classifier guidance can then be achieved by taking the difference between the conditional and unconditional models, which in practice are both obtained by training a single score function $$\mathbf{s}_\theta(\mathbf{x}, t, \mathbf{c})$$, where sometimes the last input is left empty $$\mathbf{s}_\theta(\mathbf{x}, t, \emptyset)$$.

## 9. Sources
In addition to the cited papers, the sources that have greatly helped me navigate the literature are the following:
- "Introduction to Diffusion Models" [lecture](https://www.youtube.com/watch?v=tNcDcF8J_1Y) and [slides](https://vdeborto.github.io/project/generative_modeling/) by Valentin de Bortoli.
- "Diffusion Models" and "Energy Based Models" chapters in [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) by Kevin Murphy.
- Two blog posts by [Yang Song](https://yang-song.net/blog/2021/score/) and [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
- Stanford CS236 [lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8) and [slides](https://deepgenerativemodels.github.io/syllabus.html) by Stefano Ermon.
- "SDEs and Diffusion Models" [course](https://sde-course.netlify.app/) by Kieran Didi and Francisco Vargas.