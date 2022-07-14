+++
author = "Kunal Ghosh"
title = "GSoC 2022 : Fast Gaussian process implementation in PyMC"
date = "2022-07-14"
description = "Running Gaussian processes on GPUs akin to GPytorch"
tags = [
    "gp",
    "gsoc",
    "pymc",
    "data-science",
    "open-source",
]
favourite = true
math = true
+++

Gaussian processes (GPs) are very useful class of `semi-parametric` machine learning models. 
Before their use in more modern classification and regression tasks, 
they have been very successfully applied in searching for underground oil fields. 
GPs were called <cite>__kriging models__ [^1]</cite> back then, but the idea was the same.

[^1]: [Wikipedia: Kriging](https://en.wikipedia.org/wiki/Kriging)

GPs belong to a general class of models known as <cite> __kernel methods__ [^2] </cite>. 
Kernel methods use something called the __kernel function__, denoted as  
$k(\bold{x},\bold{x'})$. Where $\bold{x} \in R^{d}$ represents the input data 
and $k$ can be any function which returns a scalar. 
For example, the `dot-product kernel` $$ k(\bold{x}, \bold{x'}) \coloneqq \bold{x}^T\bold{x} $$ 

[^2]: An overview of kernel methods is out of scope of this post, but a good overview of Gaussian processes can be found in [Rassmussen and Williams](https://gaussianprocess.org/gpml/).

Assuming $N$ such vectors $\bold{x} \in R^{d}$ are stacked, then we can write the input data as 
$X \in R^{NxD}$ and correspondingly the kernel is written as $K_{X,X} \in R^{NxN}$. 
Let's say, we are interested in building a regression model where the target values 
are denoted as $\bold{y} \in R^{n}$ then gaussian process models are trained by optimizing 
something called the <cite>__log marginal likelihood__ [^3]</cite>. 

[^3]: [Equation 2.30](https://gaussianprocess.org/gpml/chapters/RW2.pdf) in _Gaussian Processes for Machine Learning_ gives the log marginal likelihood for a zero-mean Gaussian process.

Log marginal likelihood is a function of the input $X, \bold{y}$ and is written as, 

\begin{equation}
  \tag{1}
  L(\theta | X, \bold{y}) \approx log \left| K_{X,X}\right| - \bold{y}^{T}K_{X,X}^{-1}\bold{y}
\end{equation}

If we want to optimize the above function using gradient based methods we need to 
compute the gradient $ \frac{dL}{d\theta} $ which looks like, 

\begin{equation}
  \tag{2}
  \frac{dL}{d\theta} = \bold{y}^{T} K_{X,X}^{-1} \frac{dK_{X,X}}{d\theta} K_{X,X}^{-1}\bold{y} + \text{Tr} \left( K_{X,X}^{-1} \frac{dK_{X,X}}{d\theta} \right)
\end{equation}

> In equation 1 and 2 above, the most expensive compute steps are 
> 1. The log determinant : $ log \left| K_{X,X}\right| $ 
> 2. Inverse of the kernel or compute the `solve` : $ K_{X,X}^{-1}\bold{y} $
> 3. Trace : $ \text{Tr} \left( K_{X,X}^{-1} \frac{dK_{X,X}}{d\theta} \right)  $ 

In <cite> Gardner, et.al 2018 [^4] </cite> they proposed a few algorithms that expresses each of the above three expensive
computations to large matrix computations which can be sped-up when running on a GPU.

> For my GSoC, I will implement a sub-class the [MarginalGP](https://github.com/pymc-devs/pymc/blob/562be3781c9d37d3300c4efd4cf6598e5739c32d/pymc/gp/gp.py#L358)
> and override the `_build_conditional()` and `_build_marginal_likelihood()` as prescribed in <cite> Gardner, et.al 2018 [^4] </cite> and that should significantly
> speed up Gaussian process inference in PyMC :heart_eyes:

[^4]: GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration. [arxiv](http://arxiv.org/abs/1809.11165)

