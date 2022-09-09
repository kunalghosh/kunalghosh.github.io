+++
author = "Kunal Ghosh"
title = "GSoC 2022 : Final Report"
date = "2022-09-9"
description = "Final progres report of my Google summer of code project"
tags = [
    "gsoc",
    "pymc",
    "data-science",
    "open-source",
]
favourite = true
math = true
comments = true
+++

## GSoC ends but the contribution continues

This GSoC has been a great experience for me. Partly because of the technical aspects of the project,
I loved coding the algorithms and contributing to an open-source project felt refreshing ! 
But most importantly because of my mentors [Chris Fonnesbeck](https://github.com/fonnesbeck) and [Bill Engels](https://github.com/bwengals). 
Their feedback was timely and precise, thank you ! I also found the PyMC communitiny to be very helpful, considerate and welcoming ! 
I plan to continue working on my project and beyond, and can highly recomment this project to future contributors.

# Status update

The objective of my Google summer of code project was to speed up Gaussian process inference in PyMC by 
leveraging Fast Matrix-Matrix multiplication based algorithms to speed up the 
`log-determinant` and `solve` computations which are the most expensive computations during GP inference.

The project was mainly divided into the the following parts:  

  * [x] Familiarise myself with PyMC Codebase  
  * [x] Come up with a plan to implement the algorithms
  * [x] Finish initial implementations in NumPy of:
  * [x] pivoted Cholesky Algorithm.
  * [x] Linear Conjugate Gradients, which works on batches of data.
  * [x] Check-in above two implementations.
  * [ ] Change the algorithms to use JAX instead, check GPU utilisation.
  * [ ] Implement log-determinant and solve functions.
  * [ ] Implement a GP using the GPU accelerated log-determinant and solve functions.
  * [ ] Convert Log-determinant and solve functions to Aesara OPs.
  * [ ] Integrate code into PyMC

The list above also indicates the status of the individual milestones. I plan to continue working on the unfinished items after GSoC.

# Contributions

As a part of my GSoC, I sent two pull-requests which will be merged soon [PR:62](https://github.com/pymc-devs/pymc-experimental/pull/62) and [PR:63](https://github.com/pymc-devs/pymc-experimental/pull/63).
These two PRs implement the core logic of pivoted Cholesky algorithm and the batched linear conjugate gradients, in NumPy. Apart from forming a good starting point for moving on to JAX for GPU speed-up.
The NumPy impementations are also readable reference implementations for someone looking to working on something similar in the future.

I also maintained a log of my work on [HackMD](https://hackmd.io/@CblWjfoIRO2tmCH8-j2AZA/HJTP7aPO9) and wrote a few blog posts [here](https://kunalghosh.github.io/tags/gsoc/).
I really like the process of maintaining a work log, it helps to give a sense of progres to the project and also helps to 
keep track of hacks and work-arounds which one tends to forget. I can highly recommend keeping a digital work diary for your projects.


