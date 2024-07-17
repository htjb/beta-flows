# $\beta$-flows: Learning with the whole Nested Sampling run

|bflows | Temperature dependent Normalising Flows |
|---------|-----------|
|Author | Harry Bevins|
|Homepage | |

UNDER CONSTRUCTION

$\beta$-flows are a density estimation tool that were intially designed
to improve the accuracy of more traditional normalising flows.

They are built using conditional masked autoregressive flows and utilise the
thermodynamic description of nested sampling during training. The structure
of the $\beta$-flows class and the interface are largely based on
the code [margarine](https://github.com/htjb/margarine).

The nested sampling algorithm is used to perform Bayesian inference and estimate
the normalising factor on Bayes theorem known as the evidence

$$\mathcal{Z} = \int \mathcal{L}(\theta)\pi(\theta) d\theta$$

where $\mathcal{L}(\theta)$ is the postulated probability, known as the
likelihood, of the data given a set of parameters $\theta$ and $\pi(\theta)$
is the prior on the parameters. In the process of estimating the evidence the
nested sampling algorithm also returns samples on the posterior 
probability $P(\theta) = P(\theta|D, M)$.

An analogy can be made between the Bayesian evidence and the partition function
in statistical mechanics. If one writes the energy of a given state as
$E(\theta) = - \log \mathcal{L}(\theta)$ then the partition function can be 
written as

$$\mathcal{Z}(\beta) = \int e^{-\beta E} g(E) dE = \int \mathcal{L}(\theta)^\beta \pi(\theta)
d\theta$$

where $g(E)$ is the density of states with a given energy and $\beta$ is an inverse
temperature. It can be seen that when $\beta=1$ we recover the integral over the posterior
and when $\beta=0$ the integral is over the prior.

We can therefore track over the nested sampling algorihtm transforms the prior
into the posterior by varying $\beta$ between 0 and 1. Further we can
generate a set of samples from any distribution between the prior and posterior
by reweighting the returned posterior points according to

$$p_i^\beta = \frac{w_i \mathcal{L}_i^\beta} {\mathcal{Z}(\beta)}$$

where $w_i$ corresponds to the prior volume contraction at each iteration in the
run.

It has been shown that normalising flows can be used to emulate posteriors, priors
and likelihood functions and that researchers can take advantage of these emulators
in a number of different ways.

Typically normalising are trained on the $\beta=1$ posterior distribution or
the $\beta=0$ prior distribution. When learning the posterior researchers often
find that the flows perform poorly in the tails of the distribution.

The inovation behind $\beta$-flows is to train the flow conditioned on $\beta$ using 
appropriately weighted samples from a nested sampling run.

This has been seen to improve the accuracy of emulation of the $\beta=1$
distribution and has many other applications.

## Example



## Citation

Please cite this code base if you use Beta-flows in your work.

```bibtex
@article{lsbi,
    year  = {2023},
    author = {Will Handley et al},
    title = {lsbi: Linear Simulation Based Inference},
    journal = {In preparation}
}
```

## Contribution

Contributions are welcome. Please open up an issue to discuss.

