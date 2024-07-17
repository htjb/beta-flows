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
in statistical mechanics...

It has been shown that normalising flows can be used to emulate posteriors, priors
and likelihood functions and that researchers can take advantage of these emulators
in a number of different ways.

Typically normalising are trained on the $\beta=1$ posterior distribution or
the $\beta=0$ prior distribution.

The inovation behind $\beta$-flows is to train the flow conditioned on $\beta$.



# Citation

Please cite this code base if you use Beta-flows in your work.


# Contribution

