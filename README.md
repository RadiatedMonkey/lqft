# LQFT

This repository contains a high performance Rust lattice QFT simulator on a Euclidean lattice. The main use of this project is to learn about Monte Carlo simulations, Markov chain stochastic processes, advanced statistics and high performance computing. 

*Note*:
This project currently requires nightly Rust to build due to usage of the [`portable_simd`](https://doc.rust-lang.org/std/simd/index.html) feature. There are no guarantees on an MSRV.

### Features
- Metropolis-Hastings algorithm on a checkerboarded lattice to perform updates.
- Supports $n$-dimensional arbitrarily sized lattices.
- Configurable floating point precision and SIMD lane count.
- Custom initial lattice states: load from an HDF5 snapshot, perform a cold start or generate a random hot start.
- Easily programmable observable measurements.

### Method
Currently only scalar $\phi^4$ theory is supported. The used action is the Euclidean discretised version of the $\phi^4$ action
$$
	S = \int \mathrm{d}^4 x \,\left(\frac{1}{2}\partial^\mu \phi \partial_\mu \phi - \frac{1}{2}m^2 \phi^2 - \frac{\lambda}{4!}\phi^4\right)
$$
which is 
$$
		S = a^4 \sum_n \left[\frac{1}{2} \sum_{\mu = 1}^4 \left(\frac{\phi_{n + a \hat{\mu}} - \phi_n}{a}\right)^2 + \frac{1}{2}m^2 \phi_n^2 + \frac{\lambda}{4!} \phi_n^4\right]
$$
Every sweep (a full iteration over the entire lattice), the system first iterates over all the even (red) sites and then over the odd (black sites). The currently selected site gets assigned a random new value and the change in action is computed. This new value is then accepted with a probability
$$
	P = \mathrm{min}(1, e^{-\Delta S})
$$

### Tech setup
The simulation loop has been optimised with AVX-256 SIMD instructions and a multi-threaded iterator. Optimising the data layout for cache locality and adding vectorised instructions reduced the sweep time of a 40 x 20 x 20 x 20 double-valued lattice down from around 15 ms to 3 ms on my Ryzen 5 3600.

The lattice is divided into odd and even (black and red). As red sites do not interact with other red sites (due to the nearest neighbour interaction), they can be updated simultaneously without race conditions. Furthermore, these two colours are stored separately in a linear way to ensure that the data is contiguous, improving cache locality. 

Further planned improvements include simulating using CUDA, hybrid Monte Carlo, gauge theories and fermions.

Metrics and logs are collected using [Prometheus](https://prometheus.io/) and [Loki](https://grafana.com/docs/loki/latest/). These can be displayed in a dashboard in a locally hosted Grafana instance. The metrics are not used to compute any observables, they are purely a visualisation tool.

