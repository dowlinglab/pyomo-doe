# Quantized observation likelihood and residual checks

Sources inspected October 25, 2026. No source PDF bytes were downloaded or hashed.

[Dunn and Smyth, 1996, author preprint dated April 24](https://gksmyth.github.io/pubs/residual.pdf),
section 3 and the discussion, defines Gaussian quantile residuals through the
fitted CDF, randomizes within each observed probability mass for discrete data,
and discusses extending to dependent data through conditional distributions.
Estimated parameters and model selection affect calibration; an attractive QQ
plot does not establish independence or correct physical dynamics. Our filter
uses the predictive CDF conditional on earlier quantized readings and repeats
randomizations with recorded seeds. This is not unstructured jitter on temperature.

[Cedeño et al., 2021, arXiv v1](https://arxiv.org/abs/2112.07828),
abstract inspected, describes quantized state-space estimation by several filters,
including a Gaussian-sum approach based on a bin-integrated observation law and
Gauss-Legendre quadrature. Only the abstract was inspected here, so its numerical
performance is not used as evidence for our implementation. Our scalar OU
posterior grid is a separately derived quadrature implementation, not a claimed
reproduction of that paper. An attempted 2023 publisher review page returned
HTTP 429; no methods claim is based on the inaccessible page.

For a latent analogue value m(t)+X(t)+v(t), the recorded code specifies an
interval. Its conditional probability is the integral of the analogue density
over that interval. Our code evaluates Gaussian CDF differences analytically,
propagates scalar OU latent states, and integrates each posterior by deterministic
quadrature. Full correlated likelihood is a product of conditional bin
probabilities; a product of marginal bins would discard dependence. Node-count
convergence, posterior normalization and independent low-dimensional Gaussian
rectangle integrals are required before interpreting hardware results.

The observed 0.3223 C shifted grid is established by the hardware-data audit;
ADC thresholds, analogue-noise variance and acquisition firmware are not known.
Nearest-code and floor-code bin phases remain explicit sensitivities. The
emulator's unshifted floor and prescribed noise variance are not used.
