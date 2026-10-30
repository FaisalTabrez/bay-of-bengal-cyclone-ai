# Mathematical contract

The symbols below describe the implemented method. Native intensity units are knots and hPa; track vectors are kilometres. Main-paper equations are exported directly into LaTeX and as high-resolution mathematical figures in Word.

For each observed input, `z_j = (x_j - mu_j) / s_j`. Missing values have `z_j = 0` and `m_j = 1`; observed values have `m_j = 0`. Inputs concatenate `z` and `m`. Here `s_j` is the training sample standard deviation unless at most 1e-8, in which case it is one.

Let `y_{t+6}` contain the spherical log-map east/north displacement of the future position and its native future wind/pressure. Let `a_t` contain the mapped constant-velocity reference displacement and persistence/training-mean intensity anchors. Let `S` be the diagonal matrix of training residual scales.

```math
r_t = S^{-1}(y_{t+6}-a_t),\qquad \widehat y_{t+6} = a_t + S f_\theta(x_{\leq t}).
```

The common track scale is `max(sqrt(mean over training rows and both axes of (y-a)^2)), 1 km)`. Each intensity scale is its observed-label residual RMS with a one-native-unit floor. No residual mean subtraction occurs. Track components of the reconstructed vector require the spherical exponential map, not direct addition to degrees.

For active tasks `T = {track, wind, pressure}` and the observed prediction/target entries `O_q` of each task:

```math
L = |T|^{-1} \sum_{q\in T} |O_q|^{-1}
    \sum_{(i,j)\in O_q} \ell_q(\widehat r_{ij}-r_{ij}).
```

Tasks without observed labels in a batch are omitted. Track combines its two observed axes. Original losses are SmoothL1 with beta one; the bounded follow-up substitutes absolute error only for wind. Index out missing entries before arithmetic. The loss averages task-specific errors, not all valid labels in a single pool.

The deterministic checkpoint/family score uses paired native-unit references:

```math
J = (T/T_cv + W/W_pers + P/P_pers)/3.
```

Here `T` is mean great-circle track error, and `W`/`P` are intensity MAEs over the same persistence-eligible rows as their denominators. The two intensity tasks can have different eligible populations. Family score is the mean of three individual-seed scores. The ensemble's own score is not the family-selection statistic.

Let `e_E, e_N` be true minus mean tangent-plane displacement, with standard deviations `sigma_E,sigma_N`, correlation `rho`, and standardised errors `z_E,z_N`:

```math
q = (z_E^2 - 2 rho z_E z_N + z_N^2)/(1-rho^2),
NLL_track = log(2 pi) + log(sigma_E sigma_N)
            + 0.5 log(1-rho^2) + 0.5 q.
```

Each univariate intensity uses `0.5 log(2 pi) + log(sigma) + 0.5 (e/sigma)^2`. Training uses scale-normalised errors and scales; reporting uses the declared native-unit density. The omitted transformations are fixed unit constants, not evidence of a latitude-longitude density.

For fixed means and correlation, the unbounded maximum-likelihood spread factors are:

```math
alpha_track = sqrt(mean(q)/2),
alpha_intensity = sqrt(mean((e/sigma)^2)).
```

Factors are clipped to the predeclared interval [0.25,4]. Track covariance becomes `alpha_track^2 Sigma`; means and correlation do not change. Factors fit on calibration storms do not constitute independent performance evidence.

The normaliser rebase from NIO to Bay statistics changes the first linear layer without changing its function before fine-tuning. For value weights W, mask weights M, bias c, `a=s_Bay/s_NIO` and `b=(mu_Bay-mu_NIO)/s_NIO`:

```math
W_new = W_old * a,
M_new = M_old - W_old * b,
c_new = c_old + W_old b.
```

Multiplication by a and b in the first two expressions is columnwise. Retain the NIO output scales/fallbacks and reset the optimiser. This algebra preserves the pretrained function including missing values; it does not demonstrate a transfer benefit.
