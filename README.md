# Hypoexponential

Probability density (PDF), cumulative distribution (CDF), and random sampling for
the sum of independent exponential random variables. The implementation supports
distinct, nearly equal, and repeated parameters, including the Erlang case.

The standalone module [hypoexp.py](hypoexp.py) provides a functional API using
**scales** and an object-oriented API using **rates**.

## Installation and dependencies

Clone the repository and run the examples from its root, where `hypoexp.py` is
importable. This repository currently has no `pyproject.toml` or `setup.py`;
the following instructions use the module directly.

```bash
git clone https://github.com/kevinzagalo/hypoexponential.git
cd hypoexponential
python -m venv .venv
. .venv/bin/activate  # POSIX shells
python -m pip install numpy scipy
```

| Purpose | Dependencies |
| --- | --- |
| Importing and using `hypoexp` | NumPy, SciPy |
| Running the existing tests | Runtime dependencies plus pytest and mpmath; use NumPy >= 2.0 (the tests use `np.trapezoid`) |
| Running the plotting demonstration with `python hypoexp.py` | Matplotlib, in addition to the runtime dependencies |

`decimal` and the other standard-library imports require no separate installation.

## Model: scales versus rates

Let $X = \sum_{i=1}^{n} X_i$, where the $X_i$ are independent exponential
variables with rates $\eta_i > 0$. Their scales (means) are
$\theta_i = 1 / \eta_i$, so

$$
\mathbb{E}[X] = \sum_{i=1}^{n} \theta_i,
\qquad
\operatorname{Var}(X) = \sum_{i=1}^{n} \theta_i^2.
$$

- `hypoexp_pdf`, `hypoexp_cdf`, and `hypoexp_rvs` take `scales = theta`.
- `Hypoexponential(eta)` takes **rates**, and computes the scales as `1 / eta`.
- Scales have the same units as `x`; rates have the reciprocal units.

For example, `scales=[1.0, 2.0]` and `Hypoexponential([1.0, 0.5])` describe
the same distribution, with mean 3 and variance 5. A single component is an
exponential distribution; $n$ equal scales give an Erlang distribution with
shape $n$ and that common scale.

Parameters must be nonempty, finite, and strictly positive. A scalar denotes one
component; array-like parameters are converted to a flat floating-point array.
For extreme rates, the class also requires that the reciprocal scales be
representable as finite positive floats.

## Functional API

| Function | Result |
| --- | --- |
| `hypoexp_pdf(x, scales, tolerance=1e-6, method='auto')` | Density at `x` |
| `hypoexp_cdf(x, scales, tolerance=1e-6, method='auto')` | Probability $P(X \leq x)$ |
| `hypoexp_rvs(scales, size=1000)` | Independent samples of the sum |

### PDF

```python
import numpy as np
from hypoexp import hypoexp_pdf

scales = [1.0, 2.0]
x = np.array([0.0, 1.0, 2.0, 4.0])
pdf = hypoexp_pdf(x, scales)

# For these two scales, f(x) = exp(-x/2) - exp(-x), x >= 0.
expected = np.exp(-x / 2.0) - np.exp(-x)
np.testing.assert_allclose(pdf, expected, rtol=1e-12, atol=1e-15)
print(pdf)
# [0.         0.23865122 0.23254416 0.11701964]

value = hypoexp_pdf(1.0, scales)
assert isinstance(value, float)
print(f"{value:.8f}")  # 0.23865122
```

### CDF

```python
import numpy as np
from hypoexp import hypoexp_cdf

scales = [1.0, 2.0]
x = np.array([0.0, 1.0, 2.0, 4.0])
cdf = hypoexp_cdf(x, scales)

# The integral of the PDF in the preceding example.
expected = 1.0 - 2.0 * np.exp(-x / 2.0) + np.exp(-x)
np.testing.assert_allclose(cdf, expected, rtol=1e-12, atol=1e-15)
print(cdf)
# [0.         0.15481812 0.3995764  0.74764507]

print(f"{hypoexp_cdf(4.0, scales):.8f}")  # 0.74764507
```

For both functions, a scalar `x` returns a Python `float`; an array-like `x`
returns a NumPy array of the same shape. Pointwise evaluation supports arbitrary
spacing and order, including a scalar or a grid covering only part of the
distribution. Explicit multi-component convolution has additional grid
requirements described below.

| Input | PDF | CDF |
| --- | --- | --- |
| `x < 0` (including `-np.inf`) | 0 | 0 |
| `x == 0` | `1 / scale` for one component; 0 for multiple components | 0 |
| `x == np.inf` | 0 | 1 |
| `x` is NaN | NaN | NaN |

### Random variates

```python
import numpy as np
from hypoexp import hypoexp_rvs

np.random.seed(123)
samples = hypoexp_rvs([1.0, 2.0], size=1000)
assert samples.shape == (1000,)
assert np.all(np.isfinite(samples)) and np.all(samples >= 0)

# Re-seeding reproduces the draw in the same environment.
np.random.seed(123)
np.testing.assert_array_equal(samples, hypoexp_rvs([1.0, 2.0], size=1000))

batch = hypoexp_rvs([1.0, 2.0], size=(2, 3))
print(batch.shape)  # (2, 3)
```

Use a nonnegative integer `size` for a one-dimensional sample or a tuple of
nonnegative integers for a specified shape. The default output shape is `(1000,)`;
`size=0` returns an empty array. Each draw is a sum of independent exponential draws,
one per scale.

The current API has no `rng` or `random_state` argument. It calls
`scipy.stats.expon.rvs` using NumPy's global random state, which is why the
example uses `np.random.seed`. The theoretical mean and variance above describe
the distribution; empirical sample moments fluctuate.

## Object-oriented API

```python
import numpy as np
from hypoexp import Hypoexponential, hypoexp_cdf, hypoexp_pdf

rates = np.array([1.0, 0.5])
scales = 1.0 / rates
dist = Hypoexponential(rates)
x = np.array([0.0, 1.0, 2.0, 4.0])

np.testing.assert_allclose(dist.pdf(x), hypoexp_pdf(x, scales), rtol=1e-12)
np.testing.assert_allclose(dist.cdf(x), hypoexp_cdf(x, scales), rtol=1e-12)
print(dist.params)   # {'rates': [1.0, 0.5]}
print(dist.weights)  # [-1.0, 2.0]
assert dist.sample(10).shape == (10,)

# Repeated rates: Erlang with shape 2 and rate 1.
erlang = Hypoexponential([1.0, 1.0])
np.testing.assert_allclose(erlang.pdf(1.0), np.exp(-1.0), rtol=1e-12)
np.testing.assert_allclose(erlang.cdf(1.0), 1.0 - 2.0 * np.exp(-1.0), rtol=1e-12)
```

| Member | Behavior |
| --- | --- |
| `dist.pdf(x)`, `dist.cdf(x)` | Delegate to the functional API with its default `method='auto'` and `tolerance=1e-6` |
| `dist.sample(n_sample=1)` | Delegate to `hypoexp_rvs`; the default returns an array of shape `(1,)` |
| `dist.params` | Property returning `{'rates': [...]}` |
| `dist.weights` | Property returning the signed closed-form coefficients for distinct rates; accessing it with repeated rates raises `ValueError` |

The coefficients in `weights` can be negative. They are the coefficients of
the closed-form sum, not probabilities of mixture components. They can also
suffer numerical cancellation for nearly equal rates; PDF and CDF evaluation
use the numerical methods below independently of this property.

The class methods do not accept `method` or `tolerance` arguments. To choose
these explicitly, call `hypoexp_pdf` or `hypoexp_cdf` with `scales=1.0 / rates`.

## Numerical methods and restrictions

| `method` | PDF | CDF | Requirements and behavior |
| --- | --- | --- | --- |
| `'auto'` (default) | Yes | Yes | Selects methods and retries according to the rules below; supports repeated scales |
| `'closed_form'` | Yes | Yes | Distinct scales for positive finite evaluation points; evaluates the closed-form sum in floating point |
| `'decimal'` | Yes | Yes | Distinct scales for positive finite evaluation points; uses adaptive internal `decimal` precision, with phase-type fallback if its convergence criterion is not met |
| `'phase_type'` | Yes | Yes | Supports distinct, nearly equal, and repeated scales; evaluates a phase-type representation by uniformization |
| `'convolution'` | Yes | No | For multiple components, requires an increasing, uniformly spaced, one-dimensional grid starting at exactly 0, with at least two points |

For distinct scales, the closed-form expressions for $x \geq 0$ are

$$
w_i = \prod_{j \ne i}\frac{\theta_i}{\theta_i-\theta_j},
\qquad
f_X(x) = \sum_i \frac{w_i}{\theta_i}e^{-x/\theta_i},
\qquad
F_X(x) = \sum_i w_i\left(1-e^{-x/\theta_i}\right).
$$

Close scales can make the signed coefficients large and cause cancellation.
Forcing `'closed_form'` bypasses the cancellation-based acceptance test used by
`'auto'`: it can return inaccurate or even negative values, with a warning if
the final sanity checks detect a problem.

`'decimal'` evaluates the same expressions with increased internal precision,
up to 2000 decimal digits. Points that fail its convergence criterion fall back
to `'phase_type'`, even when `'decimal'` was explicitly requested. Inputs have
already been converted to binary floats, and results are returned as floats;
this is not an arbitrary-precision public interface.

`'phase_type'` computes the absorption-time distribution of successive
exponential phases. The implementation uses a nonnegative uniformization
series, scaling and squaring, and a logarithmic path near floating-point range
limits. It supports repeated scales without dividing by scale differences.

```python
import numpy as np
from scipy.stats import gamma
from hypoexp import hypoexp_cdf, hypoexp_pdf

x = np.array([0.5, 1.0, 3.0])
close_scales = [1.0, 1.00000001, 1.00000002]

# Distinct but nearly equal scales: compare two evaluation methods.
for evaluate in (hypoexp_pdf, hypoexp_cdf):
    decimal_values = evaluate(x, close_scales, method='decimal')
    phase_values = evaluate(x, close_scales, method='phase_type')
    np.testing.assert_allclose(decimal_values, phase_values, rtol=1e-10)

# Equal scales: compare with the corresponding Erlang/Gamma distribution.
np.testing.assert_allclose(
    hypoexp_pdf(x, [1.0, 1.0], method='phase_type'),
    gamma.pdf(x, a=2, scale=1.0), rtol=1e-12,
)
np.testing.assert_allclose(
    hypoexp_cdf(x, [1.0, 1.0], method='phase_type'),
    gamma.cdf(x, a=2, scale=1.0), rtol=1e-12,
)
```

### Convolution

Convolution is a discretization of the convolution integral with trapezoidal
endpoint corrections. Its accuracy depends on the grid spacing. Explicit
`method='convolution'` skips the sampled accuracy check performed by `'auto'`;
a coarse grid can produce inaccurate values even if no warning is emitted.

```python
import numpy as np
from hypoexp import hypoexp_pdf

x = np.linspace(0.0, 1.0, 100_000)
pdf = hypoexp_pdf(x, [1.0, 1.0], method='convolution')

# Erlang(2, rate=1) has density x * exp(-x).
np.testing.assert_allclose(pdf, x * np.exp(-x), rtol=0, atol=1e-9)
```

The 100,000-point threshold below applies only to automatic selection;
explicit convolution accepts shorter valid grids. With one scale, the current
driver evaluates the exponential pointwise even if `'convolution'` is selected.
`hypoexp_cdf(..., method='convolution')` raises `ValueError`.

### How `auto` and `tolerance` work

1. For a PDF with multiple components on a uniform grid starting at zero and
   containing at least 100,000 points, try convolution. Compare its values with
   pointwise evaluations at the peak and 15 selected grid indices. Accept when
   the sampled absolute discrepancies are at most `tolerance * max(pdf)`;
   otherwise warn and fall back to pointwise methods.
2. For pointwise evaluation with distinct scales, accept finite, positive
   closed-form values whose estimated relative rounding-error bound is at most
   `tolerance`. Recompute other points using Decimal or phase type, selected
   by an internal cost heuristic. A tolerance below `1e-12` selects Decimal
   for this fallback.
3. For pointwise evaluation with repeated scales, use phase type directly.
   Unconverged Decimal evaluations also fall back to phase type.
4. Check properties such as finiteness, nonnegativity, CDF monotonicity and
   bounds, and PDF shape. A PDF normalization check is also applied on suitable
   grids. Under `'auto'`, failures trigger further evaluation with phase type
   and, when applicable, Decimal. If checks still fail, emit a warning and
   return the best available result. With an explicitly selected method,
   these final checks warn instead of triggering this retry sequence.

`tolerance` controls acceptance criteria; it is **not a uniform bound on the
final error**. In particular, the convolution check is sampled and scaled by
the peak density, so it does not bound relative error throughout the tails.
The Decimal and phase-type routines use their own internal stopping criteria.
Reducing `tolerance` does not change the floating-point return type or guarantee
arbitrarily many correct digits. Use a finite positive tolerance.

The method-selection cost model is a heuristic. Runtime depends on the number
of components and evaluation points, parameter spacing, evaluation locations,
and the environment; no universal speed or accuracy claim is made here.

## Tests and example verification

Install the additional test dependencies and run the existing suite from the
repository root:

```bash
python -m pip install "numpy>=2.0" pytest mpmath
python -m pytest -q test_hypoexp.py
```

The self-contained Python examples above were executed against the current
implementation. The PDF and CDF examples use analytical checks; the repeated
scale examples use Erlang/Gamma identities; sampling checks shape, support,
and reproducibility. Related coverage in [test_hypoexp.py](test_hypoexp.py)
includes high-precision comparisons using mpmath, PDF/CDF consistency,
repeated and nearly equal parameters, method restrictions and fallbacks,
sampling statistics, array shapes, and floating-point range regressions.

Some existing tests include timing assertions. Passing them on a given machine
does not establish a portable performance guarantee.

## Scientific reference

Original reference: https://link.springer.com/article/10.1007/s10440-008-9224-4
