# FINUFFT Pk: A faster and more accurate power spectrum computation

The matter power spectrum is an essential measure of fundamental cosmology, because the distribution of matter today directly traces the distribution of matter at the time of inflation. By reframing the cosmological problem of deriving the matter power spectrum from a theoretical standpoint, to a non-uniform fast fourier transform, we can infer fundamental physics from simulations quickly and explore massive parameter spaces. This is the motivation behind creating the fast, and more accurate power spectrum calculator `finufft_pk`.

## Installation

`finufft_pk` is not yet published to PyPI, so it must be built from source. The build
compiles a C++ extension for your machine's CPU (`-march=native`), so it should be
installed on the machine it will run on rather than distributed as a pre-built wheel.

### Prerequisites

- Python >= 3.11.11
- CMake >= 3.15
- A C++ compiler with OpenMP support
- `nanobind`, findable via CMake config (installed automatically as a build dependency)

`finufft` (the Python bindings this package depends on) ships pre-built wheels with
FFTW bundled, so no separate FFTW/FINUFFT C library installation is required.

### Build

```bash
git clone git@github.com:pk0207/finufft_pk.git
cd finufft_pk
pip install .
```

or with uv

```bash
git clone git@github.com:pk0207/finufft_pk.git
cd finufft_pk
uv sync
```

## Usage

Capabilities:

- Arbitrary number of dimensions (1-3)
- Arbitrary mesh set up to make uniform grid
- Independently update weights without re-initializing points distribution
- Fast weight updates to enable MCMC applications

Limitations (do not yet support):

- Aperiodic bounding boxes
- Concatenating points onto the same field to calculate a piecewise power spectrum
- Interlaced mesh grids

# Getting a Power Spectrum

## `powerspectrum_field()`

A quick way to get one power spectrum from a set of non-uniform points.

### Usage example

```python
import numpy as np
from finufft_pk import FinufftPk
# Create a set of non-uniform points with some density in a 3D box
# Accepts 1-3 dimensional data

boxsize = 250 # (Mpc / h)
density = 1e-4 # (Mpc / h)**-3
N = int(density * (boxsize**3))
positions = np.random.uniform(0, boxsize, size=(3, N))

# Set the number of grid-cells to divide the box into.
# Allowed to be any arbitrary set of integers

xmesh = 500
ymesh = 400
zmesh = 200

# Returns a FinufftPkResult object
ps_result = powerspectrum_field((xmesh, ymesh, zmesh), boxsize, positions)
```

## "Guru" interface

`powerspectrum_field()` is a thin wrapper around the `FinufftPk` class: it builds a
plan, sets positions, computes the field, and bins the power spectrum in one call.
Using `FinufftPk` directly gives fine-grained control over FINUFFT internals, and lets
you reuse a plan (e.g. re-run `compute_field`/`compute_bandpower` with new weights
without redoing the spreading setup).

### Step-by-step usage

```python
import numpy as np
from finufft_pk import FinufftPk

boxsize = 250  # (Mpc / h)
density = 1e-4  # (Mpc / h)**-3
N = int(density * (boxsize**3))
positions = np.random.uniform(0, boxsize, size=(3, N)).astype(np.float32)

# 1. Build the FINUFFT plan for a given mode grid, box size, and precision.
#    Extra finufft kwargs (eps, upsampfac, nthreads, ...) can be passed here.
plan = FinufftPk((500, 400, 200), boxsize, dtype=np.complex64)

# 2. Rescale the points to [-pi, pi) and hand them to the FINUFFT plan.
#    This does not spread the points onto the grid yet.
plan.set_positions(positions)

# 3. Spread the (optionally weighted) points onto the mode grid and take
#    the transform, producing the complex density field.
weights = np.ones(N, dtype=np.complex64)
field = plan.compute_field(weights)

# 4. Bin the field into k (and optionally mu) bins to get the power spectrum.
k_binc, counts, weighted_counts, bandpower = plan.compute_bandpower(field, kbins=100)
```

Because `set_positions` and `compute_field` are separate steps, weights can be updated
and the field/bandpower recomputed without re-initializing the plan or re-spreading
positions:

```python
new_weights = np.random.uniform(0.5, 1.5, size=N).astype(np.complex64)
field = plan.compute_field(new_weights)
plan.compute_bandpower(field, kbins=100)
```

### The `FinufftPkResult`

Every `FinufftPk` instance owns a `result` attribute (a `FinufftPkResult` dataclass)
that is populated as each step runs, and is what `powerspectrum_field()` returns:

```python
result = plan.result

result.field            # complex ndarray, shape (500, 400, 101) -- the density field
result.power            # ndarray, shape (kbins * mubins,) -- flattened band power
result.k_avg            # ndarray, shape (kbins,) -- bin-center k values
result.counts           # ndarray, shape (kbins, mubins) -- mode counts per bin
result.weighted_counts  # ndarray, shape (kbins, mubins) -- mean power per bin
result.boxsize          # 250
result.nmesh            # (500, 400, 200)
result.finufft_kwargs   # dict of the finufft plan kwargs used (modeord, eps, ...)
result.kwargs           # dict recording inplace/kbins/mubins/input_weights/out_array

result.Nyquist()        # Nyquist wavenumber implied by nmesh and boxsize
```

### FINUFFT keyword arguments

Any keyword accepted by `finufft.Plan` (besides `modeord`, which `finufft_pk` fixes to
`0`) can be passed straight through `FinufftPk(...)` or `powerspectrum_field(...)`,
for example `eps`, `upsampfac`, `nthreads`, and `fftw`.

#### `eps` and `upsampfac`

`eps` and `upsampfac` are internal FINUFFT keyword arguments that affect how the input mesh grid is treated and how kernel width is set.

- `eps` sets the number of decimals to which the solution should be calculated (precision). The FINUFFT default is 1e-6, while the `finufft_pk` default is 1e-4.
- `upsampfac` determines by how much the internal meshgrid spacing is multiplied for the spreading step. The FINUFFT default is 2, while the `finufft_pk` default is 1.25.

These design decisions were made to improve speed performance.

#### `nthreads`

`nthreads` is the number of CPUs used throughout FINUFFT and `finufft_pk`. The default is set to `0`, which means all available CPUs will be used.

#### `fftw`

FINUFFT uses the FFTW fourier transform code to do a fourier transform. `fftw` refers to the FFTW mode to use.

- `0` is measure mode, where the FFTW plan is more carefully estimated. This slows down the initial set up but significantly speeds up subsequent runs on the same initialization.
- `64` is estimate mode, which is faster to initialize but slower when running multiple times.

`fftw=0` is the default in the guru interface while `fftw=64` is the default in the `powerspectrum_field()` function.

#### `spreadinterponly`

FINUFFT's spreading and its FFT step are normally both run as part of a single `execute`
call. Setting `spreadinterponly=1` as a plan kwarg tells FINUFFT to perform *only* the
spreading step and skip the FFT entirely -- `compute_field` then returns the spread
(real-space) grid rather than the Fourier-space field.

This is useful when you want the uniform-grid representation of your points for its
own sake (e.g. to inspect the mass assignment, feed it into a different FFT backend or
pipeline, or diagnose the spreading kernel) without paying for a transform you don't
need, or when you plan to do the FFT step yourself with different settings than
FINUFFT's internal one:

```python
plan = FinufftPk((500, 400, 200), boxsize, spreadinterponly=1)
plan.set_positions(positions)
spread_grid = plan.compute_field(weights)  # real-space grid, not P(k)-ready
```

Note that `compute_bandpower` expects a Fourier-space field, so it should not be
called on the output of a `spreadinterponly` plan.

## 2D bandpower binnings

By default, `finufft_pk` calculates binning only along the line of sight (i.e., only uses radial k bins). It does contain the option to do angular binning `mubins` in the `compute_bandpower()` step, which determines the number of angular bins one wants to set up.

Both the pure-Python (`bandpower_from_field`) and C++-accelerated
(`bandpower_from_field_cpp`) binning implementations support arbitrary `mubins`; the
angular bin count only changes the shape of `counts`/`weighted_counts`
(`(kbins, mubins)`) and the length of the flattened `bandpower` array
(`kbins * mubins`), so no other code changes are needed to go from a 1D $P(k)$ to a 2D
$P(k, \mu)$:

```python
k_binc, counts, weighted_counts, bandpower = plan.compute_bandpower(
    field, kbins=100, mubins=5
)
bandpower.reshape(100, 5)  # P(k, mu), one row per k bin, one column per mu wedge
```

# Cosmological and Mathematical Background

The power spectrum - P(k) - of the distribution of galaxies in the universe (observed or simulated) completely encodes the large-scale spatial distribution of matter in the universe. Knowing this distribution helps cosmologists infer information about the growth of large scale structure, and fundamental physics.
P(k) is the amplitude of fourier modes k, which represents the number of galaxies that are a fourier space distance k from each other.

## Steps to calculate a power spectrum from a set of galaxies

1. Galaxies are not uniformly distributed in space, but fast fourier transform algorithms can only be performed on uniform grids. So, first the galaxy particles are distributed onto a uniform mesh grid. This is called the "spreading" step. Once the mesh grid dimensions are selected, some "mass" of each particle is assigned to each mesh cell. This operation is done by convolving the points with a kernel that smears the mass of the particle onto the grid.

<div align="center">

  <img src="readme_images/spreading%20diagram.png"
  alt="A diagram depicting the spreading step of the powerspectrum creation. Three grids are shown. The first grid is shown over a distribution of points that represent galaxy particles. The second grid shows an example of how 'mass' might be distributed on the grid for some of the points. The third grid shows the amplitude of the points 'mass' in each grid cell, with a darker red representing more 'mass'. This is a simple illustration of the spreading step of the power spectrum creation."
  width="20%">

  <p><strong>Fig 1.</strong> A diagram depicting the spreading step of the powerspectrum creation. Three grids are shown. The first (top left) grid is shown over a distribution of points that represent galaxy particles. The second grid (top right) shows an example of how 'mass' might be distributed on the grid for some of the points. The third grid (bottom) shows the amplitude of the points 'mass' in each grid cell, with a darker red representing more 'mass'. This is a simple illustration of the spreading step of the power spectrum creation.</p>
</div>

2. Once a uniform grid is created, the fourier transform of the grid can be taken. This results in a field in fourier $k$-space where $k$ is a fourier mode, and the smallest $k$ is the fundamental mode spanning the size of the box. This power of the field is a description of the amount of "clumpiness" of galaxies at different scales.

3. Once the field is created, to retrieve the information about the distribution of points before spreading, a deconvolution step with the spreading kernel has to be taken.

Insert kernel comparison diagram

4. Once the field is deconvolved, the power at points in the field with the same $k$ value in different dimensions are added up. This is the binning process that builds up the 1 dimensional powerspectrum that only depends on $k$ (P(k)).

<div align="center">

  <img src="readme_images/binning%20diagram.png"
  alt="A diagram depicting the binning step of the powerspectrum creation. A resulting 2D grid of points (uniform for ease of interpretation) is shown in purple. The red curved lines are the locations in k-space that all have the same abosolute k-value. The points along the red lines are binned and normalized by counts to create the power spectrum."
  width="20%">

  <p><strong>Fig 2.</strong> A diagram depicting the binning step of the powerspectrum creation. A resulting 2D grid of points (uniform for ease of interpretation) is shown in purple. The red curved lines are the locations in k-space that all have the same abosolute k-value. The points along the red lines are binned and normalized by counts to create the power spectrum.</p>
</div>

## Mathematical Definitions

This package is powered by the flatiron institute non-uniform fast fourier transform (FINUFFT). FINUFFT was constructed to mathematically optimize the spreading step, using a kernel that would retain more information than simpler kernels. FINUFFT uses an "exponential of semi-circle" kernel, rather than the more common Triangle Shaped Cloud (TSC) or Cloud In Cell (CIC) kernel. The difference in shapes is shown below in figure _

Since `finufft_pk` is built around FINUFFT, is inherits its mathematical conventions [FINUFFT documentation](https://finufft.readthedocs.io/en/latest/math.html).

1. **The Fourier Transform Convention:**
$$ f_k := \sum_{j=1}^{M} c_j e^{-i \text{\textbf{k}} \cdot \text{\textbf{x}}_j} $$
By this definition, the points have to be rescaled to $[- \pi, \pi)$ before transforming.

2. **Mode Order:**
$$
K_{N_i} := \begin{cases}
\{-N_i/2, \ldots, N_i/2-1\}, & N_i \text{ even}, \\
\{-(N_i-1)/2, \ldots, (N_i-1)/2\}, & N_i \text{ odd}.
\end{cases}
$$
The FINUFFT fourier mode convention is ${-N_i/2,...0,...,N_i/2 - 1}$.

3. **The Half-Plane Trick:** `finufft_pk` always takes a real distribution of points. A real matrix of points is Hermitian symmetric, so only half of the solution (along the diagonal) needs to be calculated. The solution for the other half is identical, and can be obtained just by reflecting the first half's solution.

FINUFFT always assumes a complex distribution. This means that there is no assumption of Hermitian symmetry, and FINUFFT is doing double the amount of work for real points by calculating the solution for the full distribution of points.

We can trick FINUFFT into doing only half the work by only giving it half the matrix. In order to maintain the correct mode ordering (expecting {-N/2.....0} for the first half), we weight the positions in the *last* axis of the matrix to shift the points to {-N/2.....0}.
$$ c_j \rightarrow c_j e^{i \frac{M_\text{last}}{4} \text{last}_j} $$

# Acknowledgements

The author of this package would like to thank the Simons Foundation summer internship program with the Scientific Computing Core (SCC) for the opportunity to work on this project from June to August 2026. They would also like to thank Lehman Garrison for his supervision and guidance during the construction of this project, from whom they have learned so much.

# References

1. Barnett, A. H., Magland, J., & af Klinteberg, L. (2019). A parallel nonuniform fast Fourier transform library based on an “exponential of semicircle" kernel. SIAM Journal on Scientific Computing, 41(5), C479-C504.

2. Barnett, A. H. (2021). Aliasing error of the exp⁡(β1− z2) kernel in the nonuniform fast Fourier transform. Applied and Computational Harmonic Analysis, 51, 1-16. 

3. Unseel Matter Power Spectrum visualization tool: [https://unseel.com/astronomy/matter-power-spectrum](https://unseel.com/astronomy/matter-power-spectrum)
