import warnings
import cupy as cp
from .binning import bandpower_from_field_cpp
from dataclasses import dataclass
from cufinufft import Plan
from ._helper_functions import *


class FinufftPk:
    def __init__(self, nmesh: tuple[int], boxsize: float, dtype=cp.complex64, **kwargs):
        """
        Build a FINUFFT type-1 plan for computing the density field on a
        mode grid of shape ``nmesh`` (N & M convention inherited from finufft).

        nmesh: tuple[int], length 1-3
            Number of modes per axis of the real-space grid.
        boxsize: float
            Physical size of the box the input points lie in.
        dtype: cp.complex64 or cp.complex128
            Precision of the field/weights; determines the real dtype
            (float32/float64) expected for positions.
        kwargs: finufft plan/spreading arguments -- upsampfac, fftw,
            modeord (must be 0), eps, nthreads.

        Raises
        ------
        ValueError: if modeord != 0 or dtype is not complex64/complex128.
        """
        # Necessarily real space
        kwargs.setdefault("modeord", 0)
        kwargs.setdefault("eps", 1e-4)
        kwargs.setdefault("upsampfac", 1.25)
        if kwargs["modeord"] == 1:
            raise ValueError(
                "Mode order 1 is not supported in finufft_pk. Please use modeord = 0."
            )

        dtype_dict = {cp.complex64: cp.float32, cp.complex128: cp.float64}
        if dtype not in dtype_dict.keys():
            raise ValueError(
                f"Data type provided not part of list of valid inputs. Select one from: {dtype_dict.keys()}"
            )
        self.rdtype = dtype_dict[dtype]
        self.cdtype = dtype

        self.nmesh = nmesh
        self.boxsize = boxsize

        # construct FINUFFT Plan
        # unable to save Plan after it's made
        self.plan = Plan(
            nufft_type=1,
            n_modes=self._plan_shape(),
            n_trans=1,
            isign=-1,
            dtype=dtype,
            **kwargs,
        )
        self.result = FinufftPkResult(
            boxsize=self.boxsize, nmesh=self.nmesh, finufft_kwargs=kwargs
        )

    def _plan_shape(self):
        """
        Return the FINUFFT mode-grid shape, with the last axis
        collapsed to N//2+1 (real-to-complex convention).

        Returns
        -------
        tuple[int], length 1-3: plan shape, e.g. (n0, ..., n_{d-2}, n_{d-1}//2 + 1).

        Raises
        ------
        AssertionError: if self.nmesh does not have length 1-3.
        """
        n_modes = cp.atleast_1d(self.nmesh)
        if n_modes.ndim != 1 or not (1 <= len(n_modes) <= 3):
            raise AssertionError(
                f"n_modes must have length 1-3, got shape {n_modes.shape}"
            )
        *lead, last = n_modes
        last_shape = int(last // 2 + 1)
        return tuple(int(i) for i in lead) + (last_shape,)

    def set_positions(self, positions: tuple, inplace=False):
        """
        Rescale points from [-pi,pi) and pass it to the FINUFFT plan.
        Does not create uniform grid yet (no spreading step).

        positions: ndarray, shape (D, N), dtype float32/float64 matching
            self.rdtype (dtype mismatch warns unless inplace=True, in which
            case it raises). D must equal len(self.nmesh).
        inplace: bool
            If True, rescale positions in place (dtype must already match);
            if False (default), rescale a copy and only warn on mismatch.

        Side effects: sets self.pos_shape, self._Npts, self._realify_weights,
        passes points to the FINUFFT plan via setpts, and records
        self.result.kwargs['inplace'].

        Raises
        ------
        AssertionError: if positions.shape[0] != len(self.nmesh).
        TypeError: if inplace=True and positions.dtype != self.rdtype.
        """
        n_modes = cp.atleast_1d(self.nmesh)
        dim = len(n_modes)
        if positions.shape[0] != dim:
            raise AssertionError(
                f"positions has {positions.shape[0]} axes but nmesh implies dim={dim}"
            )
        if positions.dtype != self.rdtype:
            if not inplace:
                warnings.warn(
                    f"Positions array has dtype {positions.dtype}, which does not match the expected dtype {self.rdtype}. This will affect the performance of finufft.",
                    UserWarning,
                )
            elif inplace:
                raise TypeError(
                    f"Positions array has dtype {positions.dtype}, which does not match the expected dtype {self.rdtype}."
                )
        if inplace:
            positions*= 2*cp.pi/self.boxsize
        else:
            positions = positions * (2 * cp.pi / self.boxsize)
        self.pos_shape = positions.shape
        # shift = n_modes[-1] // 2  # Take half of the last axis's grid size
        # self._realify_weights = np.exp(-1j * shift * positions[-1, :]).astype(self.cdtype)
        plan_last = self._plan_shape()[-1]  # = nmesh // 2 = 256
        shift = plan_last // 2              # = 128
        self._realify_weights = cp.exp(-1j * shift * positions[-1, :]).astype(self.cdtype)
        self._Npts = positions.shape[-1]
        # FINUFFT asks for C arrays, if underlying data is not (dim, N) FINUFFT makes a copy
        self.plan.setpts(*positions)
        self.result.kwargs = {"inplace": inplace}

    def compute_field(self, weights: tuple = None, out: tuple = None):
        """
        Execute the FINUFFT plan to spread weighted points onto the
        mode grid, producing the (complex) density field. Requires
        set_positions to have been called first.

        weights: ndarray, shape (N,), dtype self.cdtype, or None
            Per-point weights; defaults to an array of ones.
        out: ndarray, shape == self._plan_shape(), dtype self.cdtype, or None
            Pre-allocated output array; if None, a new zero array is created.

        Returns
        -------
        ndarray, shape self._plan_shape(), dtype self.cdtype: the density field.

        Raises
        ------
        AssertionError: if weights.shape != (N,) or out.shape != plan shape.
        """
        # set weights
        if weights is not None:
            if not weights.shape == (self._Npts,):
                raise AssertionError(
                    f"Shape of weights {weights.shape} must match number of points ({self._Npts},)"
                )
            self.result.kwargs["input_weights"] = weights
        else:
            weights = cp.ascontiguousarray(
                cp.ones(shape=(self._Npts)), dtype=self.cdtype
            )
        self._w_shape = weights.shape

        # set output
        if out is not None:
            if not out.shape == self._plan_shape():
                raise AssertionError(
                    f"Shape of output {out.shape} and meshgrid {self._plan_shape()} must match"
                )
            self.result.kwargs["out_array"] = out
        else:
            out = cp.zeros(self._plan_shape(), dtype=self.cdtype)

        field = self.plan.execute(weights * self._realify_weights, out=out)
        self.result.field = field
        return field

    def compute_bandpower(
        self, field, kbins: int = None, mubins: int = 1, nthread: int = None
    ):
        """
        Bin the computed field into spherical (or mu-wedge) k-bins
        to produce the power spectrum, via bandpower_from_field_cpp.

        field: ndarray, shape self._plan_shape(), complex64/complex128
            Density field as returned by compute_field.
        kbins: int or None
            Number of k bins; if None, defaults to min(nmesh)//2 + 1.
        mubins: int
            Number of mu (angle-cosine) bins.
        nthread: int or None
            Threads for the binning step; defaults to self._nthreads.

        Returns
        -------
        k_binc: ndarray, shape (kbins,) -- bin-center k values.
        counts: ndarray, shape (kbins, mubins) -- mode counts per bin.
        weighted_counts: ndarray, shape (kbins, mubins) -- mean power per bin.
        bandpower: ndarray, shape (kbins * mubins,) -- flattened band power.

        Raises
        ------
        AssertionError: if field.shape != self._plan_shape().
        """
        if nthread:
            nthread_bandpower = nthread
        else:
            nthread_bandpower = self._nthreads
        if field.shape != self._plan_shape():
            raise AssertionError(
                f"Shape of field {field.shape} must match plan shape {self._plan_shape()}"
            )
        k_binc, counts, weighted_counts, bandpower = bandpower_from_field(
            field, self.boxsize, kbins, mubins, nthread=nthread_bandpower
        )
        self.result.k_avg = k_binc
        self.result.counts = counts
        self.result.weighted_counts = weighted_counts
        self.result.power = bandpower
        if not kbins:
            self.result.kwargs["kbins"] = min(self.nmesh) // 2 + 1
        else:
            self.result.kwargs["kbins"] = kbins
        self.result.kwargs["mubins"] = mubins
        return k_binc, counts, weighted_counts, bandpower


@dataclass
class FinufftPkResult:
    # Add a flag to save field if wanted
    field: cp.typing.ArrayLike | None = None  # shape (N,N, N//2+1)
    power: cp.typing.ArrayLike | None = None  # length N
    boxsize: float | None = None
    counts: cp.typing.ArrayLike | None = None  # length N
    weighted_counts: cp.typing.ArrayLike | None = None  # length N
    nmesh: tuple | None = None
    finufft_kwargs: dict | None = None
    k_avg: cp.typing.ArrayLike | None = None  # length N
    kwargs: dict | None = None  # nthreads, dtype, kbins, mubins, inplace

    def Nyquist(self):
        """
        Return the Nyquist wavenumber implied by nmesh and boxsize.

        Returns
        -------
        float: pi * min(nmesh) / boxsize.
        """
        return cp.pi * min(self.nmesh) / self.boxsize


def powerspectrum_field(
    nmesh: tuple[int],
    boxsize: float,
    positions: tuple,
    weights: tuple = None,
    out: tuple = None,
    dtype=cp.complex64,
    kbins: int = None,
    mubins: int = 1,
    nthread: int = None,
    **kwargs,
):
    """
    Convenience wrapper that builds a FinufftPk plan, sets positions,
    computes the field, and bins it into a power spectrum in one call.

    nmesh: tuple[int], length 1-3 -- number of modes per axis.
    boxsize: float -- physical size of the box the points lie in.
    positions: ndarray, shape (D, N) -- point coordinates, D == len(nmesh).
    weights: ndarray, shape (N,), or None -- per-point weights.
    out: ndarray, shape matching the plan's mode grid, or None -- output buffer.
    dtype: cp.complex64 or cp.complex128 -- field precision.
    kbins: int or None -- number of k bins; defaults to min(nmesh)//2 + 1.
    mubins: int -- number of mu (angle-cosine) bins.
    nthread: int or None -- threads for the binning step.
    kwargs: additional FINUFFT plan/spreading arguments.

    Returns
    -------
    FinufftPkResult: populated with field, power, counts, weighted_counts,
    k_avg, boxsize, nmesh, finufft_kwargs, and kwargs.
    """
    print("Initializing FINUFFT Plan")
    kwargs.setdefault("fftw", 64)
    plan = FinufftPk(nmesh=nmesh, boxsize=boxsize, dtype=dtype, **kwargs)
    print("setting positions")
    plan.set_positions(positions=positions)
    print("computing field")
    field = plan.compute_field(weights, out)
    print("computing bandpowers")
    plan.compute_bandpower(field, kbins, mubins, nthread)

    return plan.result  # Change to FINUFFT data class
