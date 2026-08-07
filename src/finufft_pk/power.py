import warnings
import numpy as np
from .binning import bandpower_from_field_cpp
from dataclasses import dataclass
from finufft import Plan


class FinufftPk:
    def __init__(self, nmesh: tuple[int], boxsize: float, dtype=np.complex64, **kwargs):
        """
        Class inputs (N & M convention inherited from finufft):
        positions: Input position grid, array of shape (D,N)
        weights: Input weights, array of shape (D,N)
        n_modes: Input number of modes, array of shape (D,M)
        dtype: precision level (32 floating point or 64)
        Lbox: int; box size in which points lie
        kwargs: finufft fft precision and spreading function arguments -- upsampfac, fftw, modeord, eps
        """
        # Necessarily real space
        kwargs.setdefault("modeord", 0)
        kwargs.setdefault("eps", 1e-4)
        kwargs.setdefault("upsampfac", 1.25)
        kwargs.setdefault("fftw", 0)
        if kwargs['modeord'] == 1:
            raise ValueError('Mode order 1 is not supported in finufft_pk. Please use modeord = 0.')
        # default num cpus is all in finufft plan
        kwargs.setdefault("nthreads", 0)

        dtype_dict = {np.complex64: np.float32, np.complex128: np.float64}
        if dtype not in dtype_dict.keys():
            raise ValueError(
                f"Data type provided not part of list of valid inputs. Select one from: {dtype_dict.keys()}"
            )
        self.rdtype = dtype_dict[dtype]
        self.cdtype = dtype

        self.nmesh = nmesh
        self.boxsize = boxsize

        # construct FINUFFT Plan
        #!TODO: Save FFT Wisdom after Plan is made
        #!TODO: have to feed in nthreads up here actually, can't retrieve from plan object
        self.plan = Plan(
            nufft_type=1,
            n_modes_or_dim=self._plan_shape(),
            n_trans=1,
            isign=-1,
            dtype=dtype,
            **kwargs,
        )

    def _plan_shape(self):
        n_modes = np.atleast_1d(self.nmesh)
        if n_modes.ndim != 1 or not (1 <= len(n_modes) <= 3):
            raise AssertionError(
                f"n_modes must have length 1-3, got shape {n_modes.shape}"
            )
        *lead, last = n_modes
        return tuple(lead) + (last // 2 + 1,)

    def set_positions(self, positions: tuple, inplace=False):
        """
        Rescale points from [-pi,pi) and pass it to the FINUFFT plan.
        Does not create uniform grid yet (no spreading step).
        """
        n_modes = np.atleast_1d(self.nmesh)
        dim = len(n_modes)
        if positions.shape[0] != dim:
            raise AssertionError(
                f"positions has {positions.shape[0]} axes but nmesh implies dim={dim}"
            )
        if positions.dtype != self.rdtype:
            if not inplace:
                warnings.warn(f"Positions array has dtype {positions.dtype}, which does not match the expected dtype {self.rdtype}. This will affect the performance of finufft.", UserWarning)
            elif inplace:
                raise TypeError(f"Positions array has dtype {positions.dtype}, which does not match the expected dtype {self.rdtype}.")
        
        if inplace:
            positions*= 2*np.pi/self.boxsize
        else:
            positions = positions * (2 * np.pi / self.boxsize)
        self.pos_shape = positions.shape
        # shift = n_modes[-1] // 2  # Take half of the last axis's grid size
        # self._realify_weights = np.exp(-1j * shift * positions[-1, :]).astype(self.cdtype)
        plan_last = self._plan_shape()[-1]  # = nmesh // 2 = 256
        shift = plan_last // 2              # = 128
        self._realify_weights = np.exp(-1j * shift * positions[-1, :]).astype(self.cdtype)
        self._Npts = positions.shape[-1]
        # FINUFFT asks for C arrays, if underlying data is not (dim, N) FINUFFT makes a copy
        self.plan.setpts(*positions)

    def compute_field(self, weights: tuple = None, out: tuple = None):
        # set weights
        if weights is not None:
            if not weights.shape == (self._Npts,):
                        raise AssertionError(
                            f"Shape of weights {self._w_shape} must match number of points ({self._Npts},)"
                        )
        else:
            # LHG: move to C++ (not high priority)
            weights = np.ascontiguousarray(
                np.ones(shape=(self._Npts)), dtype=self.cdtype
            )
        self._w_shape = weights.shape
        
        # set output
        if out is not None:
            if not out.shape == self._plan_shape():
                raise AssertionError(
                    f"Shape of output {out.shape} and meshgrid {self._plan_shape()} must match"
                )
        else:
            out = np.zeros(self._plan_shape(), dtype=self.cdtype)

        field = self.plan.execute(weights * self._realify_weights, out=out)
        return field

    def compute_bandpower(self, field, kbins:int=None, mubins:int=1, nthread:int=None, reuse_counts = False):
            if field.shape != self._plan_shape():
                raise AssertionError(
                    f"Shape of field {field.shape} must match plan shape {self._plan_shape()}"
                )
            counts = self.counts if reuse_counts and hasattr(self, 'counts') else None
            k_binc, counts, weighted_counts, bandpower = bandpower_from_field_cpp(field, self.boxsize, kbins, mubins, nthread=nthread, counts=counts)
            self.counts = counts  # do we always save counts?? probably yes
            return k_binc, counts, weighted_counts, bandpower

@dataclass
class FinufftPkResult:
    k_edges: np.typing.ArrayLike  # length N+1
    k_avg: np.typing.ArrayLike  # length N
    power: np.typing.ArrayLike  # length N
    boxsize: float

    def Nyquist(self):
        return 2*np.pi/self.boxsize

def powerspectrum_field(nmesh: tuple[int], boxsize: float, positions: tuple, weights: tuple = None, out: tuple = None, dtype=np.complex64, kbins:int=None, mubins:int=1, nthread:int=None, **kwargs):
    # LHG: should this default to FFTW_ESTIMATE?
    print('Initializing FINUFFT Plan')
    plan = FinufftPk(nmesh=nmesh, boxsize=boxsize, dtype=dtype, **kwargs)
    print("setting positions")
    plan.set_positions(positions=positions)
    print("computing field")
    field = plan.compute_field(weights, out)
    print("computing bandpowers")
    powerspectrum = plan.compute_bandpower(field, kbins, mubins, nthread)

    return powerspectrum #Change to FINUFFT data class
