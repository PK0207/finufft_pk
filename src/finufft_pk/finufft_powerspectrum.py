import numpy as np
from finufft import Plan
import warnings
# import os
# import subprocess


class FinufftPowerSpectrum:
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
        #!TODO: check that modeord = 0 (assert)
        kwargs.setdefault("modeord", 0)
        kwargs.setdefault("eps", 1e-4)
        kwargs.setdefault("upsampfac", 1.25)
        kwargs.setdefault("fftw", 0)
        if kwargs['modeord'] == 1:
            raise AssertionError('Mode order 1 is not supported in finufft_pk. Please use modeord = 0.')
        # default num cpus is all in finufft plan

        dtype_dict = {np.complex64: np.float32, np.complex128: np.float64}
        if dtype not in dtype_dict.keys():
            raise AssertionError(
                f"Data type provided not part of list of valid inputs. Select one from: {dtype_dict.keys()}"
            )
        self.rdtype = dtype_dict[dtype]
        self.cdtype = dtype

        self.nmesh = nmesh
        self.boxsize = boxsize

        # construct FINUFFT Plan
        #!TODO: Save FFT Wisdom after Plan is made
        self.plan = Plan(
            nufft_type=1,
            n_modes_or_dim=self._plan_shape(),
            n_trans=1,
            isign=-1,
            dtype=dtype,
            **kwargs,
        )

    # IN progress: original plan was get htop output but what if windows?
    # for now, just CPU-1
    # def _calc_num_cpu(nmesh, ):
    #     #Roughly estimate a good number of cpus
    #     #Each data points is 32 bits
    #     #Get L2 cache storage size
    #     htop_output = subprocess.run('htop')
    #     return num_cpu

    def _plan_shape(self):
        n_modes = np.atleast_1d(self.nmesh)
        if n_modes.ndim != 1 or not (1 <= len(n_modes) <= 3):
            raise AssertionError(
                f"n_modes must have length 1-3, got shape {n_modes.shape}"
            )
        *lead, last = n_modes
        return tuple(lead) + (last // 2,)

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
        shift = n_modes[-1] // 2  # Take half of the last axis's grid size
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

    def compute_bandpower(self, field):#, kbins, mubins):
        #!TODO: change bandpower computation so that everything except last axis is handled properly
        if field.shape != self._plan_shape():
            raise AssertionError(
                f"Shape of field {field.shape} must match plan shape {self._plan_shape()}"
            )
        raw_power = np.abs(field) ** 2
        L = self.boxsize
        dk = 2 * np.pi / self.boxsize
        # get bin edges, k is wave mode, mu is angle away from LOS,
        fold_shape = field.shape[:-1]
        len_z = field.shape[-1]
        full_n = np.array(list(fold_shape) + [2 * len_z])
        # Nyquist is limited by the coarsest axis so bins stay within every axis's range
        k_max = np.pi * full_n.min() / L
        kbins = np.linspace(0, k_max, full_n.min() // 2 + 1)

        mu = 1
        mubins = np.linspace(0, 1, mu + 1)

        # create bandpower_array
        Nk, Nmu = len(kbins) - 1, len(mubins) - 1
        counts = np.zeros((Nk, Nmu))
        weighted_counts = np.zeros((Nk, Nmu))

        # bin_kmu()
        kedges2 = (kbins / dk) ** 2
        muedges2 = mubins**2

        # all axes but the last are stored in FINUFFT's CMCL (modeord=0) order, i.e. index 0
        # is k=-N/2 and index N-1 is k=(N-1)/2 (no FFT-style wraparound); the last axis is
        # already the real-transform half-plane (0..N/2), so it's handled separately
        fold_shape = field.shape[:-1]
        len_z = field.shape[-1]

        def folded_sq(idx, n):
            return (idx - n // 2) ** 2

        for perp_idx in np.ndindex(*fold_shape):
            perp2 = sum(folded_sq(i, n) for i, n in zip(perp_idx, fold_shape))
            for k in range(len_z):
                bk, bmu = 0, 0  # k-mode counter, so we don't add counts the zero mode
                k2 = k**2  # 0 to N/2
                mag = perp2 + k2

                if mag > 0:
                    invkmag2 = mag**-1
                    mu2 = k2 * invkmag2
                else:
                    mu2 = 0

                if mag < kedges2[0]:  # if it goes below the initial mode ignore it
                    continue
                if mag >= kedges2[-1]:  # if it reaches kmax, we're done
                    break
                while mag > kedges2[bk + 1]:
                    bk += 1  # don't double count zeroth order
                while mu2 > muedges2[bmu + 1]:
                    bmu += 1

                weight = 1 if k == 0 else 2
                counts[bk, bmu] += weight
                weighted_counts[bk, bmu] += weight * raw_power[perp_idx + (k,)]

        for i in range(Nk):
            for j in range(Nmu):
                weighted_counts[i,j] /= counts[i,j]

        bandpower = weighted_counts*(L**3)
        bandpower = bandpower.flatten()
        return counts, weighted_counts, bandpower

    def powerspectrum_field(
        self, positions: tuple, weights: tuple = None, out: tuple = None
    ):
        print("setting positions")
        self.set_positions(positions=positions)
        print("computing field")
        field = self.compute_field(weights, out)
        print("computing bandpowers")
        powerspectrum = self.compute_bandpower(field)

        return powerspectrum
