import numpy as np
from finufft import Plan
from abacusnbody.analysis.power_spectrum import (
    calc_power,
    calc_pk_from_deltak,
    get_k_mu_edges,
)


class FinufftPowerSpectrum:
    def __init__(self, nmesh: tuple[int], boxsize: float, dtype=np.complex64, **kwargs):
        """
        Class inputs (N & M convention inherited from finufft):
        positions: Input position grid, array of shape (D,N)
        weights: Input weights, array of shape (D,N)
        n_modes: Input number of modes, array of shape (M,D)
        dtype: precision level (32 floating point or 64)
        Lbox: int; box size in which points lie
        kwargs: finufft fft precision and spreading function arguments -- upsampfac, fftw, modeord, eps
        """
        # Necessarily real space
        modeord = kwargs.setdefault('modeord', 0)
        eps = kwargs.setdefault("eps", 1e-4)
        upsampfac = kwargs.setdefault("upsampfac", 1.25)
        fftw = kwargs.setdefault("fftw", 0)

        dtype_dict = {np.complex64: np.float32, np.complex128:np.float64}
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
            eps=eps,
            isign=-1,
            dtype=dtype,
            fftw=fftw,
            upsampfac=upsampfac,
            modeord=modeord,
        )

    def _plan_shape(self):
        n_modes = np.atleast_1d(self.nmesh)
        if n_modes.ndim != 1 or not (1 <= len(n_modes) <= 3):
            raise AssertionError(
                f"n_modes must have length 1-3, got shape {n_modes.shape}"
            )
        *lead, last = n_modes
        return tuple(lead) + (last // 2,)

    def set_positions(self, positions: tuple):
        """
        Rescale points from [-pi,pi) and pass it to the FINUFFT plan.
        Does not create uniform grid yet (no spreading step).
        """
        # rescale from [-pi, pi) for correct physical scaling
        positions *= 2 * np.pi
        positions /= self.boxsize
        self.pos_shape = positions.shape
        shift = self.nmesh[-1] // 2  # Take half of the z-axis grid size
        self._realify_weights = np.exp(-1j * shift * positions[:, 2]).astype(self.dtype)
        self._Npts = len(positions.ravel())
        # FINUFFT asks for C arrays, if underlying data is not 3,N FINUFFT makes a copy
        self.plan.setpts(x=positions[0, :], y=positions[1, :], z=positions[2, :])

    def compute_field(self, weights:tuple=None, out:tuple=None):
        w_shape = weights.shape

        #set weights
        if weights:
            if not w_shape == self.pos_shape:
                raise AssertionError(
                    f"Shape of weights {w_shape} and positions {self.pos_shape} must match"
                )
        else:
            weights = np.ascontiguousarray(np.ones(shape=(self._Npts)), dtype=np.complex64)

        #set output
        if out:
            if not out.shape == self._plan_shape:
                raise AssertionError(
                                    f"Shape of weights {w_shape} and positions {self.pos_shape} must match"
                                )
        else:
            out = np.zeros(self._plan_shape(), dtype=self.cdtype)

        field = self.plan.execute(weights*self._realify_weights, out=out)
        return field

    def compute_bandpower(self, field):
        raw_power = np.abs(field) ** 2
        L = self.boxsize
        dk = 2 * np.pi / self.boxsize
        # get bin edges, k is wave mode, mu is angle away from LOS,
        # get_kmu_edges()

        n_modes = np.atleast_1d(self.nmesh)
        # Nyquist is limited by the coarsest axis so bins stay within every axis's range
        k_max = np.pi * n_modes.min() / L
        kbins = np.linspace(0, k_max, n_modes.min() // 2 + 1)

        mu = 1
        mubins = np.linspace(0, 1, mu + 1)

        # create bandpower_array
        Nk, Nmu = len(kbins) - 1, len(mubins) - 1
        counts = np.zeros((Nk, Nmu))
        weighted_counts = np.zeros((Nk, Nmu))

        # bin_kmu()
        kedges2 = (kbins / dk) ** 2
        muedges2 = mubins**2

        # all axes but the last need +/-N/2 wraparound folding; the last axis
        # is already the real-transform half-plane (0..N/2), so it never folds
        fold_shape = field.shape[:-1]
        len_z = field.shape[-1]

        def folded_sq(idx, n):
            return idx**2 if idx < n // 2 else (idx - n) ** 2

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