from src.finufft_pk.finufft_powerspectrum import FinufftPowerSpectrum
import numpy as np
import pickle

def main():
    fname = '/mnt/home/pkottapalli/fft_benchmark/positions_grid_512_nbar1_P01.pkl'
    with open(fname, 'rb') as f:
        pos_dict = pickle.load(f)
    ngen = pos_dict['ngen']
    # stored as (N, 3); set_positions needs (D, N)
    pos_red = np.ascontiguousarray(pos_dict['pos'].T)
    L = pos_dict['L']

    powerspectrum = FinufftPowerSpectrum(nmesh=(ngen, ngen, ngen//2), boxsize=L)

    powerspectrum.powerspectrum_field(pos_red)


if __name__ == "__main__":
    main()
