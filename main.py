from src.finufft_pk.finufft_powerspectrum import FinufftPowerSpectrum, powerspectrum_field
import numpy as np
import pickle

def main():
    fname = '/mnt/home/pkottapalli/ceph/positions_grid_512_nbar1_P01.pkl'
    with open(fname, 'rb') as f:
        pos_dict = pickle.load(f)
    ngen = pos_dict['ngen']
    # stored as (N, 3); set_positions needs (D, N)
    pos_red = np.ascontiguousarray(pos_dict['pos'].T, dtype='float32')
    L = pos_dict['L']

    # powerspectrum = FinufftPowerSpectrum(nmesh=(ngen, ngen, ngen), boxsize=L)

    _, counts, weighted_counts, bandpowers = powerspectrum_field((ngen, ngen, ngen), L, pos_red)
    with open('test_power_calc.pkl', 'wb') as file:
        pickle.dump((weighted_counts, counts, bandpowers), file)


if __name__ == "__main__":
    main()
