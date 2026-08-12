from finufft_pk.power_gpu import FinufftPk, powerspectrum_field
import cupy as cp
import pickle
import time

def main():
    fname = '/mnt/home/pkottapalli/ceph/positions_grid_512_nbar1_P01.pkl'
    with open(fname, 'rb') as f:
        pos_dict = pickle.load(f)
    ngen = pos_dict['ngen']
    xngen = ngen
    yngen = ngen
    zngen = ngen
    # stored as (N, D); set_positions needs (D, N)
    pos_red = cp.ascontiguousarray(cp.asarray(pos_dict['pos']).T, dtype=cp.float32)
    L = pos_dict['L']
    Nred = len(pos_red[0])

    print('Initializing FINUFFT Plan')
    start = time.time()
    powerspectrum = FinufftPk(nmesh=(xngen, yngen, zngen), boxsize=L)
    end = time.time()
    print(f"Plan time {end-start}")
    print("setting positions")
    start = time.time()
    powerspectrum.set_positions(positions=pos_red, inplace=True)
    end = time.time()
    print(f"setpts time {end-start}")
    print("computing field")
    weights = cp.ascontiguousarray(
                    cp.ones(shape=(Nred)), dtype=cp.complex64
                )
    out = cp.zeros((xngen, yngen, zngen//2+1), dtype=cp.complex64)
    start = time.time()
    field = powerspectrum.compute_field(weights, out)
    end = time.time()
    print(f"field time {end-start}")
    print("computing bandpowers")
    start = time.time()
    k_binc, counts, weighted_counts, bandpowers = powerspectrum.compute_bandpower(field=field)
    end = time.time()
    print(f"bandpower time {end-start}")
    print(powerspectrum.result)

    ps_result = powerspectrum_field((xngen, yngen, zngen), L, pos_red)
    with open('test_power_calc.pkl', 'wb') as file:
        pickle.dump((weighted_counts, counts, bandpowers), file)


if __name__ == "__main__":
    main()
