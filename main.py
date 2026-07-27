from finufft_powerspectrum import FinufftPowerSpectrum
import pickle 

def main():
    print("Hello from finufft-pk!")


# if __name__ == "__main__":
powerspectrum = FinufftPowerSpectrum(nmesh = (64, 64, 64), boxsize=250)
#read in data
fname = '/mnt/home/pkottapalli/fft_benchmark/positions_grid_64_nbar1000_P00.05.pkl'
with open(fname, 'rb') as f:
    pos_dict = pickle.load(f)
    ngen = pos_dict['ngen']
    pos_red = pos_dict['pos']
    N_red = len(pos_red)
    L = pos_dict['L']
    P0 = pos_dict['P0']
    n_index = pos_dict['n_index']
    nbar = pos_dict['nbar']

print('setting positions')
powerspectrum.set_positions(positions=pos_red)
print('computing field')
field = powerspectrum.compute_field()
print('computing bandpowers')
powerspectrum.compute_bandpower(field)
main()
