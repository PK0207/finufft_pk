#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>

namespace nb = nanobind;

long long folded_sq(long long idx, long long n) {
    long long folded = idx - n / 2; // n is the length of all axes except the last
    return folded * folded;
}

// This function unravels the index of a flattened array and takes the folded square of the indices.
// It is the first step to calculating the magnitude of the perpendicular component of the wavevector in Fourier space.
long long perp2_from_flat(long long flat, const std::vector<int64_t>& nmesh) {
    // fold shape is length of last axis of field, which is half the nmesh of input field
    long long total = 0;
    for (size_t axis = nmesh.size() - 1; axis > 0; --axis) {
        long long n = nmesh[axis - 1];
        long long idx = flat % n;
        flat /= n;
        total += folded_sq(idx, n);
    }
    return total;
}

template <typename T> //Allows for both float and double versions of the function to be compiled
void bin_column(
    long long perp2, long long len_z,
    const T* kedges2, long long n_kedges2,
    const T* muedges2,
    const T* raw_power_col,
    T* counts, T* weighted_counts, long long Nmu
){
    long long bk = 0;
    long long bmu = 0;
    for (long long i = 0; i < len_z; ++i) {
        long long k2 = i*i;
        long long mag = perp2 + (i*i);
        T mu2;
        if (mag > 0) {
            T invkmag2 = T(1)/T(mag);
            mu2 = k2 * invkmag2;
        } 
        else {
            mu2 = 0;
        }
        if (mag < kedges2[0]){
            continue;
        }
        else if (mag >= kedges2[n_kedges2-1]) {
            break;
        }
        while (mag > kedges2[bk+1]) {
            bk += 1;
        }
        while (mu2 > muedges2[bmu+1]) {
            bmu += 1;
        }
        int weight;
        if (i == 0) {
            weight = 1;
        } else {
            weight = 2;
        }
        counts[bk * Nmu + bmu] += weight; //mutated in place
        weighted_counts[bk * Nmu + bmu] += weight * raw_power_col[i];
        }
}

// function reads in numpy arays using nanobind ndarray and passes it to binning function
template <typename T>
void kmu_binning_cpp(
    nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu> kedges2,
    nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu> muedges2,
    nb::ndarray<const T, nb::c_contig, nb::device::cpu> raw_power,  // shape is (*fold_shape, len_z)
    nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> counts,
    nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> weighted_counts,
    int nthread
){
    // loop over all columns of the field and bin each column
    size_t ndim = raw_power.ndim();
    int N_kbin = kedges2.shape(0) - 1;
    int N_mubin = muedges2.shape(0) - 1;
    long long len_z = raw_power.shape(ndim-1);
    long long perp_volume = 1; //The first ndim-1 shapes multiplied to flatten the array indices
    std::vector<int64_t> nmesh(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        nmesh[d] = raw_power.shape(d);
        if (d + 1 < ndim) perp_volume *= nmesh[d];
    }
    const T* raw_power_ptr = raw_power.data();
    T* counts_ptr = counts.data();
    T* weighted_counts_ptr = weighted_counts.data();

    if (nthread <= 0) {
        nthread = omp_get_max_threads();
    }
    else if (nthread > omp_get_max_threads()) {
        nthread = omp_get_max_threads();
    }
    else {
        nthread = nthread;
    }

    #pragma omp parallel for schedule(static) num_threads(nthread) reduction(+:counts_ptr[:N_kbin*N_mubin]) reduction(+:weighted_counts_ptr[:N_kbin*N_mubin])
    for (long long flat = 0; flat < perp_volume; ++flat){
        long long perp2 = perp2_from_flat(flat, nmesh);
        const T* col_ptr = raw_power_ptr + flat * len_z; // this column's len_z contiguous values

        bin_column(
            perp2,
            len_z,
            kedges2.data(),
            kedges2.shape(0),
            muedges2.data(),
            col_ptr,
            counts_ptr, 
            weighted_counts_ptr,
            counts.shape(1)
        );
    }
}

// bind the C++ functions to Python using nanobind.
NB_MODULE(_binning, m) {
    m.def("kmu_binning_cpp_f32", &kmu_binning_cpp<float>);
    m.def("kmu_binning_cpp_f64", &kmu_binning_cpp<double>);
}