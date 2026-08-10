#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <omp.h>
#include <complex>

namespace nb = nanobind;

template <typename T>
void rescale_points_inplace(
    nb::ndarray<T, nb::c_contig, nb::device::cpu> points,
    float boxsize
) {
    T* points_ptr = points.data();
    long long n_points = points.size();
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n_points; ++i) {
        points_ptr[i] *= 2*M_PI/boxsize;
    }
}

template <typename T>
void realify_weights_inplace(
    nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu> positions_last,
    long long shift,
    nb::ndarray<std::complex<T>, nb::ndim<1>, nb::c_contig, nb::device::cpu> out
) {
    const T* pos_ptr = positions_last.data();
    std::complex<T>* out_ptr = out.data();
    long long n = positions_last.shape(0);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i) {
        T angle = -T(shift) * pos_ptr[i];
        out_ptr[i] = std::exp(std::complex<T>(0, angle)); 
    }
}

NB_MODULE(_helper_functions, m) {
    m.def("rescale_points_f32", &rescale_points_inplace<float>);
    m.def("rescale_points_f64", &rescale_points_inplace<double>);
    m.def("realify_weights_f32", &realify_weights_inplace<float>);
    m.def("realify_weights_f64", &realify_weights_inplace<double>);
}
