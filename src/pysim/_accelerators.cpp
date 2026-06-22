// M_PI is not in the C++ standard. GCC/glibc define it unconditionally, but
// MSVC only exposes it when _USE_MATH_DEFINES is set *before* the first math
// header is pulled in (directly or transitively via pybind11). Must stay at
// the very top of the file.
#define _USE_MATH_DEFINES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <complex>

#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include "_bspline_static_moments_inline.h"

namespace py = pybind11;

// Ubuntu/glibc <cmath> headers don't carry `omp declare simd` markers for the
// libmvec routines, so GCC's auto-vectorizer can't substitute the vectorized
// `_ZGVdN4v_sin` / `_ZGVdN4v_cos` (AVX2, 4 doubles) inside an `omp simd` loop
// without these explicit declarations. The std::cos / std::sin overloads in
// <cmath> still resolve to these underlying extern-C symbols, so the rest of
// the file's calls pick up the simd-vectorized form for free once the linker
// has libmvec available (-lmvec in setup.py).
//
// Gated to GNU-compatible, non-MSVC compilers: this trick targets glibc's
// libmvec specifically. MSVC has no libmvec and would choke on redeclaring the
// CRT's cos/sin; there the sincos calls stay scalar/autovectorized.
#if defined(__GNUC__) && !defined(_MSC_VER)
#pragma omp declare simd notinbranch simdlen(4)
extern "C" double cos(double);

#pragma omp declare simd notinbranch simdlen(4)
extern "C" double sin(double);
#endif

// The (i, j)-grid parallel loops below use `collapse(2)` for finer load
// balancing. MSVC builds with /openmp:experimental (required for the
// `#pragma omp simd` directives in the inner loops), which does NOT support the
// OpenMP 3.0 `collapse` clause. Fall back to plain outer-loop parallelism
// there — same results, only coarser scheduling across the grid. GCC keeps
// collapse(2).
#if defined(_MSC_VER)
#  define PYSIM_OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for schedule(static)")
#else
#  define PYSIM_OMP_PARALLEL_FOR_COLLAPSE2 \
       _Pragma("omp parallel for collapse(2) schedule(static)")
#endif

// Batched cross-segment Gauss-Legendre quadrature in 3D.
//
// For each k in k_array, and each (i, j) segment pair, compute:
//   J_pq[k, i, j] = sum_{q, r} w_i_q * u_i_q^p * G(k, R_qr) * w_j_r * u_j_r^q
// where
//   R_qr = sqrt(|pos_i(t_q) - pos_j(t_r)|^2 + a_squared)
//   G(k, R) = exp(-j*k*R) / (4*pi*R)
//   pos_i(t) = (1-t) * seg_l_i + t * seg_r_i  (similarly for j)
//   w_i_q = len_i * gl_w_q;   u_i_q = len_i * gl_t_q
//
// a_squared = a^2 for V's off-edge case (corner regularization), 0 for the
// Yagi cross-wire case (no shared point so the kernel is already non-singular).
//
// Parallelism: each (i, j) pair is independent — OpenMP over the (i, j) grid.
// For each pair we precompute R[q, r] once, then loop k over all wavenumbers,
// reusing R from the per-thread cache.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
seg_seg_quad_batch_3d(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_k = ka.shape(0);
    size_t n_qp = glt.shape(0);

    py::array_t<std::complex<double>> J00({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J10({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J01({n_k, N_i, N_j});
    py::array_t<std::complex<double>> J11({n_k, N_i, N_j});
    auto j00 = J00.mutable_unchecked<3>();
    auto j10 = J10.mutable_unchecked<3>();
    auto j01 = J01.mutable_unchecked<3>();
    auto j11 = J11.mutable_unchecked<3>();

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment quadrature-point positions and lengths -- k-independent so
    // compute once outside the parallel region.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            // n_qp <= 8 in practice, so n_qp^2 <= 64. Stack-allocated, aligned
            // for 32-byte AVX2 loads. The hot loop is the sincos at line ~Y
            // below; the explicit per-(qr) array layout lets `#pragma omp simd`
            // batch the sincos calls into the libmvec vectorized form
            // (_ZGVdN4v_sin / _ZGVdN4v_cos) instead of N_pairs individual calls.
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double wi_arr[64], wj_arr[64], ui_arr[64], uj_arr[64];

            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            size_t n_pairs = n_qp * n_qp;

            // k-independent precompute -- weights, u-coords, and 1/(4πR).
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                for (size_t r = 0; r < n_qp; r++) {
                    size_t qr = q*n_qp + r;
                    wi_arr[qr] = wi;
                    wj_arr[qr] = glw(r) * Lj;
                    ui_arr[qr] = ui;
                    uj_arr[qr] = glt(r) * Lj;
                }
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);

                // Stage 1: phase = -k * R, fully vectorizable.
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }

                // Stage 2: vectorized cos and sin via libmvec. Split into two
                // loops on purpose — if cos/sin appear in the same loop body
                // on the same input, GCC fuses them into a single scalar
                // `sincos` call (smart for serial code), but libmvec has no
                // vector `sincos`, only `_ZGVdN4v_cos` and `_ZGVdN4v_sin`.
                // Splitting keeps them as independent vectorizable calls.
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }

                // Stage 3: scalar reductions of the 4 J integrals. The compiler
                // vectorizes the per-component accumulation across qr.
                double s00_re = 0.0, s00_im = 0.0;
                double s10_re = 0.0, s10_im = 0.0;
                double s01_re = 0.0, s01_im = 0.0;
                double s11_re = 0.0, s11_im = 0.0;
                #pragma omp simd reduction(+:s00_re,s00_im,s10_re,s10_im,s01_re,s01_im,s11_re,s11_im)
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double iR = inv_R_4pi[qr];
                    double G_re = cos_phases[qr] * iR;
                    double G_im = sin_phases[qr] * iR;
                    double wij  = wi_arr[qr] * wj_arr[qr];
                    double wG_re = wij * G_re;
                    double wG_im = wij * G_im;
                    s00_re += wG_re;
                    s00_im += wG_im;
                    s10_re += ui_arr[qr] * wG_re;
                    s10_im += ui_arr[qr] * wG_im;
                    s01_re += uj_arr[qr] * wG_re;
                    s01_im += uj_arr[qr] * wG_im;
                    double uij = ui_arr[qr] * uj_arr[qr];
                    s11_re += uij * wG_re;
                    s11_im += uij * wG_im;
                }
                j00(kk, i, j) = std::complex<double>(s00_re, s00_im);
                j10(kk, i, j) = std::complex<double>(s10_re, s10_im);
                j01(kk, i, j) = std::complex<double>(s01_re, s01_im);
                j11(kk, i, j) = std::complex<double>(s11_re, s11_im);
            }
        }
    }

    return std::make_tuple(J00, J10, J01, J11);
}


// Batched same-wire Gauss-Legendre quadrature on the *regularized* kernel
//   G_reg(R, k) = (exp(-j k R) - 1) / (4 pi R),    R = sqrt(diff^2 + a^2)
// for all segment pairs (i, j) on a shared 1D arc-length line.
//
// Inputs:
//   seg_endpoints : (N+1,) arc-length / position of segment boundaries
//   a             : wire radius (corner regularization)
//   k_array       : (n_k,) wavenumbers
//   gl_t, gl_w    : Gauss-Legendre nodes/weights mapped to [0, 1]
//                   (i.e. t = (xi + 1)/2 and w = w_legendre / 2)
//
// For each pair (i, j) we precompute the per-pair R table once and reuse it
// across the k axis. OpenMP parallelizes over (i, j).
//
// Output:
//   (J00, J10, J01, J11), each (n_k, N, N) complex.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
seg_seg_reg_quad_batch_1d(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_endpoints,
    double a,
    py::array_t<double, py::array::c_style | py::array::forcecast> k_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto se  = seg_endpoints.unchecked<1>();
    auto ka  = k_array.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    if (se.shape(0) < 2) {
        throw std::runtime_error("seg_endpoints must have at least 2 entries");
    }

    size_t N    = se.shape(0) - 1;
    size_t n_k  = ka.shape(0);
    size_t n_qp = glt.shape(0);
    if (n_qp * n_qp > 64) {
        throw std::runtime_error("n_qp too large (n_qp^2 must be <= 64)");
    }

    const double a_sq    = a * a;
    const double inv_4pi = 1.0 / (4.0 * M_PI);

    py::array_t<std::complex<double>> J00({n_k, N, N});
    py::array_t<std::complex<double>> J10({n_k, N, N});
    py::array_t<std::complex<double>> J01({n_k, N, N});
    py::array_t<std::complex<double>> J11({n_k, N, N});
    auto j00 = J00.mutable_unchecked<3>();
    auto j10 = J10.mutable_unchecked<3>();
    auto j01 = J01.mutable_unchecked<3>();
    auto j11 = J11.mutable_unchecked<3>();

    // Per-segment quadrature positions and lengths.
    std::vector<double> pos(N * n_qp);
    std::vector<double> Lvec(N);
    for (size_t i = 0; i < N; i++) {
        double sl = se(i);
        double sr = se(i + 1);
        double Li = sr - sl;
        Lvec[i] = Li;
        for (size_t q = 0; q < n_qp; q++) {
            pos[i * n_qp + q] = sl + Li * glt(q);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            // Same split-loop layout as seg_seg_quad_batch_3d: the per-(qr)
            // arrays let `#pragma omp simd` substitute libmvec's
            // _ZGVdN4v_cos / _ZGVdN4v_sin for the inner sincos calls.
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double wi_arr[64], wj_arr[64], ui_arr[64], uj_arr[64];

            double Li = Lvec[i];
            double Lj = Lvec[j];
            const double *pi_q = &pos[i * n_qp];
            const double *pj_r = &pos[j * n_qp];
            for (size_t q = 0; q < n_qp; q++) {
                double piq = pi_q[q];
                for (size_t r = 0; r < n_qp; r++) {
                    double d = piq - pj_r[r];
                    R[q * n_qp + r] = std::sqrt(d * d + a_sq);
                }
            }

            size_t n_pairs = n_qp * n_qp;
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                for (size_t r = 0; r < n_qp; r++) {
                    size_t qr = q * n_qp + r;
                    wi_arr[qr] = wi;
                    wj_arr[qr] = glw(r) * Lj;
                    ui_arr[qr] = ui;
                    uj_arr[qr] = glt(r) * Lj;
                }
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
            }

            for (size_t kk = 0; kk < n_k; kk++) {
                double k = ka(kk);

                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    phases[qr] = -k * R[qr];
                }
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    cos_phases[qr] = std::cos(phases[qr]);
                }
                #pragma omp simd
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sin_phases[qr] = std::sin(phases[qr]);
                }

                // exp(-j k R) - 1 = (cos(-kR) - 1) + j sin(-kR)
                double s00_re = 0.0, s00_im = 0.0;
                double s10_re = 0.0, s10_im = 0.0;
                double s01_re = 0.0, s01_im = 0.0;
                double s11_re = 0.0, s11_im = 0.0;
                #pragma omp simd reduction(+:s00_re,s00_im,s10_re,s10_im,s01_re,s01_im,s11_re,s11_im)
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    double iR = inv_R_4pi[qr];
                    double Greg_re = (cos_phases[qr] - 1.0) * iR;
                    double Greg_im = sin_phases[qr] * iR;
                    double wij = wi_arr[qr] * wj_arr[qr];
                    double wG_re = wij * Greg_re;
                    double wG_im = wij * Greg_im;
                    s00_re += wG_re;
                    s00_im += wG_im;
                    s10_re += ui_arr[qr] * wG_re;
                    s10_im += ui_arr[qr] * wG_im;
                    s01_re += uj_arr[qr] * wG_re;
                    s01_im += uj_arr[qr] * wG_im;
                    double uij = ui_arr[qr] * uj_arr[qr];
                    s11_re += uij * wG_re;
                    s11_im += uij * wG_im;
                }
                j00(kk, i, j) = std::complex<double>(s00_re, s00_im);
                j10(kk, i, j) = std::complex<double>(s10_re, s10_im);
                j01(kk, i, j) = std::complex<double>(s01_re, s01_im);
                j11(kk, i, j) = std::complex<double>(s11_re, s11_im);
            }
        }
    }

    return std::make_tuple(J00, J10, J01, J11);
}


// Assemble the (n_k, n_basis, n_basis) Z matrix from the four J tensors,
// per-segment h, the segment tangent-dot-product table, the left/right
// segment index of each basis, and omega(k).
//
// For each basis pair (m, n) and each wavenumber k:
//   Z[k, m, n] = (j w mu) * I_A + S / (j w eps)
// with
//   m_l = left_seg[m],  m_r = right_seg[m]
//   n_l = left_seg[n],  n_r = right_seg[n]
//   h*_m = h[m_*],      h*_n = h[n_*]
//   td_xy = td_all[m_x, n_y]
//   S   =  J00_ll/(hl_m hl_n) - J00_lr/(hl_m hr_n)
//        - J00_rl/(hr_m hl_n) + J00_rr/(hr_m hr_n)
//   I_A =  td_ll *  J11_ll/(hl_m hl_n)
//        + td_lr * (J10_lr/hl_m   - J11_lr/(hl_m hr_n))
//        + td_rl * (J01_rl/hl_n   - J11_rl/(hr_m hl_n))
//        + td_rr * (J00_rr - J10_rr/hr_m - J01_rr/hr_n + J11_rr/(hr_m hr_n))
//
// Loop order: collapse(2) parallel for (k, m); the inner n loop reads
// contiguous rows of the J tensors (when left_seg/right_seg are consecutive,
// which is the common case for both V and Yagi within a wire).
static py::array_t<std::complex<double>>
assemble_Z(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J00,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J10,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J01,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J11,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> left_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> right_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps,
    double mu
) {
    auto j00_info = J00.request();
    auto j10_info = J10.request();
    auto j01_info = J01.request();
    auto j11_info = J11.request();

    if (j00_info.ndim != 3) throw std::runtime_error("J tensors must be 3D");
    size_t n_k = (size_t)j00_info.shape[0];
    size_t N   = (size_t)j00_info.shape[1];
    if ((size_t)j00_info.shape[2] != N) {
        throw std::runtime_error("J tensors must be square (n_k, N, N)");
    }

    auto h   = h_per_seg.unchecked<1>();
    auto td  = td_all.unchecked<2>();
    auto ls  = left_seg.unchecked<1>();
    auto rs  = right_seg.unchecked<1>();
    auto om  = omega_array.unchecked<1>();

    if ((size_t)h.shape(0) != N) {
        throw std::runtime_error("h_per_seg length must match J tensor N");
    }
    if ((size_t)td.shape(0) != N || (size_t)td.shape(1) != N) {
        throw std::runtime_error("td_all must be (N, N)");
    }
    if (ls.shape(0) != rs.shape(0)) {
        throw std::runtime_error("left_seg / right_seg must have matching length");
    }
    if ((size_t)om.shape(0) != n_k) {
        throw std::runtime_error("omega_array length must match n_k");
    }
    size_t n_basis = (size_t)ls.shape(0);

    py::array_t<std::complex<double>> Z({n_k, n_basis, n_basis});
    auto z_info = Z.request();

    const std::complex<double>* j00_ptr = static_cast<const std::complex<double>*>(j00_info.ptr);
    const std::complex<double>* j10_ptr = static_cast<const std::complex<double>*>(j10_info.ptr);
    const std::complex<double>* j01_ptr = static_cast<const std::complex<double>*>(j01_info.ptr);
    const std::complex<double>* j11_ptr = static_cast<const std::complex<double>*>(j11_info.ptr);
    std::complex<double>* z_ptr = static_cast<std::complex<double>*>(z_info.ptr);

    const std::complex<double> j_unit(0.0, 1.0);

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t kk = 0; kk < n_k; kk++) {
        for (size_t m = 0; m < n_basis; m++) {
            int64_t m_l = ls(m);
            int64_t m_r = rs(m);
            double hl_m = h(m_l);
            double hr_m = h(m_r);
            double inv_hl_m = 1.0 / hl_m;
            double inv_hr_m = 1.0 / hr_m;

            double omega_k = om(kk);
            std::complex<double> jw_mu    = j_unit * (omega_k * mu);
            std::complex<double> inv_jw_eps = 1.0 / (j_unit * (omega_k * eps));

            size_t base_kk = kk * N * N;
            const std::complex<double> *j00_ml = j00_ptr + base_kk + (size_t)m_l * N;
            const std::complex<double> *j00_mr = j00_ptr + base_kk + (size_t)m_r * N;
            const std::complex<double> *j10_ml = j10_ptr + base_kk + (size_t)m_l * N;
            const std::complex<double> *j10_mr = j10_ptr + base_kk + (size_t)m_r * N;
            const std::complex<double> *j01_mr = j01_ptr + base_kk + (size_t)m_r * N;
            const std::complex<double> *j11_ml = j11_ptr + base_kk + (size_t)m_l * N;
            const std::complex<double> *j11_mr = j11_ptr + base_kk + (size_t)m_r * N;

            const double *td_ml = &td(m_l, 0);
            const double *td_mr = &td(m_r, 0);

            std::complex<double> *z_row = z_ptr + kk * n_basis * n_basis + m * n_basis;

            for (size_t n = 0; n < n_basis; n++) {
                int64_t n_l = ls(n);
                int64_t n_r = rs(n);
                double inv_hl_n = 1.0 / h(n_l);
                double inv_hr_n = 1.0 / h(n_r);

                double td_ll = td_ml[n_l];
                double td_lr = td_ml[n_r];
                double td_rl = td_mr[n_l];
                double td_rr = td_mr[n_r];

                std::complex<double> J00_ll = j00_ml[n_l];
                std::complex<double> J00_lr = j00_ml[n_r];
                std::complex<double> J00_rl = j00_mr[n_l];
                std::complex<double> J00_rr = j00_mr[n_r];

                std::complex<double> J10_lr = j10_ml[n_r];
                std::complex<double> J10_rr = j10_mr[n_r];

                std::complex<double> J01_rl = j01_mr[n_l];
                std::complex<double> J01_rr = j01_mr[n_r];

                std::complex<double> J11_ll = j11_ml[n_l];
                std::complex<double> J11_lr = j11_ml[n_r];
                std::complex<double> J11_rl = j11_mr[n_l];
                std::complex<double> J11_rr = j11_mr[n_r];

                double inv_hlm_hln = inv_hl_m * inv_hl_n;
                double inv_hlm_hrn = inv_hl_m * inv_hr_n;
                double inv_hrm_hln = inv_hr_m * inv_hl_n;
                double inv_hrm_hrn = inv_hr_m * inv_hr_n;

                std::complex<double> S =
                      J00_ll * inv_hlm_hln
                    - J00_lr * inv_hlm_hrn
                    - J00_rl * inv_hrm_hln
                    + J00_rr * inv_hrm_hrn;

                std::complex<double> I_A =
                      td_ll * (J11_ll * inv_hlm_hln)
                    + td_lr * (J10_lr * inv_hl_m - J11_lr * inv_hlm_hrn)
                    + td_rl * (J01_rl * inv_hl_n - J11_rl * inv_hrm_hln)
                    + td_rr * (J00_rr - J10_rr * inv_hr_m - J01_rr * inv_hr_n + J11_rr * inv_hrm_hrn);

                z_row[n] = jw_mu * I_A + S * inv_jw_eps;
            }
        }
    }

    return Z;
}


// Assemble Z from per-basis (segment, L, R) support arrays (n_basis, 2). Each
// basis has up to 2 wings, with arbitrary level and slope per wing. Interior
// tent bases encode (left, right) as ((sm_l, 0, 1), (sm_r, 1, 0)); junction
// directional bases use wing 0 only and zero out wing 1 (L=R=0 → slope=0 → no
// contribution). The general 2×2 (a, b) sum per (m, n):
//
//   slope[m, a] = (R[m, a] - L[m, a]) / h[support_seg[m, a]]
//   S[k, m, n]   = Σ_a Σ_b slope[m,a]*slope[n,b] * J00[k, sm_a, sn_b]
//   I_A[k, m, n] = Σ_a Σ_b td_all[sm_a, sn_b] * (
//                     L[m,a]*L[n,b]*J00 + L[m,a]*slope[n,b]*J01
//                   + slope[m,a]*L[n,b]*J10 + slope[m,a]*slope[n,b]*J11 )
//   Z[k, m, n]   = jωμ I_A + S / (jωε)
//
// Parallel collapse(2) over (k, m); inner n loop reads contiguous J rows.
static py::array_t<std::complex<double>>
assemble_Z_general(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J00,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J10,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J01,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J11,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> support_L,
    py::array_t<double, py::array::c_style | py::array::forcecast> support_R,
    py::array_t<double, py::array::c_style | py::array::forcecast> omega_array,
    double eps,
    double mu
) {
    auto j00_info = J00.request();
    auto j10_info = J10.request();
    auto j01_info = J01.request();
    auto j11_info = J11.request();

    if (j00_info.ndim != 3) throw std::runtime_error("J tensors must be 3D");
    size_t n_k = (size_t)j00_info.shape[0];
    size_t N   = (size_t)j00_info.shape[1];
    if ((size_t)j00_info.shape[2] != N) {
        throw std::runtime_error("J tensors must be square (n_k, N, N)");
    }

    auto h   = h_per_seg.unchecked<1>();
    auto td  = td_all.unchecked<2>();
    auto ssg = support_seg.unchecked<2>();
    auto sL  = support_L.unchecked<2>();
    auto sR  = support_R.unchecked<2>();
    auto om  = omega_array.unchecked<1>();

    if ((size_t)h.shape(0) != N) {
        throw std::runtime_error("h_per_seg length must match J tensor N");
    }
    if ((size_t)td.shape(0) != N || (size_t)td.shape(1) != N) {
        throw std::runtime_error("td_all must be (N, N)");
    }
    if (ssg.shape(1) != 2 || sL.shape(1) != 2 || sR.shape(1) != 2) {
        throw std::runtime_error("support_seg/L/R must have shape (n_basis, 2)");
    }
    if (ssg.shape(0) != sL.shape(0) || ssg.shape(0) != sR.shape(0)) {
        throw std::runtime_error("support_seg/L/R must have matching n_basis");
    }
    if ((size_t)om.shape(0) != n_k) {
        throw std::runtime_error("omega_array length must match n_k");
    }
    size_t n_basis = (size_t)ssg.shape(0);

    std::vector<int64_t> ssg_flat(n_basis * 2);
    std::vector<double>  L_flat(n_basis * 2);
    std::vector<double>  slope_flat(n_basis * 2);
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t a = 0; a < 2; a++) {
            int64_t s = ssg(m, a);
            double Lv = sL(m, a);
            double Rv = sR(m, a);
            ssg_flat[m * 2 + a] = s;
            L_flat[m * 2 + a]   = Lv;
            slope_flat[m * 2 + a] = (Rv - Lv) / h(s);
        }
    }

    py::array_t<std::complex<double>> Z({n_k, n_basis, n_basis});
    auto z_info = Z.request();

    const std::complex<double>* j00_ptr = static_cast<const std::complex<double>*>(j00_info.ptr);
    const std::complex<double>* j10_ptr = static_cast<const std::complex<double>*>(j10_info.ptr);
    const std::complex<double>* j01_ptr = static_cast<const std::complex<double>*>(j01_info.ptr);
    const std::complex<double>* j11_ptr = static_cast<const std::complex<double>*>(j11_info.ptr);
    std::complex<double>* z_ptr = static_cast<std::complex<double>*>(z_info.ptr);

    const std::complex<double> j_unit(0.0, 1.0);

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t kk = 0; kk < n_k; kk++) {
        for (size_t m = 0; m < n_basis; m++) {
            double omega_k = om(kk);
            std::complex<double> jw_mu      = j_unit * (omega_k * mu);
            std::complex<double> inv_jw_eps = 1.0 / (j_unit * (omega_k * eps));

            int64_t sm0 = ssg_flat[m * 2 + 0];
            int64_t sm1 = ssg_flat[m * 2 + 1];
            double Lm0 = L_flat[m * 2 + 0];
            double Lm1 = L_flat[m * 2 + 1];
            double Sm0 = slope_flat[m * 2 + 0];
            double Sm1 = slope_flat[m * 2 + 1];

            size_t base_kk = kk * N * N;
            const std::complex<double> *j00_m0 = j00_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j00_m1 = j00_ptr + base_kk + (size_t)sm1 * N;
            const std::complex<double> *j10_m0 = j10_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j10_m1 = j10_ptr + base_kk + (size_t)sm1 * N;
            const std::complex<double> *j01_m0 = j01_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j01_m1 = j01_ptr + base_kk + (size_t)sm1 * N;
            const std::complex<double> *j11_m0 = j11_ptr + base_kk + (size_t)sm0 * N;
            const std::complex<double> *j11_m1 = j11_ptr + base_kk + (size_t)sm1 * N;

            const double *td_m0 = &td(sm0, 0);
            const double *td_m1 = &td(sm1, 0);

            std::complex<double> *z_row = z_ptr + kk * n_basis * n_basis + m * n_basis;

            for (size_t n = 0; n < n_basis; n++) {
                int64_t sn0 = ssg_flat[n * 2 + 0];
                int64_t sn1 = ssg_flat[n * 2 + 1];
                double Ln0 = L_flat[n * 2 + 0];
                double Ln1 = L_flat[n * 2 + 1];
                double Sn0 = slope_flat[n * 2 + 0];
                double Sn1 = slope_flat[n * 2 + 1];

                std::complex<double> J00_00 = j00_m0[sn0];
                std::complex<double> J01_00 = j01_m0[sn0];
                std::complex<double> J10_00 = j10_m0[sn0];
                std::complex<double> J11_00 = j11_m0[sn0];
                double td00 = td_m0[sn0];

                std::complex<double> J00_01 = j00_m0[sn1];
                std::complex<double> J01_01 = j01_m0[sn1];
                std::complex<double> J10_01 = j10_m0[sn1];
                std::complex<double> J11_01 = j11_m0[sn1];
                double td01 = td_m0[sn1];

                std::complex<double> J00_10 = j00_m1[sn0];
                std::complex<double> J01_10 = j01_m1[sn0];
                std::complex<double> J10_10 = j10_m1[sn0];
                std::complex<double> J11_10 = j11_m1[sn0];
                double td10 = td_m1[sn0];

                std::complex<double> J00_11 = j00_m1[sn1];
                std::complex<double> J01_11 = j01_m1[sn1];
                std::complex<double> J10_11 = j10_m1[sn1];
                std::complex<double> J11_11 = j11_m1[sn1];
                double td11 = td_m1[sn1];

                std::complex<double> S =
                      (Sm0 * Sn0) * J00_00
                    + (Sm0 * Sn1) * J00_01
                    + (Sm1 * Sn0) * J00_10
                    + (Sm1 * Sn1) * J00_11;

                std::complex<double> I_A =
                      td00 * (Lm0*Ln0*J00_00 + Lm0*Sn0*J01_00 + Sm0*Ln0*J10_00 + Sm0*Sn0*J11_00)
                    + td01 * (Lm0*Ln1*J00_01 + Lm0*Sn1*J01_01 + Sm0*Ln1*J10_01 + Sm0*Sn1*J11_01)
                    + td10 * (Lm1*Ln0*J00_10 + Lm1*Sn0*J01_10 + Sm1*Ln0*J10_10 + Sm1*Sn0*J11_10)
                    + td11 * (Lm1*Ln1*J00_11 + Lm1*Sn1*J01_11 + Sm1*Ln1*J10_11 + Sm1*Sn1*J11_11);

                z_row[n] = jw_mu * I_A + S * inv_jw_eps;
            }
        }
    }

    return Z;
}


// Templated B-spline moment-integral kernel.
//
// For each (i, j) segment pair, compute the (D+1)^2 polynomial moments
//   J[p, P, i, j] = sum_{q, r} wi[q] * ui[q]^p * wj[r] * uj[r]^P * G(R_qr)
// where
//   R_qr = sqrt(|pos_i(t_q) - pos_j(t_r)|^2 + a_squared)
//   G(R) = exp(-j*k*R) / (4*pi*R)
//   ui[q] = t_q * len_i,  uj[r] = t_r * len_j  (local arc lengths)
//
// Used by BSplinePySim._build_J_blocks for the all-pairs off-edge piece
// (same a^2 wire-radius regularization handles touching segments at kinks
// and at junctions). Single-k for now (BSplinePySim hasn't grown a swept
// path yet); add a batched k_array variant later if/when needed.
//
// Template parameter D = B-spline degree (1 or 2 currently — explicit
// instantiations below). Hardcoding D as a compile-time constant lets the
// compiler fully unroll the (D+1)^2 polynomial-moment inner loop, getting
// the same scalar-unrolled tight assembly the d=1 seg_seg_quad_batch_3d
// achieves with hand-rolled s00 / s10 / s01 / s11 accumulators.
//
// n_qp <= 8 assumed (n_qp^2 <= 64 scratch buffer size).
template<int D>
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline_kernel(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    static constexpr int NM = D + 1;          // moments per axis
    static constexpr int NMM = NM * NM;       // total moments

    auto sli = seg_l_i.unchecked<2>();
    auto sri = seg_r_i.unchecked<2>();
    auto slj = seg_l_j.unchecked<2>();
    auto srj = seg_r_j.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (sli.shape(1) != 3 || sri.shape(1) != 3 ||
        slj.shape(1) != 3 || srj.shape(1) != 3) {
        throw std::runtime_error("segment endpoint arrays must have shape (N, 3)");
    }
    if (sli.shape(0) != sri.shape(0) || slj.shape(0) != srj.shape(0)) {
        throw std::runtime_error("seg_l and seg_r must have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }
    size_t n_qp_in = glt.shape(0);
    if (n_qp_in > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    size_t N_i = sli.shape(0);
    size_t N_j = slj.shape(0);
    size_t n_qp = n_qp_in;

    py::array_t<std::complex<double>> J({(size_t)NM, (size_t)NM, N_i, N_j});
    auto j_view = J.mutable_unchecked<4>();

    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Per-segment quadrature-point positions and lengths -- k-independent,
    // computed once outside the parallel region.
    std::vector<double> pos_i(N_i * n_qp * 3);
    std::vector<double> pos_j(N_j * n_qp * 3);
    std::vector<double> len_i(N_i);
    std::vector<double> len_j(N_j);
    for (size_t i = 0; i < N_i; i++) {
        double dx = sri(i,0) - sli(i,0);
        double dy = sri(i,1) - sli(i,1);
        double dz = sri(i,2) - sli(i,2);
        len_i[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            pos_i[(i*n_qp + q)*3 + 0] = (1.0 - t) * sli(i,0) + t * sri(i,0);
            pos_i[(i*n_qp + q)*3 + 1] = (1.0 - t) * sli(i,1) + t * sri(i,1);
            pos_i[(i*n_qp + q)*3 + 2] = (1.0 - t) * sli(i,2) + t * sri(i,2);
        }
    }
    for (size_t j = 0; j < N_j; j++) {
        double dx = srj(j,0) - slj(j,0);
        double dy = srj(j,1) - slj(j,1);
        double dz = srj(j,2) - slj(j,2);
        len_j[j] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t r = 0; r < n_qp; r++) {
            double t = glt(r);
            pos_j[(j*n_qp + r)*3 + 0] = (1.0 - t) * slj(j,0) + t * srj(j,0);
            pos_j[(j*n_qp + r)*3 + 1] = (1.0 - t) * slj(j,1) + t * srj(j,1);
            pos_j[(j*n_qp + r)*3 + 2] = (1.0 - t) * slj(j,2) + t * srj(j,2);
        }
    }

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t i = 0; i < N_i; i++) {
        for (size_t j = 0; j < N_j; j++) {
            alignas(32) double R[64];
            alignas(32) double inv_R_4pi[64];
            alignas(32) double phases[64];
            alignas(32) double cos_phases[64];
            alignas(32) double sin_phases[64];
            alignas(32) double G_re[64], G_im[64];
            // wuwu[pP, qr]: precomputed wi[q]*ui[q]^p * wj[r]*uj[r]^P,
            // flattened with pP = p*NM + P. For D=2: NMM*64 = 576 doubles = 4.5KB,
            // fits comfortably in L1.
            alignas(32) double wuwu[NMM * 64];

            const double *pi = &pos_i[i * n_qp * 3];
            const double *pj = &pos_j[j * n_qp * 3];
            for (size_t q = 0; q < n_qp; q++) {
                double pix = pi[q*3 + 0];
                double piy = pi[q*3 + 1];
                double piz = pi[q*3 + 2];
                for (size_t r = 0; r < n_qp; r++) {
                    double dx = pix - pj[r*3 + 0];
                    double dy = piy - pj[r*3 + 1];
                    double dz = piz - pj[r*3 + 2];
                    R[q*n_qp + r] = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                }
            }

            double Li = len_i[i];
            double Lj = len_j[j];
            size_t n_pairs = n_qp * n_qp;

            // Build wuwu[pP, qr]: pP indexes the moment (p*NM + P), qr the
            // quadrature pair (q*n_qp + r). The NM is a template constant so
            // ui_pow[p] / uj_pow[P] arrays unroll.
            for (size_t q = 0; q < n_qp; q++) {
                double wi = glw(q) * Li;
                double ui = glt(q) * Li;
                double ui_pow[NM];
                ui_pow[0] = 1.0;
                for (int p = 1; p < NM; p++) ui_pow[p] = ui_pow[p-1] * ui;
                for (size_t r = 0; r < n_qp; r++) {
                    double wj = glw(r) * Lj;
                    double uj = glt(r) * Lj;
                    double uj_pow[NM];
                    uj_pow[0] = 1.0;
                    for (int P = 1; P < NM; P++) uj_pow[P] = uj_pow[P-1] * uj;
                    double wij = wi * wj;
                    size_t qr = q * n_qp + r;
                    for (int p = 0; p < NM; p++) {
                        for (int P = 0; P < NM; P++) {
                            wuwu[(p * NM + P) * n_pairs + qr] = wij * ui_pow[p] * uj_pow[P];
                        }
                    }
                }
            }

            // Stage 1: phases = -k * R, then sincos via libmvec.
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                phases[qr] = -k * R[qr];
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                cos_phases[qr] = std::cos(phases[qr]);
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                sin_phases[qr] = std::sin(phases[qr]);
            }
            #pragma omp simd
            for (size_t qr = 0; qr < n_pairs; qr++) {
                inv_R_4pi[qr] = inv_4pi / R[qr];
                G_re[qr] = cos_phases[qr] * inv_R_4pi[qr];
                G_im[qr] = sin_phases[qr] * inv_R_4pi[qr];
            }

            // Stage 2: NMM moment reductions, each a vectorizable sum over qr.
            for (int pP = 0; pP < NMM; pP++) {
                double sr = 0.0, si = 0.0;
                const double *w_row = &wuwu[pP * n_pairs];
                #pragma omp simd reduction(+:sr,si)
                for (size_t qr = 0; qr < n_pairs; qr++) {
                    sr += w_row[qr] * G_re[qr];
                    si += w_row[qr] * G_im[qr];
                }
                j_view(pP / NM, pP % NM, i, j) = std::complex<double>(sr, si);
            }
        }
    }

    return J;
}

// Templated B-spline Z assembly kernel.
//
// For each (m, n) basis pair, assembles the EFIE Galerkin entry from the
// polynomial-moment tensor J and the per-(basis, wing, poly-degree)
// coefficient table:
//   Z[m,n] = j*omega*mu * sum_{a,b} (t·t)[sm, sn]
//            * sum_{p, q} polys[m, a, p] * polys[n, b, q] * J[p, q, sm, sn]
//          + (1/jωε)    * sum_{a,b}
//            * sum_{p≥1, q≥1} p*q * polys[m, a, p] * polys[n, b, q]
//                           * J[p-1, q-1, sm, sn]
// where sm = support_seg[m, a], sn = support_seg[n, b].
//
// Inactive wings of boundary / junction-directional bases have polys = 0
// at every p, so they contribute nothing — no special handling needed.
//
// Template parameter D = B-spline degree (1 or 2). NM = D+1 wings per basis
// and D+1 polynomial moments per wing. Hardcoding NM as a compile-time
// constant unrolls the (D+1)^4 inner muladd loop.
//
// Single-k for now (BSplinePySim doesn't have a swept path yet); the inputs
// are scalar omega instead of an omega_array.
template<int D>
static py::array_t<std::complex<double>>
assemble_Z_bspline_kernel(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_
) {
    static constexpr int NM = D + 1;

    auto j_view = J.unchecked<4>();
    auto ss_view = support_seg.unchecked<2>();
    auto p_view = polys.unchecked<3>();
    auto td_view = td_all.unchecked<2>();

    size_t n_basis = (size_t)support_seg.shape(0);
    if (support_seg.shape(1) != NM) {
        throw std::runtime_error("support_seg.shape(1) must equal D+1");
    }
    if (polys.shape(0) != (long)n_basis || polys.shape(1) != NM ||
        polys.shape(2) != NM) {
        throw std::runtime_error("polys.shape must be (n_basis, D+1, D+1)");
    }
    if (J.shape(0) != NM || J.shape(1) != NM) {
        throw std::runtime_error("J.shape(0:2) must be (D+1, D+1)");
    }

    py::array_t<std::complex<double>> Z({n_basis, n_basis});
    auto z_view = Z.mutable_unchecked<2>();

    // Z = j*omega*mu * Z_A_accum + (1/(j*omega*eps)) * Z_Phi_accum
    // For Z_A_accum = re + j*im:    j*omega*mu * (re + j*im) = -omega*mu*im + j*omega*mu*re
    // For Z_Phi_accum = re + j*im:  (re + j*im)/(j*omega*eps) = im/(omega*eps) - j*re/(omega*eps)
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < n_basis; m++) {
        for (size_t n = 0; n < n_basis; n++) {
            double zA_re = 0.0, zA_im = 0.0;
            double zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t sm = ss_view(m, a);
                for (int b = 0; b < NM; b++) {
                    int64_t sn = ss_view(n, b);
                    double td = td_view(sm, sn);

                    double wA_re = 0.0, wA_im = 0.0;
                    double wPhi_re = 0.0, wPhi_im = 0.0;

                    for (int p = 0; p < NM; p++) {
                        double mp_ap = p_view(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq_bq = p_view(n, b, q);
                            std::complex<double> Jpq = j_view(p, q, sm, sn);
                            double prod = mp_ap * nq_bq;
                            wA_re += prod * Jpq.real();
                            wA_im += prod * Jpq.imag();
                            // p, q in {1..D}: Z_Phi contribution
                            if (p >= 1 && q >= 1) {
                                std::complex<double> Jpm1qm1 = j_view(p - 1, q - 1, sm, sn);
                                double pq = (double)(p * q) * prod;
                                wPhi_re += pq * Jpm1qm1.real();
                                wPhi_im += pq * Jpm1qm1.imag();
                            }
                        }
                    }

                    zA_re += td * wA_re;
                    zA_im += td * wA_im;
                    zPhi_re += wPhi_re;
                    zPhi_im += wPhi_im;
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    return Z;
}

static py::array_t<std::complex<double>>
assemble_Z_bspline(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> J,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> support_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    double omega,
    double eps_,
    double mu_,
    int max_d
) {
    switch (max_d) {
        case 1:
            return assemble_Z_bspline_kernel<1>(J, support_seg, polys, td_all, omega, eps_, mu_);
        case 2:
            return assemble_Z_bspline_kernel<2>(J, support_seg, polys, td_all, omega, eps_, mu_);
        default:
            throw std::runtime_error(
                "assemble_Z_bspline: max_d must be 1 or 2");
    }
}


// Fused off-edge block assembler for the hierarchical (H-matrix / ACA) solver.
//
// Computes a dense Z[I, J] block where every basis pair is OFF-EDGE (the
// caller guarantees admissibility / well-separation), fusing the moment
// quadrature and the Galerkin assembly into one pass — no intermediate
// (D+1, D+1, N, N) moment tensor and no numpy einsum. ACA's row/column
// sampling calls this with a single-row or single-column basis slice, so the
// whole per-row Python orchestration (np.unique / np.vectorize / dict maps /
// einsum) is replaced by one C++ call.
//
// Segment data is passed as the union of segments referenced by the I-side
// and J-side bases (resolved once per block in Python); support_*_local index
// into those union arrays. Same EFIE Galerkin formula as
// assemble_Z_bspline_kernel, but the per-pair moments are quadratured inline
// from the segment endpoints (a²-regularised full kernel, the off-edge path).
template<int D>
static py::array_t<std::complex<double>>
bspline_assemble_offedge_block_kernel(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,   // (nI, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,   // (nI, NM, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,    // (nSegI, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,    // (nSegI, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,     // (nSegI, 3)
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,   // (nJ, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,   // (nJ, NM, NM)
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,    // (nSegJ, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,    // (nSegJ, 3)
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,     // (nSegJ, 3)
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    static constexpr int NM = D + 1;

    auto sI = supp_I.unchecked<2>();
    auto pI = polys_I.unchecked<3>();
    auto slI = segl_I.unchecked<2>();
    auto srI = segr_I.unchecked<2>();
    auto tI = tan_I.unchecked<2>();
    auto sJ = supp_J.unchecked<2>();
    auto pJ = polys_J.unchecked<3>();
    auto slJ = segl_J.unchecked<2>();
    auto srJ = segr_J.unchecked<2>();
    auto tJ = tan_J.unchecked<2>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    size_t nI = (size_t)supp_I.shape(0);
    size_t nJ = (size_t)supp_J.shape(0);
    size_t nSegI = (size_t)segl_I.shape(0);
    size_t nSegJ = (size_t)segl_J.shape(0);
    size_t n_qp = (size_t)gl_t.shape(0);
    if (supp_I.shape(1) != NM || supp_J.shape(1) != NM) {
        throw std::runtime_error("support arrays must have shape (n, D+1)");
    }
    if (n_qp > 8) {
        throw std::runtime_error("n_qp > 8 not supported (scratch buffer size)");
    }

    // Per-segment quadrature positions + lengths, precomputed once.
    std::vector<double> posI(nSegI * n_qp * 3), lenI(nSegI);
    std::vector<double> posJ(nSegJ * n_qp * 3), lenJ(nSegJ);
    for (size_t s = 0; s < nSegI; s++) {
        double dx = srI(s,0)-slI(s,0), dy = srI(s,1)-slI(s,1), dz = srI(s,2)-slI(s,2);
        lenI[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            posI[(s*n_qp+q)*3+0] = (1.0-t)*slI(s,0) + t*srI(s,0);
            posI[(s*n_qp+q)*3+1] = (1.0-t)*slI(s,1) + t*srI(s,1);
            posI[(s*n_qp+q)*3+2] = (1.0-t)*slI(s,2) + t*srI(s,2);
        }
    }
    for (size_t s = 0; s < nSegJ; s++) {
        double dx = srJ(s,0)-slJ(s,0), dy = srJ(s,1)-slJ(s,1), dz = srJ(s,2)-slJ(s,2);
        lenJ[s] = std::sqrt(dx*dx + dy*dy + dz*dz);
        for (size_t q = 0; q < n_qp; q++) {
            double t = glt(q);
            posJ[(s*n_qp+q)*3+0] = (1.0-t)*slJ(s,0) + t*srJ(s,0);
            posJ[(s*n_qp+q)*3+1] = (1.0-t)*slJ(s,1) + t*srJ(s,1);
            posJ[(s*n_qp+q)*3+2] = (1.0-t)*slJ(s,2) + t*srJ(s,2);
        }
    }

    py::array_t<std::complex<double>> Z({nI, nJ});
    auto z_view = Z.mutable_unchecked<2>();

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    PYSIM_OMP_PARALLEL_FOR_COLLAPSE2
    for (size_t m = 0; m < nI; m++) {
        for (size_t n = 0; n < nJ; n++) {
            double zA_re = 0.0, zA_im = 0.0, zPhi_re = 0.0, zPhi_im = 0.0;

            for (int a = 0; a < NM; a++) {
                int64_t smi = sI(m, a);
                double tix = tI(smi,0), tiy = tI(smi,1), tiz = tI(smi,2);
                const double *pi = &posI[smi * n_qp * 3];
                double Li = lenI[smi];
                for (int b = 0; b < NM; b++) {
                    int64_t snj = sJ(n, b);
                    const double *pj = &posJ[snj * n_qp * 3];
                    double Lj = lenJ[snj];
                    double td = tix*tJ(snj,0) + tiy*tJ(snj,1) + tiz*tJ(snj,2);

                    // Moment tensor Jc[p][P] for this single segment pair.
                    std::complex<double> Jc[NM][NM];
                    {
                        alignas(32) double R[64], G_re[64], G_im[64];
                        alignas(32) double wuwu[(NM*NM) * 64];
                        size_t n_pairs = n_qp * n_qp;
                        for (size_t q = 0; q < n_qp; q++) {
                            double pix = pi[q*3+0], piy = pi[q*3+1], piz = pi[q*3+2];
                            for (size_t r = 0; r < n_qp; r++) {
                                double dx = pix - pj[r*3+0];
                                double dy = piy - pj[r*3+1];
                                double dz = piz - pj[r*3+2];
                                R[q*n_qp+r] = std::sqrt(dx*dx+dy*dy+dz*dz+a_squared);
                            }
                        }
                        for (size_t q = 0; q < n_qp; q++) {
                            double wi = glw(q) * Li, ui = glt(q) * Li;
                            double uip[NM]; uip[0] = 1.0;
                            for (int p = 1; p < NM; p++) uip[p] = uip[p-1]*ui;
                            for (size_t r = 0; r < n_qp; r++) {
                                double wj = glw(r) * Lj, uj = glt(r) * Lj;
                                double ujp[NM]; ujp[0] = 1.0;
                                for (int P = 1; P < NM; P++) ujp[P] = ujp[P-1]*uj;
                                double wij = wi*wj;
                                size_t qr = q*n_qp + r;
                                for (int p = 0; p < NM; p++)
                                    for (int P = 0; P < NM; P++)
                                        wuwu[(p*NM+P)*n_pairs + qr] = wij*uip[p]*ujp[P];
                            }
                        }
                        #pragma omp simd
                        for (size_t qr = 0; qr < n_pairs; qr++) {
                            double inv = inv_4pi / R[qr];
                            double ph = -k * R[qr];
                            G_re[qr] = std::cos(ph) * inv;
                            G_im[qr] = std::sin(ph) * inv;
                        }
                        for (int pP = 0; pP < NM*NM; pP++) {
                            double sr_ = 0.0, si_ = 0.0;
                            const double *w_row = &wuwu[pP * n_pairs];
                            #pragma omp simd reduction(+:sr_,si_)
                            for (size_t qr = 0; qr < n_pairs; qr++) {
                                sr_ += w_row[qr]*G_re[qr];
                                si_ += w_row[qr]*G_im[qr];
                            }
                            Jc[pP/NM][pP%NM] = std::complex<double>(sr_, si_);
                        }
                    }

                    // Galerkin combine for this wing pair.
                    double wA_re = 0.0, wA_im = 0.0, wPhi_re = 0.0, wPhi_im = 0.0;
                    for (int p = 0; p < NM; p++) {
                        double mp = pI(m, a, p);
                        for (int q = 0; q < NM; q++) {
                            double nq = pJ(n, b, q);
                            double prod = mp * nq;
                            wA_re += prod * Jc[p][q].real();
                            wA_im += prod * Jc[p][q].imag();
                            if (p >= 1 && q >= 1) {
                                double pq = (double)(p*q) * prod;
                                wPhi_re += pq * Jc[p-1][q-1].real();
                                wPhi_im += pq * Jc[p-1][q-1].imag();
                            }
                        }
                    }
                    zA_re += td * wA_re;
                    zA_im += td * wA_im;
                    zPhi_re += wPhi_re;
                    zPhi_im += wPhi_im;
                }
            }

            double Zre = -omega_mu * zA_im + zPhi_im * inv_omega_eps;
            double Zim = omega_mu * zA_re - zPhi_re * inv_omega_eps;
            z_view(m, n) = std::complex<double>(Zre, Zim);
        }
    }

    return Z;
}

static py::array_t<std::complex<double>>
bspline_assemble_offedge_block(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_I,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_I,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segl_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> segr_J,
    py::array_t<double, py::array::c_style | py::array::forcecast> tan_J,
    double a_squared, double k, double omega, double eps_, double mu_, int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    switch (max_d) {
        case 1:
            return bspline_assemble_offedge_block_kernel<1>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w);
        case 2:
            return bspline_assemble_offedge_block_kernel<2>(
                supp_I, polys_I, segl_I, segr_I, tan_I, supp_J, polys_J,
                segl_J, segr_J, tan_J, a_squared, k, omega, eps_, mu_, gl_t, gl_w);
        default:
            throw std::runtime_error(
                "bspline_assemble_offedge_block: max_d must be 1 or 2");
    }
}


// Runtime dispatch wrapper. Picks the right template instantiation based on
// max_d (the maximum polynomial moment degree, == B-spline degree D).
static py::array_t<std::complex<double>>
seg_seg_full_moments_bspline(
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_i,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r_j,
    double a_squared,
    double k,
    int max_d,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    switch (max_d) {
        case 1:
            return seg_seg_full_moments_bspline_kernel<1>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w);
        case 2:
            return seg_seg_full_moments_bspline_kernel<2>(
                seg_l_i, seg_r_i, seg_l_j, seg_r_j, a_squared, k, gl_t, gl_w);
        default:
            throw std::runtime_error(
                "seg_seg_full_moments_bspline: max_d must be 1 or 2 "
                "(add an explicit template instantiation in _accelerators.cpp)");
    }
}


// Toeplitz fast-path B-spline static-moment evaluation.
//
// For a single straight edge with uniform-h segments, the J_pq^static[i, j]
// integrals are translation-invariant in the arc direction — the matrix is
// Toeplitz with 2N-1 unique values per (p, q) moment. This function computes
// those 2N-1 values via the sympy-derived closed forms (inlined from
// _bspline_static_moments_inline.h) and gathers them to the (max_d+1,
// max_d+1, N, N) output.
//
// Replaces the per-edge numpy loop in `_seg_seg_static_moments` — that path
// took ~5 ms / call mainly from numpy dispatch overhead; the C++ inlined
// closed forms run in ~0.1 ms / call. Big win on multi-edge polylines like
// the hentenna where the static moments dominate after the all-pairs J kernel.
//
// max_d ∈ {0, 1, 2} currently — extends automatically when the header file
// is regenerated for larger MAX_D in scripts/derive_bspline_static_moments.py
// (and the case-list in J_static_dispatch below is extended).
static double J_static_dispatch(int p, int q,
                                double alpha, double beta,
                                double A, double B, double a) {
    int pq = p * 3 + q;
    switch (pq) {
        case 0: return J_static_pq_0_0(alpha, beta, A, B, a);
        case 1: return J_static_pq_0_1(alpha, beta, A, B, a);
        case 2: return J_static_pq_0_2(alpha, beta, A, B, a);
        case 3: return J_static_pq_1_0(alpha, beta, A, B, a);
        case 4: return J_static_pq_1_1(alpha, beta, A, B, a);
        case 5: return J_static_pq_1_2(alpha, beta, A, B, a);
        case 6: return J_static_pq_2_0(alpha, beta, A, B, a);
        case 7: return J_static_pq_2_1(alpha, beta, A, B, a);
        case 8: return J_static_pq_2_2(alpha, beta, A, B, a);
        default:
            throw std::runtime_error("J_static: (p, q) out of inline range");
    }
}

static py::array_t<double>
seg_seg_static_moments_bspline_uniform(double h, double a, size_t N, int max_d) {
    if (max_d < 0 || max_d > 2) {
        throw std::runtime_error("max_d out of range [0, 2]");
    }
    size_t NM = (size_t)(max_d + 1);
    py::array_t<double> out({NM, NM, N, N});
    auto v = out.mutable_unchecked<4>();

    // 2N-1 unique Toeplitz values per moment, indexed by Δ = j - i ∈ [-(N-1), N-1].
    // delta_idx = Δ + (N - 1) ∈ [0, 2N-2].
    size_t n_delta = 2 * N - 1;
    const double inv_4pi = 1.0 / (4.0 * M_PI);

    // Build (NM, NM, n_delta) Toeplitz table
    std::vector<double> table(NM * NM * n_delta);
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            for (size_t di = 0; di < n_delta; di++) {
                long long delta = (long long)di - (long long)(N - 1);
                double alpha = 0.0;
                double beta = h;
                double A_ = (double)delta * h;
                double B_ = ((double)delta + 1.0) * h;
                double val = J_static_dispatch((int)p, (int)q, alpha, beta, A_, B_, a);
                table[(p * NM + q) * n_delta + di] = val * inv_4pi;
            }
        }
    }

    // Gather: v(p, q, i, j) = table[p, q, j - i + (N - 1)]
    for (size_t p = 0; p < NM; p++) {
        for (size_t q = 0; q < NM; q++) {
            const double *row = &table[(p * NM + q) * n_delta];
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    size_t di = (size_t)((long long)j - (long long)i + (long long)(N - 1));
                    v(p, q, i, j) = row[di];
                }
            }
        }
    }
    return out;
}


// Assemble the (Z_pe, Z_ep, Z_ee) blocks for the singular basis enrichment at
// K≥3 junctions (PR #47 productized path).
//
// Each enrichment basis e lives on a single segment adjacent to a junction.
// The shape on that segment is Φ_sing(u) = (u/h)·log(u/h) where u is measured
// from the junction node (u_origin=0 → u = t·h_e, u_origin=1 → u = (1-t)·h_e).
// dΦ_sing/du = (log(u/h) + 1) / h  — log-singular at u=0, matching the K≥3
// junction charge-density singularity.
//
// Integrals (all complex, single k):
//   Z_ee[e, f] = j*ω*μ * td * I_A  +  I_Phi / (j*ω*ε)
//     I_A   = ∫∫ Φ_e(u) Φ_f(u') G du du'
//     I_Phi = ∫∫ Φ_e'(u) Φ_f'(u') G du du'
//   Z_pe[m, e] = same, with polynomial basis m on one side and Φ_e on the other.
//   Z_ep[e, m] = same, but computed independently (no .T shortcut) — the two
//     match to floating-point precision when the same GL rule is used on both
//     axes, but computing them separately verifies that and keeps the path
//     robust if a future quadrature change breaks the symmetry.
//
// Parallelism: outer loop over m (polynomial basis index) for the (Z_pe, Z_ep)
// work, which dominates cost (n_poly ≫ n_enrich). Z_ee is small (n_enrich²);
// computed serially after.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
assemble_Z_enrich(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> spec_seg,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> spec_origin,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_l,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_r,
    py::array_t<double, py::array::c_style | py::array::forcecast> h_per_seg,
    py::array_t<double, py::array::c_style | py::array::forcecast> td_all,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> supp_seg_poly,
    py::array_t<double, py::array::c_style | py::array::forcecast> polys_poly,
    double a_squared,
    double k,
    double omega,
    double eps_,
    double mu_,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t01,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w01,
    py::array_t<double, py::array::c_style | py::array::forcecast> proj_coeffs
) {
    auto specs_v   = spec_seg.unchecked<1>();
    auto origin_v  = spec_origin.unchecked<1>();
    auto sl_v      = seg_l.unchecked<2>();
    auto sr_v      = seg_r.unchecked<2>();
    auto h_v       = h_per_seg.unchecked<1>();
    auto td_v      = td_all.unchecked<2>();
    auto ss_v      = supp_seg_poly.unchecked<2>();
    auto polys_v   = polys_poly.unchecked<3>();
    auto t01_v     = gl_t01.unchecked<1>();
    auto w01_v     = gl_w01.unchecked<1>();
    auto pc_v      = proj_coeffs.unchecked<1>();

    size_t n_enrich = (size_t)spec_seg.shape(0);
    size_t n_poly   = (size_t)supp_seg_poly.shape(0);
    size_t n_wings  = (size_t)supp_seg_poly.shape(1);
    size_t n_qp     = (size_t)gl_t01.shape(0);

    if ((size_t)spec_origin.shape(0) != n_enrich) {
        throw std::runtime_error("spec_origin must match spec_seg length");
    }
    if ((size_t)polys_poly.shape(0) != n_poly ||
        (size_t)polys_poly.shape(1) != n_wings) {
        throw std::runtime_error("polys_poly first two dims must match supp_seg_poly");
    }
    size_t d_plus_1 = (size_t)polys_poly.shape(2);
    if ((size_t)gl_w01.shape(0) != n_qp) {
        throw std::runtime_error("gl_t01 and gl_w01 must have matching length");
    }
    if (sl_v.shape(1) != 3 || sr_v.shape(1) != 3) {
        throw std::runtime_error("seg_l/seg_r must have shape (N_seg, 3)");
    }
    if ((size_t)proj_coeffs.shape(0) != d_plus_1) {
        throw std::runtime_error("proj_coeffs length must equal d+1 (degree + 1)");
    }

    py::array_t<std::complex<double>> Z_pe({n_poly, n_enrich});
    py::array_t<std::complex<double>> Z_ep({n_enrich, n_poly});
    py::array_t<std::complex<double>> Z_ee({n_enrich, n_enrich});
    auto zpe_v = Z_pe.mutable_unchecked<2>();
    auto zep_v = Z_ep.mutable_unchecked<2>();
    auto zee_v = Z_ee.mutable_unchecked<2>();

    if (n_enrich == 0) {
        // Nothing to compute. Return empty arrays.
        return std::make_tuple(Z_pe, Z_ep, Z_ee);
    }

    const double inv_4pi = 1.0 / (4.0 * M_PI);
    const double omega_mu = omega * mu_;
    const double inv_omega_eps = 1.0 / (omega * eps_);

    // -----------------------------------------------------------------
    // Per-enrichment precompute: 3D quad-point positions, Φ_sing values,
    // dΦ_sing/du in arc length, and quadrature weights pre-scaled by h_e.
    // -----------------------------------------------------------------
    std::vector<double> pos_e_all(n_enrich * n_qp * 3);
    std::vector<double> sing_val_all(n_enrich * n_qp);
    std::vector<double> sing_dval_all(n_enrich * n_qp);
    std::vector<double> w_e_all(n_enrich * n_qp);
    std::vector<double> h_e_arr(n_enrich);
    std::vector<int64_t> seg_e_arr(n_enrich);

    const double eps_tiny = 1e-300;
    for (size_t e = 0; e < n_enrich; e++) {
        int64_t se = specs_v(e);
        int orig = (int)origin_v(e);
        double he = h_v(se);
        h_e_arr[e] = he;
        seg_e_arr[e] = se;
        // d(u_norm)/d(u_arc_along_wire): for orig=0 the junction is at the
        // segment's left endpoint, so u_norm = t = u_arc/h and the derivative
        // is +1/h. For orig=1 the junction is at the right endpoint, so
        // u_norm = 1 − t = 1 − u_arc/h and the derivative is −1/h. The
        // singular basis's slope dΦ/du_arc inherits that sign — without
        // it, every "end"-orientation enrichment basis enters the Φ-piece
        // of Z_pe/Z_ep (and the mixed-orig off-diagonals of Z_ee) with the
        // wrong sign, breaking L-R symmetry on geometries like hentenna
        // where mirror junctions have opposite orig.
        double dphi_sign = (orig == 0) ? 1.0 : -1.0;
        for (size_t q = 0; q < n_qp; q++) {
            double t = t01_v(q);
            double w = w01_v(q);
            double u_norm = (orig == 0) ? t : (1.0 - t);
            double u_safe = u_norm > eps_tiny ? u_norm : eps_tiny;
            double log_u = std::log(u_safe);
            // Stable XFEM: Φ_sing_stable(t) = t·log(t) − Σ c_p t^p, so the
            // enrichment basis is L²-orthogonal to the local polynomial
            // space {1, t, …, t^d} on the segment. The projection is in
            // u_norm — the segment's natural orientation-aware coordinate
            // — so it carries through both orientations unchanged.
            // dΦ/du_arc = (dΦ/du_norm) · (du_norm/du_arc) = (...) · sign/h.
            double poly_val = 0.0;
            double poly_dval = 0.0;
            // Horner on Σ c_p t^p and its derivative Σ p·c_p t^(p-1).
            poly_val  = pc_v(d_plus_1 - 1);
            poly_dval = (double)(d_plus_1 - 1) * pc_v(d_plus_1 - 1);
            for (size_t pp = d_plus_1 - 1; pp-- > 0; ) {
                poly_val = poly_val * u_norm + pc_v(pp);
                if (pp >= 1) {
                    poly_dval = poly_dval * u_norm + (double)pp * pc_v(pp);
                }
            }
            sing_val_all[e * n_qp + q] = u_norm * log_u - poly_val;
            sing_dval_all[e * n_qp + q] = dphi_sign * (log_u + 1.0 - poly_dval) / he;
            w_e_all[e * n_qp + q] = w * he;
            double *pe = &pos_e_all[(e * n_qp + q) * 3];
            pe[0] = (1.0 - t) * sl_v(se, 0) + t * sr_v(se, 0);
            pe[1] = (1.0 - t) * sl_v(se, 1) + t * sr_v(se, 1);
            pe[2] = (1.0 - t) * sl_v(se, 2) + t * sr_v(se, 2);
        }
    }

    // -----------------------------------------------------------------
    // Z_ee assembly: pairs (e, f). Symmetric; fill upper triangle then mirror.
    // -----------------------------------------------------------------
    for (size_t e = 0; e < n_enrich; e++) {
        for (size_t f = e; f < n_enrich; f++) {
            double td = td_v(seg_e_arr[e], seg_e_arr[f]);
            double IA_re = 0.0, IA_im = 0.0;
            double IPhi_re = 0.0, IPhi_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double wq    = w_e_all[e * n_qp + q];
                double phiq  = sing_val_all[e * n_qp + q];
                double dphiq = sing_dval_all[e * n_qp + q];
                const double *pq = &pos_e_all[(e * n_qp + q) * 3];
                double wq_phi  = wq * phiq;
                double wq_dphi = wq * dphiq;
                for (size_t r = 0; r < n_qp; r++) {
                    double wr    = w_e_all[f * n_qp + r];
                    double phir  = sing_val_all[f * n_qp + r];
                    double dphir = sing_dval_all[f * n_qp + r];
                    const double *pr = &pos_e_all[(f * n_qp + r) * 3];
                    double dx = pq[0] - pr[0];
                    double dy = pq[1] - pr[1];
                    double dz = pq[2] - pr[2];
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                    double phase = -k * R;
                    double iR_4pi = inv_4pi / R;
                    double Gre = std::cos(phase) * iR_4pi;
                    double Gim = std::sin(phase) * iR_4pi;
                    double wprod_A   = wq_phi  * (wr * phir);
                    double wprod_Phi = wq_dphi * (wr * dphir);
                    IA_re   += wprod_A * Gre;
                    IA_im   += wprod_A * Gim;
                    IPhi_re += wprod_Phi * Gre;
                    IPhi_im += wprod_Phi * Gim;
                }
            }
            // Z = j*ωμ*td*I_A + I_Phi/(jωε)
            // j*ωμ * (re + j im) = -ωμ im + j ωμ re
            // (re + j im) / (j ωε) = im/(ωε) - j re/(ωε)
            double Zre = -omega_mu * td * IA_im + IPhi_im * inv_omega_eps;
            double Zim =  omega_mu * td * IA_re - IPhi_re * inv_omega_eps;
            std::complex<double> Z_val(Zre, Zim);
            zee_v(e, f) = Z_val;
            if (e != f) zee_v(f, e) = Z_val;
        }
    }

    // -----------------------------------------------------------------
    // (Z_pe, Z_ep) assembly. For each polynomial basis m, for each wing of m,
    // for each enrichment e: integrate (poly_m vs Φ_e) and (Φ_e vs poly_m).
    //
    // Z_pe[m, e] integrates with quad-i = polynomial axis, quad-j = singular.
    // Z_ep[e, m] integrates with quad-i = singular axis, quad-j = polynomial.
    // The kernel G is symmetric so the two yield identical sums in exact
    // arithmetic; floating-point rounding is bit-for-bit identical given
    // matching summation order, which we deliberately mirror below.
    //
    // OpenMP parallelizes over m. Z_pe is row-disjoint by m; Z_ep is column-
    // disjoint by m. No reductions needed.
    // -----------------------------------------------------------------
    #pragma omp parallel
    {
        // Per-thread scratch for the polynomial-basis quad-point values.
        std::vector<double> pos_m(n_qp * 3);
        std::vector<double> poly_val(n_qp);
        std::vector<double> poly_dval(n_qp);
        std::vector<double> w_m(n_qp);

        #pragma omp for schedule(static)
        for (size_t m = 0; m < n_poly; m++) {
            for (size_t e = 0; e < n_enrich; e++) {
                zpe_v(m, e) = std::complex<double>(0.0, 0.0);
                zep_v(e, m) = std::complex<double>(0.0, 0.0);
            }
            for (size_t w = 0; w < n_wings; w++) {
                // Skip inactive wings (all-zero polynomial coefficients).
                bool any_nz = false;
                for (size_t p = 0; p < d_plus_1; p++) {
                    if (polys_v(m, w, p) != 0.0) { any_nz = true; break; }
                }
                if (!any_nz) continue;

                int64_t seg_m = ss_v(m, w);
                double hm = h_v(seg_m);

                for (size_t q = 0; q < n_qp; q++) {
                    double t = t01_v(q);
                    double u_arc = t * hm;
                    // Horner over polys_v(m, w, :) — evaluate poly value and derivative.
                    double pv = 0.0, dv = 0.0;
                    // value: P(u) = Σ c_p u^p ; deriv: Σ p·c_p u^(p-1)
                    // Evaluate as: pv = c_D ; for p = D-1..0: pv = pv*u + c_p
                    // and dv with Horner on (p+1)*c_{p+1}: dv = D·c_D ; ...
                    pv = polys_v(m, w, d_plus_1 - 1);
                    dv = (double)(d_plus_1 - 1) * polys_v(m, w, d_plus_1 - 1);
                    for (size_t pp = d_plus_1 - 1; pp-- > 0; ) {
                        pv = pv * u_arc + polys_v(m, w, pp);
                        if (pp >= 1) {
                            dv = dv * u_arc + (double)pp * polys_v(m, w, pp);
                        }
                    }
                    poly_val[q] = pv;
                    poly_dval[q] = dv;
                    w_m[q] = w01_v(q) * hm;
                    pos_m[q*3 + 0] = (1.0 - t) * sl_v(seg_m, 0) + t * sr_v(seg_m, 0);
                    pos_m[q*3 + 1] = (1.0 - t) * sl_v(seg_m, 1) + t * sr_v(seg_m, 1);
                    pos_m[q*3 + 2] = (1.0 - t) * sl_v(seg_m, 2) + t * sr_v(seg_m, 2);
                }

                for (size_t e = 0; e < n_enrich; e++) {
                    int64_t seg_e = seg_e_arr[e];
                    double td_me = td_v(seg_m, seg_e);
                    double td_em = td_v(seg_e, seg_m);

                    // Z_pe[m, e]: i = m-axis, j = e-axis.
                    double pe_IA_re = 0.0, pe_IA_im = 0.0;
                    double pe_IP_re = 0.0, pe_IP_im = 0.0;
                    // Z_ep[e, m]: i = e-axis, j = m-axis.
                    double ep_IA_re = 0.0, ep_IA_im = 0.0;
                    double ep_IP_re = 0.0, ep_IP_im = 0.0;

                    for (size_t q = 0; q < n_qp; q++) {
                        double wmq      = w_m[q];
                        double pvq      = poly_val[q];
                        double dvq      = poly_dval[q];
                        const double *pmq = &pos_m[q*3];

                        double weq_eax  = w_e_all[e * n_qp + q];
                        double phiq_eax = sing_val_all[e * n_qp + q];
                        double dphiq_eax= sing_dval_all[e * n_qp + q];
                        const double *peq_eax = &pos_e_all[(e * n_qp + q) * 3];

                        double wmq_pv  = wmq * pvq;
                        double wmq_dv  = wmq * dvq;
                        double weq_phi = weq_eax * phiq_eax;
                        double weq_dphi= weq_eax * dphiq_eax;

                        for (size_t r = 0; r < n_qp; r++) {
                            // -- Z_pe leg: i on m, j on e --
                            {
                                double wer      = w_e_all[e * n_qp + r];
                                double phir     = sing_val_all[e * n_qp + r];
                                double dphir    = sing_dval_all[e * n_qp + r];
                                const double *per = &pos_e_all[(e * n_qp + r) * 3];
                                double dx = pmq[0] - per[0];
                                double dy = pmq[1] - per[1];
                                double dz = pmq[2] - per[2];
                                double R = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                                double phase = -k * R;
                                double iR_4pi = inv_4pi / R;
                                double Gre = std::cos(phase) * iR_4pi;
                                double Gim = std::sin(phase) * iR_4pi;
                                double wprod_A   = wmq_pv * (wer * phir);
                                double wprod_Phi = wmq_dv * (wer * dphir);
                                pe_IA_re += wprod_A   * Gre;
                                pe_IA_im += wprod_A   * Gim;
                                pe_IP_re += wprod_Phi * Gre;
                                pe_IP_im += wprod_Phi * Gim;
                            }
                            // -- Z_ep leg: i on e, j on m --
                            {
                                double wmr      = w_m[r];
                                double pvr      = poly_val[r];
                                double dvr      = poly_dval[r];
                                const double *pmr = &pos_m[r*3];
                                double dx = peq_eax[0] - pmr[0];
                                double dy = peq_eax[1] - pmr[1];
                                double dz = peq_eax[2] - pmr[2];
                                double R = std::sqrt(dx*dx + dy*dy + dz*dz + a_squared);
                                double phase = -k * R;
                                double iR_4pi = inv_4pi / R;
                                double Gre = std::cos(phase) * iR_4pi;
                                double Gim = std::sin(phase) * iR_4pi;
                                double wprod_A   = weq_phi  * (wmr * pvr);
                                double wprod_Phi = weq_dphi * (wmr * dvr);
                                ep_IA_re += wprod_A   * Gre;
                                ep_IA_im += wprod_A   * Gim;
                                ep_IP_re += wprod_Phi * Gre;
                                ep_IP_im += wprod_Phi * Gim;
                            }
                        }
                    }
                    double Zpe_re = -omega_mu * td_me * pe_IA_im + pe_IP_im * inv_omega_eps;
                    double Zpe_im =  omega_mu * td_me * pe_IA_re - pe_IP_re * inv_omega_eps;
                    double Zep_re = -omega_mu * td_em * ep_IA_im + ep_IP_im * inv_omega_eps;
                    double Zep_im =  omega_mu * td_em * ep_IA_re - ep_IP_re * inv_omega_eps;
                    zpe_v(m, e) += std::complex<double>(Zpe_re, Zpe_im);
                    zep_v(e, m) += std::complex<double>(Zep_re, Zep_im);
                }
            }
        }
    }  // end omp parallel

    return std::make_tuple(Z_pe, Z_ep, Z_ee);
}


// Sinusoidal-basis (NEC2 three-term) tangential-field tensor.
//
// For each (m=obs, n=src) pair of segments, compute the three scalar tensors
//   Phi_const[m, n] = ŝ_m · E^const_n(c_m)
//   Phi_sin  [m, n] = ŝ_m · E^sin_n  (c_m)
//   Phi_cos  [m, n] = ŝ_m · E^cos_n  (c_m)
// where the source's local frame is centered on segment n with z-axis along
// src_tangents[n]; the const/sin/cos sources are I(z')=1 / sin(k z') /
// cos(k z') over z' ∈ [-H_n, +H_n], H_n = h_n/2. Result is in NEC's natural-
// arc convention (σ accounting is the caller's job).
//
// Closed forms for the const-source `int G_0 dz'` are 1/r_0 singularity
// extraction: ∫ 1/r_0 dz' = arcsinh((H-z)/ρ) - arcsinh((-H-z)/ρ); regular
// remainder via Gauss-Legendre on the (G_0 - 1/r_0) integrand. Sin/cos
// sources are fully closed-form per Eqs 76-79 of the LLNL theory manual
// (mirrored by the numpy reference in src/pysim/sinusoidal.py _field_tensor).
//
// Parallelism: each (m, n) pair is independent. OpenMP collapse(2) over the
// (m, n) grid; per-n constants (H_n, sin(kH_n), cos(kH_n)) are precomputed
// outside the parallel region.
static std::tuple<py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>,
                  py::array_t<std::complex<double>>>
sinusoidal_field_tensor(
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> obs_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_centers,
    py::array_t<double, py::array::c_style | py::array::forcecast> src_tangents,
    py::array_t<double, py::array::c_style | py::array::forcecast> seg_h,
    double a, double k, double eta,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_t,
    py::array_t<double, py::array::c_style | py::array::forcecast> gl_w
) {
    auto oc = obs_centers.unchecked<2>();
    auto ot = obs_tangents.unchecked<2>();
    auto sc = src_centers.unchecked<2>();
    auto st = src_tangents.unchecked<2>();
    auto sh = seg_h.unchecked<1>();
    auto glt = gl_t.unchecked<1>();
    auto glw = gl_w.unchecked<1>();

    if (oc.shape(1) != 3 || ot.shape(1) != 3 ||
        sc.shape(1) != 3 || st.shape(1) != 3) {
        throw std::runtime_error("center/tangent arrays must have shape (N, 3)");
    }
    if (oc.shape(0) != ot.shape(0)) {
        throw std::runtime_error("obs_centers and obs_tangents must have matching N");
    }
    if (sc.shape(0) != st.shape(0) || sc.shape(0) != sh.shape(0)) {
        throw std::runtime_error("src arrays must all have matching N");
    }
    if (glt.shape(0) != glw.shape(0)) {
        throw std::runtime_error("gl_t and gl_w must have matching length");
    }

    size_t M = oc.shape(0);
    size_t N = sc.shape(0);
    size_t n_qp = glt.shape(0);

    py::array_t<std::complex<double>> Phi_const({M, N});
    py::array_t<std::complex<double>> Phi_sin({M, N});
    py::array_t<std::complex<double>> Phi_cos({M, N});
    auto pc = Phi_const.mutable_unchecked<2>();
    auto ps = Phi_sin.mutable_unchecked<2>();
    auto pco = Phi_cos.mutable_unchecked<2>();

    // Per-source-segment precompute: H_n = h_n/2, sin(kH_n), cos(kH_n).
    std::vector<double> H_n(N), sin_kH(N), cos_kH(N);
    for (size_t n = 0; n < N; n++) {
        H_n[n] = 0.5 * sh(n);
        sin_kH[n] = std::sin(k * H_n[n]);
        cos_kH[n] = std::cos(k * H_n[n]);
    }

    // Cache GL nodes/weights in std::vector so the per-pair loops avoid
    // bouncing through the pybind11 unchecked accessor in the inner loop.
    std::vector<double> glt_v(n_qp), glw_v(n_qp);
    for (size_t q = 0; q < n_qp; q++) {
        glt_v[q] = glt(q);
        glw_v[q] = glw(q);
    }

    // Scalar prefactors. pref_z = +j eta / (4 pi k); pref_rho_const has the
    // same form (the per-pair 1/rho_eval scaling lives inside pref_rho).
    const double four_pi_k = 4.0 * M_PI * k;
    const double pref_z_im = eta / four_pi_k;
    const double pref_rho_const_im = -eta / four_pi_k;
    const double a_sq = a * a;

    // The oscillatory sincos is batched across all N source segments (per
    // observer m) into one flat buffer so it vectorizes — the per-pair scalar
    // calls were unvectorizable. Three stages per m: (A) geometry + phases,
    // (B) one omp-simd sincos sweep over the buffer, (C) assembly.
    //
    // Phases per source segment: 2 boundary (r0_2, r0_1) + n_qp quadrature nodes.
    const size_t S = n_qp + 2;
    const size_t P = N * S;

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        // Per-iteration scratch (per-thread under the parallel-for). Sizes are
        // tiny (P ~ N*(n_qp+2)); allocation cost is negligible next to the
        // sincos + assembly work it feeds.
        std::vector<double> ph(P), cphb(P), sphb(P);
        std::vector<double> rho_eval_a(N), dz1_a(N), dz2_a(N),
                            r0_1_a(N), r0_2_a(N), td_a(N), rpf_a(N);
        std::vector<double> r0q_inv_a(N * n_qp);

        double cmx = oc(m, 0), cmy = oc(m, 1), cmz = oc(m, 2);
        double tmx = ot(m, 0), tmy = ot(m, 1), tmz = ot(m, 2);

        // ---- Stage A: geometry + phase generation -------------------------
        for (size_t n = 0; n < N; n++) {
            double cnx = sc(n, 0), cny = sc(n, 1), cnz = sc(n, 2);
            double tnx = st(n, 0), tny = st(n, 1), tnz = st(n, 2);
            double rvx = cmx - cnx, rvy = cmy - cny, rvz = cmz - cnz;
            double z_eval = rvx * tnx + rvy * tny + rvz * tnz;
            double rho_vx = rvx - z_eval * tnx;
            double rho_vy = rvy - z_eval * tny;
            double rho_vz = rvz - z_eval * tnz;
            double rho_axis = std::sqrt(rho_vx*rho_vx + rho_vy*rho_vy + rho_vz*rho_vz);
            double rho_eval = std::sqrt(rho_axis*rho_axis + a_sq);
            double td = tmx*tnx + tmy*tny + tmz*tnz;
            double rho_dot_tobs = rho_vx*tmx + rho_vy*tmy + rho_vz*tmz;

            double H = H_n[n];
            double dz2 = z_eval - H;
            double dz1 = z_eval + H;
            double r0_2 = std::sqrt(rho_eval*rho_eval + dz2*dz2);
            double r0_1 = std::sqrt(rho_eval*rho_eval + dz1*dz1);

            rho_eval_a[n] = rho_eval;
            dz1_a[n] = dz1; dz2_a[n] = dz2;
            r0_1_a[n] = r0_1; r0_2_a[n] = r0_2;
            td_a[n] = td; rpf_a[n] = rho_dot_tobs / rho_eval;

            size_t base = n * S;
            ph[base + 0] = -k * r0_2;
            ph[base + 1] = -k * r0_1;
            for (size_t q = 0; q < n_qp; q++) {
                double z_q = H * glt_v[q];
                double dz_q = z_eval - z_q;
                double r0_q = std::sqrt(rho_eval*rho_eval + dz_q*dz_q);
                ph[base + 2 + q] = -k * r0_q;
                r0q_inv_a[n * n_qp + q] = 1.0 / r0_q;
            }
        }

        // ---- Stage B: vectorized sincos over every phase ------------------
        // Split cos and sin into separate omp-simd loops (libmvec has no vector
        // sincos) so each body stays vectorizable to _ZGVdN4v_{cos,sin}
        // (AVX2, 4 doubles per call).
        #pragma omp simd
        for (size_t i = 0; i < P; i++) cphb[i] = std::cos(ph[i]);
        #pragma omp simd
        for (size_t i = 0; i < P; i++) sphb[i] = std::sin(ph[i]);

        // ---- Stage C: assembly --------------------------------------------
        for (size_t n = 0; n < N; n++) {
            size_t base = n * S;
            double cph_2 = cphb[base + 0], sph_2 = sphb[base + 0];
            double cph_1 = cphb[base + 1], sph_1 = sphb[base + 1];
            double rho_eval = rho_eval_a[n];
            double dz1 = dz1_a[n], dz2 = dz2_a[n];
            double r0_1 = r0_1_a[n], r0_2 = r0_2_a[n];
            double td = td_a[n], rho_proj_factor = rpf_a[n];
            double H = H_n[n];
            double inv_r0_2 = 1.0 / r0_2;
            double inv_r0_1 = 1.0 / r0_1;
            double G0_2_re = cph_2 * inv_r0_2, G0_2_im = sph_2 * inv_r0_2;
            double G0_1_re = cph_1 * inv_r0_1, G0_1_im = sph_1 * inv_r0_1;

            // (1 + j k r0) / r0² split into re/im
            double inv_r0_2_sq = inv_r0_2 * inv_r0_2;
            double inv_r0_1_sq = inv_r0_1 * inv_r0_1;
            double one_jkr_2_re = inv_r0_2_sq;
            double one_jkr_2_im = k * r0_2 * inv_r0_2_sq;
            double one_jkr_1_re = inv_r0_1_sq;
            double one_jkr_1_im = k * r0_1 * inv_r0_1_sq;

            // ---- Const source -------------------------------------------------
            // Erho_const = pref_rho_const * (
            //     (1 + j k r0_2) * rho_eval * G0_2 / r0_2²
            //   - (1 + j k r0_1) * rho_eval * G0_1 / r0_1²
            // )
            // Let A_2 = (1 + jk r0_2) / r0_2² = one_jkr_2 (complex).
            // term_2 = A_2 * G0_2 — complex product.
            auto cmul = [](double ar, double ai, double br, double bi,
                           double &cr, double &ci) {
                cr = ar*br - ai*bi;
                ci = ar*bi + ai*br;
            };

            double term_const2_re, term_const2_im;
            cmul(one_jkr_2_re, one_jkr_2_im, G0_2_re, G0_2_im,
                 term_const2_re, term_const2_im);
            double term_const1_re, term_const1_im;
            cmul(one_jkr_1_re, one_jkr_1_im, G0_1_re, G0_1_im,
                 term_const1_re, term_const1_im);
            double rho_diff_re = rho_eval * (term_const2_re - term_const1_re);
            double rho_diff_im = rho_eval * (term_const2_im - term_const1_im);
            // pref_rho_const = j * pref_rho_const_im  (pure imaginary scalar)
            double Erho_const_re = -pref_rho_const_im * rho_diff_im;
            double Erho_const_im =  pref_rho_const_im * rho_diff_re;

            // u2, u1, int_inv_r0.  H - z_eval = -dz2;  -H - z_eval = -dz1.
            double inv_rho_eval = 1.0 / rho_eval;
            double u2 = -dz2 * inv_rho_eval;
            double u1 = -dz1 * inv_rho_eval;
            double int_inv_r0 = std::asinh(u2) - std::asinh(u1);

            // Quadrature for the smooth remainder of int_G0:
            //   reg(q) = (exp(-jk r0_q) - 1) / r0_q,   r0_q = sqrt(ρ² + (z - H gx[q])²)
            //   int_reg = H * Σ_q reg(q) * gw[q]
            double int_reg_re = 0.0, int_reg_im = 0.0;
            for (size_t q = 0; q < n_qp; q++) {
                double cph_q = cphb[base + 2 + q];
                double sph_q = sphb[base + 2 + q];
                double inv_r0_q = r0q_inv_a[n * n_qp + q];
                // (exp(jphase) - 1) / r0
                double reg_re = (cph_q - 1.0) * inv_r0_q;
                double reg_im = sph_q * inv_r0_q;
                int_reg_re += reg_re * glw_v[q];
                int_reg_im += reg_im * glw_v[q];
            }
            int_reg_re *= H;
            int_reg_im *= H;
            double int_G0_re = int_inv_r0 + int_reg_re;
            double int_G0_im = int_reg_im;

            // Ez_const_boundary = (1+jk r0_2) dz2 G0_2 / r0_2² - (1+jk r0_1) dz1 G0_1 / r0_1²
            double Ez_boundary_re = dz2 * term_const2_re - dz1 * term_const1_re;
            double Ez_boundary_im = dz2 * term_const2_im - dz1 * term_const1_im;
            double k_sq = k * k;
            // Ez_const = -pref_z * (Ez_boundary + k² int_G0). pref_z = j * pref_z_im.
            double inside_re = Ez_boundary_re + k_sq * int_G0_re;
            double inside_im = Ez_boundary_im + k_sq * int_G0_im;
            // Multiply by -j * pref_z_im
            double Ez_const_re =  pref_z_im * inside_im;
            double Ez_const_im = -pref_z_im * inside_re;

            // ---- Sine source (Eq 76, 77) --------------------------------------
            double sin2 = sin_kH[n];
            double cos2 = cos_kH[n];
            double sin1 = -sin2;
            double cos1 =  cos2;

            // bracket_sin_2 = G0_2 * (k dz2 cos2 + (1 - dz2² (1+jk r0_2)/r0_2²) sin2)
            double inner_2_re = 1.0 - dz2*dz2 * one_jkr_2_re;
            double inner_2_im =     - dz2*dz2 * one_jkr_2_im;
            double bracket_sin_2_re = k*dz2*cos2 + inner_2_re*sin2;
            double bracket_sin_2_im =              inner_2_im*sin2;
            double bsin2_re, bsin2_im;
            cmul(G0_2_re, G0_2_im, bracket_sin_2_re, bracket_sin_2_im,
                 bsin2_re, bsin2_im);

            double inner_1_re = 1.0 - dz1*dz1 * one_jkr_1_re;
            double inner_1_im =     - dz1*dz1 * one_jkr_1_im;
            double bracket_sin_1_re = k*dz1*cos1 + inner_1_re*sin1;
            double bracket_sin_1_im =              inner_1_im*sin1;
            double bsin1_re, bsin1_im;
            cmul(G0_1_re, G0_1_im, bracket_sin_1_re, bracket_sin_1_im,
                 bsin1_re, bsin1_im);
            double Erho_sin_inner_re = bsin2_re - bsin1_re;
            double Erho_sin_inner_im = bsin2_im - bsin1_im;
            // pref_rho = -j eta / (4 pi k rho_eval)
            double pref_rho_im = pref_rho_const_im * inv_rho_eval;
            double Erho_sin_re = -pref_rho_im * Erho_sin_inner_im;
            double Erho_sin_im =  pref_rho_im * Erho_sin_inner_re;

            // bracket_sin_z = G0 * (k cos - (1+jk r0) dz / r0² sin)
            double bracket_sin_z_2_re = k*cos2 - dz2*one_jkr_2_re*sin2;
            double bracket_sin_z_2_im =        - dz2*one_jkr_2_im*sin2;
            double bszin2_re, bszin2_im;
            cmul(G0_2_re, G0_2_im, bracket_sin_z_2_re, bracket_sin_z_2_im,
                 bszin2_re, bszin2_im);
            double bracket_sin_z_1_re = k*cos1 - dz1*one_jkr_1_re*sin1;
            double bracket_sin_z_1_im =        - dz1*one_jkr_1_im*sin1;
            double bszin1_re, bszin1_im;
            cmul(G0_1_re, G0_1_im, bracket_sin_z_1_re, bracket_sin_z_1_im,
                 bszin1_re, bszin1_im);
            double Ez_sin_inner_re = bszin2_re - bszin1_re;
            double Ez_sin_inner_im = bszin2_im - bszin1_im;
            // pref_z = +j pref_z_im
            double Ez_sin_re = -pref_z_im * Ez_sin_inner_im;
            double Ez_sin_im =  pref_z_im * Ez_sin_inner_re;

            // ---- Cosine source ------------------------------------------------
            // bracket_cos_2 = G0_2 * (-k dz2 sin2 + (1 - dz2² (1+jk r0_2)/r0_2²) cos2)
            double bracket_cos_2_re = -k*dz2*sin2 + inner_2_re*cos2;
            double bracket_cos_2_im =              inner_2_im*cos2;
            double bcos2_re, bcos2_im;
            cmul(G0_2_re, G0_2_im, bracket_cos_2_re, bracket_cos_2_im,
                 bcos2_re, bcos2_im);
            double bracket_cos_1_re = -k*dz1*sin1 + inner_1_re*cos1;
            double bracket_cos_1_im =              inner_1_im*cos1;
            double bcos1_re, bcos1_im;
            cmul(G0_1_re, G0_1_im, bracket_cos_1_re, bracket_cos_1_im,
                 bcos1_re, bcos1_im);
            double Erho_cos_inner_re = bcos2_re - bcos1_re;
            double Erho_cos_inner_im = bcos2_im - bcos1_im;
            double Erho_cos_re = -pref_rho_im * Erho_cos_inner_im;
            double Erho_cos_im =  pref_rho_im * Erho_cos_inner_re;

            double bracket_cos_z_2_re = -k*sin2 - dz2*one_jkr_2_re*cos2;
            double bracket_cos_z_2_im =         - dz2*one_jkr_2_im*cos2;
            double bczin2_re, bczin2_im;
            cmul(G0_2_re, G0_2_im, bracket_cos_z_2_re, bracket_cos_z_2_im,
                 bczin2_re, bczin2_im);
            double bracket_cos_z_1_re = -k*sin1 - dz1*one_jkr_1_re*cos1;
            double bracket_cos_z_1_im =         - dz1*one_jkr_1_im*cos1;
            double bczin1_re, bczin1_im;
            cmul(G0_1_re, G0_1_im, bracket_cos_z_1_re, bracket_cos_z_1_im,
                 bczin1_re, bczin1_im);
            double Ez_cos_inner_re = bczin2_re - bczin1_re;
            double Ez_cos_inner_im = bczin2_im - bczin1_im;
            double Ez_cos_re = -pref_z_im * Ez_cos_inner_im;
            double Ez_cos_im =  pref_z_im * Ez_cos_inner_re;

            // Project to obs tangent: Phi = td * Ez + rho_proj * Erho.
            pc(m, n) = std::complex<double>(
                td * Ez_const_re + rho_proj_factor * Erho_const_re,
                td * Ez_const_im + rho_proj_factor * Erho_const_im);
            ps(m, n) = std::complex<double>(
                td * Ez_sin_re + rho_proj_factor * Erho_sin_re,
                td * Ez_sin_im + rho_proj_factor * Erho_sin_im);
            pco(m, n) = std::complex<double>(
                td * Ez_cos_re + rho_proj_factor * Erho_cos_re,
                td * Ez_cos_im + rho_proj_factor * Erho_cos_im);
        }
    }

    return std::make_tuple(Phi_const, Phi_sin, Phi_cos);
}


PYBIND11_MODULE(_accelerators, m) {
    m.def("seg_seg_quad_batch_3d", &seg_seg_quad_batch_3d,
          "Batched 3D cross-segment Gauss-Legendre quadrature over a k vector. "
          "Returns (J00, J10, J01, J11) each (n_k, N_i, N_j) complex.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k_array"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_reg_quad_batch_1d", &seg_seg_reg_quad_batch_1d,
          "Batched same-wire Gauss-Legendre quadrature on the regularized "
          "kernel (exp(-jkR)-1)/(4 pi R) over a k vector. "
          "Returns (J00, J10, J01, J11) each (n_k, N, N) complex.",
          py::arg("seg_endpoints"), py::arg("a"),
          py::arg("k_array"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("assemble_Z", &assemble_Z,
          "Assemble the (n_k, n_basis, n_basis) Z matrix from the four J "
          "tensors, per-segment h, tangent-dot table, left/right basis-to-"
          "segment mappings, and omega(k). Returns Z complex.",
          py::arg("J00"), py::arg("J10"),
          py::arg("J01"), py::arg("J11"),
          py::arg("h_per_seg"), py::arg("td_all"),
          py::arg("left_seg"), py::arg("right_seg"),
          py::arg("omega_array"),
          py::arg("eps"), py::arg("mu"));
    m.def("assemble_Z_general", &assemble_Z_general,
          "Assemble Z from per-basis (support_seg, support_L, support_R) "
          "(n_basis, 2) arrays. Handles arbitrary 2-wing basis layouts "
          "including junction directional bases (one wing inactive with "
          "L=R=0). Returns Z complex (n_k, n_basis, n_basis).",
          py::arg("J00"), py::arg("J10"),
          py::arg("J01"), py::arg("J11"),
          py::arg("h_per_seg"), py::arg("td_all"),
          py::arg("support_seg"),
          py::arg("support_L"), py::arg("support_R"),
          py::arg("omega_array"),
          py::arg("eps"), py::arg("mu"));
    m.def("seg_seg_full_moments_bspline", &seg_seg_full_moments_bspline,
          "Single-k full-kernel polynomial moment integrals for the B-spline "
          "Galerkin MoM. Returns J of shape (max_d+1, max_d+1, N_i, N_j) "
          "complex. Templated on max_d at compile time; currently "
          "instantiated for max_d in {1, 2}.",
          py::arg("seg_l_i"), py::arg("seg_r_i"),
          py::arg("seg_l_j"), py::arg("seg_r_j"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("seg_seg_static_moments_bspline_uniform",
          &seg_seg_static_moments_bspline_uniform,
          "Closed-form same-edge static-kernel polynomial moments J_pq for a "
          "uniform-h edge with N segments. Uses Toeplitz structure (2N-1 "
          "unique values per moment) and inlined sympy-derived closed forms. "
          "Returns J_static of shape (max_d+1, max_d+1, N, N), with the "
          "1/(4π) prefactor folded in.",
          py::arg("h"), py::arg("a"), py::arg("N"), py::arg("max_d"));
    m.def("assemble_Z_bspline", &assemble_Z_bspline,
          "Assemble the (n_basis, n_basis) Z matrix from the polynomial-"
          "moment tensor J, per-basis polynomial coefficients, support-segment "
          "map, and tangent-dot table. Templated on max_d at compile time; "
          "currently instantiated for max_d in {1, 2}. Single-k.",
          py::arg("J"), py::arg("support_seg"),
          py::arg("polys"), py::arg("td_all"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("max_d"));
    m.def("bspline_assemble_offedge_block", &bspline_assemble_offedge_block,
          "Fused off-edge Z[I, J] block assembly for the H-matrix / ACA "
          "solver: quadratures the a²-regularised full-kernel moments and "
          "performs the EFIE Galerkin combine in one pass, with no "
          "intermediate moment tensor. Segments are the per-block union "
          "referenced by the I/J bases; support_*_local index into them. "
          "Templated on max_d in {1, 2}; single-k.",
          py::arg("supp_I"), py::arg("polys_I"), py::arg("segl_I"),
          py::arg("segr_I"), py::arg("tan_I"),
          py::arg("supp_J"), py::arg("polys_J"), py::arg("segl_J"),
          py::arg("segr_J"), py::arg("tan_J"),
          py::arg("a_squared"), py::arg("k"), py::arg("omega"),
          py::arg("eps"), py::arg("mu"), py::arg("max_d"),
          py::arg("gl_t"), py::arg("gl_w"));
    m.def("assemble_Z_enrich", &assemble_Z_enrich,
          "Assemble (Z_pe, Z_ep, Z_ee) for the stable XFEM singular basis "
          "enrichment at K≥3 junctions. Each enrichment basis is "
          "Φ_sing_stable(t) = t·log(t) − Σ_p proj_coeffs[p]·t^p with "
          "t = u_norm = u/h (origin=0) or 1 − u/h (origin=1), so the "
          "enrichment is L²-orthogonal to the local polynomial space on "
          "each segment. proj_coeffs must have length degree+1 and match "
          "the polys_poly third dim. Z_ep is computed independently from "
          "Z_pe (no .T shortcut). Single-k.",
          py::arg("spec_seg"), py::arg("spec_origin"),
          py::arg("seg_l"), py::arg("seg_r"),
          py::arg("h_per_seg"), py::arg("td_all"),
          py::arg("supp_seg_poly"), py::arg("polys_poly"),
          py::arg("a_squared"), py::arg("k"),
          py::arg("omega"), py::arg("eps"), py::arg("mu"),
          py::arg("gl_t01"), py::arg("gl_w01"),
          py::arg("proj_coeffs"));
    m.def("sinusoidal_field_tensor", &sinusoidal_field_tensor,
          "Tangential field tensor for the NEC2 three-term basis. Returns "
          "(Phi_const, Phi_sin, Phi_cos), each (M, N) complex. obs_*/src_* "
          "can be the same arrays (free-space build) or src_* mirrored "
          "(PEC image build).",
          py::arg("obs_centers"), py::arg("obs_tangents"),
          py::arg("src_centers"), py::arg("src_tangents"),
          py::arg("seg_h"),
          py::arg("a"), py::arg("k"), py::arg("eta"),
          py::arg("gl_t"), py::arg("gl_w"));
}
