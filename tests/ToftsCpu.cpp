#define BOOST_TEST_MODULE Gpufit

#include "Cpufit/cpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <cmath>
#include <vector>

BOOST_AUTO_TEST_CASE(ToftsCpu)
{
    std::size_t const n_fits = 1;
    std::size_t const n_points = 60;

    std::vector<REAL> time_x(n_points);
    std::vector<REAL> cp(n_points);
    REAL const dt = (15.f - 0.25f) / REAL(n_points - 1);

    for (std::size_t i = 0; i < n_points; i++)
    {
        time_x[i] = 0.25f + REAL(i) * dt;
        cp[i] = time_x[i] >= 1.f ? 5.5f * std::exp(-0.6f * time_x[i]) : 0.f;
    }

    REAL const ktrans_true = 0.18f;
    REAL const ve_true = 0.45f;

    std::vector<REAL> data(n_points, 0.f);
    for (std::size_t point_index = 0; point_index < n_points; point_index++)
    {
        REAL convolution = 0.f;
        for (std::size_t i = 1; i <= point_index; i++)
        {
            REAL const spacing = time_x[i] - time_x[i - 1];
            REAL const ct = cp[i] * std::exp(-ktrans_true * (time_x[point_index] - time_x[i]) / ve_true);
            REAL const ct_prev = cp[i - 1] * std::exp(-ktrans_true * (time_x[point_index] - time_x[i - 1]) / ve_true);
            convolution += (ct + ct_prev) * spacing / 2.f;
        }
        data[point_index] = ktrans_true * convolution;
    }

    std::vector<REAL> initial_parameters{ 0.12f, 0.30f };
    std::vector<int> parameters_to_fit{ 1, 1 };

    std::vector<REAL> user_info;
    user_info.reserve(2 * n_points);
    user_info.insert(user_info.end(), time_x.begin(), time_x.end());
    user_info.insert(user_info.end(), cp.begin(), cp.end());

    std::vector<REAL> output_parameters(2, 0.f);
    int output_state = -1;
    REAL output_chi_square = -1.f;
    int output_iterations = -1;

    int const status = cpufit(
        n_fits,
        n_points,
        data.data(),
        nullptr,
        TOFTS,
        initial_parameters.data(),
        1e-7f,
        200,
        parameters_to_fit.data(),
        LSE,
        user_info.size() * sizeof(REAL),
        reinterpret_cast<char*>(user_info.data()),
        output_parameters.data(),
        &output_state,
        &output_chi_square,
        &output_iterations);

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_state == 0);
    BOOST_CHECK(std::abs(output_parameters[0] - ktrans_true) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[1] - ve_true) < 1e-6f);
}
