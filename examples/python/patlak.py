"""
Patlak fitting example using pygpufit.

If CUDA is unavailable and pycpufit is installed, pygpufit automatically
falls back to Cpufit.
"""

import numpy as np
import pygpufit.gpufit as gf


def generate_patlak_data(time_x, cp, parameters):
    n_points = time_x.shape[0]
    y = np.zeros(n_points, dtype=np.float32)
    for point_index in range(n_points):
        conv_cp = np.float32(0.0)
        for i in range(1, point_index):
            spacing = time_x[i] - time_x[i - 1]
            conv_cp += (cp[i - 1] + cp[i]) * np.float32(0.5) * spacing
        y[point_index] = parameters[0] * conv_cp + parameters[1] * cp[point_index]
    return y


if __name__ == '__main__':

    print('CUDA available: {}'.format(gf.cuda_available()))
    if gf.cuda_available():
        print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))
    else:
        print('Running via Cpufit fallback if pycpufit is installed.')

    # model configuration
    number_fits = 200
    number_points = 60
    number_parameters = 2
    model_id = gf.ModelID.PATLAK

    # time points and plasma concentration curve
    time_x = np.array([
        0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
        3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0,
        6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0,
        9.25, 9.5, 9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75,
        12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5,
        14.75, 15.0
    ], dtype=np.float32)

    cp = np.array([
        0.0, 0.0, 0.0, 3.0184639985, 2.5980160401, 2.2361331286, 1.9246576201,
        1.6565681655, 1.4258214336, 1.2272158808, 1.0562744974, 0.9091438852,
        0.7825073937, 0.6735103554, 0.5796957351, 0.4989487431, 0.4294491630,
        0.3696303201, 0.3181437648, 0.2738288760, 0.2356866978, 0.2028574207,
        0.1746010001, 0.1502804735, 0.1293476022, 0.1113305130, 0.0958230605,
        0.0824756725, 0.0709874692, 0.0610994810, 0.0525888106, 0.0452636088,
        0.0389587491, 0.0335321061, 0.0288613512, 0.0248411952, 0.0213810148,
        0.0184028100, 0.0158394454, 0.0136331370, 0.0117341497, 0.0100996763,
        0.0086928719, 0.0074820242, 0.0064398379, 0.0055428199, 0.0047707493,
        0.0041062219, 0.0035342580, 0.0030419640, 0.0026182427, 0.0022535424,
        0.0019396419, 0.0016694653, 0.0014369221, 0.0012367703, 0.0010644980,
        0.0009162220, 0.0007885995, 0.0006787539
    ], dtype=np.float32)

    user_info = np.concatenate([time_x, cp]).astype(np.float32)

    true_parameters = np.array([0.05, 0.03], dtype=np.float32)

    # synthetic data for all fits
    one_fit = generate_patlak_data(time_x, cp, true_parameters)
    data = np.tile(one_fit, (number_fits, 1)).astype(np.float32)

    # initial guesses (randomized around truth)
    rng = np.random.default_rng(7)
    initial_parameters = np.tile(true_parameters, (number_fits, 1)).astype(np.float32)
    initial_parameters *= (0.8 + 0.4 * rng.random((number_fits, number_parameters))).astype(np.float32)

    parameters_to_fit = np.ones(number_parameters, dtype=np.int32)

    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(
        data,
        None,
        model_id,
        initial_parameters,
        tolerance=1e-8,
        max_number_iterations=200,
        parameters_to_fit=parameters_to_fit,
        estimator_id=gf.EstimatorID.LSE,
        user_info=user_info,
    )

    converged = states == 0
    print('\nPatlak fit summary')
    print('fits: {}'.format(number_fits))
    print('converged ratio: {:.2f}%'.format(100.0 * np.mean(converged)))
    print('mean iterations: {:.2f}'.format(np.mean(number_iterations)))
    print('mean chi square: {:.6f}'.format(np.mean(chi_squares)))
    print('mean parameters: {}'.format(np.mean(parameters, axis=0)))
    print('execution time: {:.4f}s'.format(execution_time))
