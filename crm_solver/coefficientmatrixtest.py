import unittest
from atomic.atomic_db import AtomicDB
from crm_solver.coefficientmatrix import CoefficientMatrix
from utility.input import BeamletInput
import numpy
import pandas


class CoefficientMatrixTest(unittest.TestCase):

    INPUT_q = [1, 1, 1, 2, 4]
    INPUT_z = [1, 1, 2, 2, 4]
    INPUT_a = [1, 2, 3, 4, 9]
    INPUT_m = [None, None, None, None, None]

    INPUT_neutral_q = [-1, 0, 1]
    INPUT_neutral_z = [0, 1, 1]
    INPUT_neutral_a = [0, 1, 1]
    INPUT_neutral_m = [None, None, None]

    INPUT_GRID = numpy.array([1., 2., 3., 4., 5., 6., 7.])
    INPUT_TEMPERATURE = numpy.array([0, 1, 2, 2.5, 3, 8, 10])
    INPUT_DENSITY = numpy.array([1]*len(INPUT_GRID))

    EXPECTED_NEUTRAL_COUNT = 1
    EXPECTED_NEUTRAL_ATOMIC_LEVELS = 4
    EXPECTED_DECIMAL_PRECISION_4 = 4
    EXPECTED_DECIMAL_PRECISION_6 = 6
    EXPECTED_ATTRIBUTES = ['matrix', 'electron_terms', 'ion_terms', 'photon_terms', 'beamlet_profiles',
                           'electron_impact_trans_np', 'electron_impact_loss_np', 'ion_impact_trans_np',
                           'ion_impact_loss_np']
    EXPECTED_ELECTRON_IMPACT_TERMS = numpy.array([[[0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0012, 0.0112, 0.0212, 0.0262, 0.0312, 0.0812, 0.1012],
                                                   [0.0013, 0.0113, 0.0213, 0.0263, 0.0313, 0.0813, 0.1013]],
                                                  [[0.0021, 0.0121, 0.0221, 0.0271, 0.0321, 0.0821, 0.1021],
                                                   [0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0023, 0.0123, 0.0223, 0.0273, 0.0323, 0.0823, 0.1023]],
                                                  [[0.0031, 0.0131, 0.0231, 0.0281, 0.0331, 0.0831, 0.1031],
                                                   [0.0032, 0.0132, 0.0232, 0.0282, 0.0332, 0.0832, 0.1032],
                                                   [0., 0., 0., 0., 0., 0., 0.]]])
    EXPECTED_ELECTRON_LOSS_TERMS = numpy.array([[0.0011, 0.0111, 0.0211, 0.0261, 0.0311, 0.0811, 0.1011],
                                                [0.0021, 0.0121, 0.0221, 0.0271, 0.0321, 0.0821, 0.1021],
                                                [0.0031, 0.0131, 0.0231, 0.0281, 0.0331, 0.0831, 0.1031]])
    EXPECTED_NEUTRAL_LOSS_TERMS = numpy.array([[[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
                                                [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002],
                                                [0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003],
                                                [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004]]])
    EXPECTED_NEUTRAL_TRANS_TERMS = numpy.array([[[[0., 0., 0., 0., 0., 0., 0.],
                                                  [0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012],
                                                  [0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013],
                                                  [0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014]],
                                                 [[0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021],
                                                  [0., 0., 0., 0., 0., 0., 0.],
                                                  [0.0023, 0.0023, 0.0023, 0.0023, 0.0023, 0.0023, 0.0023],
                                                  [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024]],
                                                 [[0.0031, 0.0031, 0.0031, 0.0031, 0.0031, 0.0031, 0.0031],
                                                  [0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032],
                                                  [0., 0., 0., 0., 0., 0., 0.],
                                                  [0.0034, 0.0034, 0.0034, 0.0034, 0.0034, 0.0034, 0.0034]],
                                                 [[0.0041, 0.0041, 0.0041, 0.0041, 0.0041, 0.0041, 0.0041],
                                                  [0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.0042],
                                                  [0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043],
                                                  [0., 0., 0., 0., 0., 0., 0.]]]])
    EXPECTED_NEUTRAL_TERMS = numpy.array([[[[-0.004, -0.004, -0.004, -0.004, -0.004, -0.004, -0.004],
                                            [0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012],
                                            [0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013],
                                            [0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014]],
                                           [[0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021, 0.0021],
                                            [-0.007, -0.007, -0.007, -0.007, -0.007, -0.007, -0.007],
                                            [0.0023, 0.0023, 0.0023, 0.0023, 0.0023, 0.0023, 0.0023],
                                            [0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024, 0.0024]],
                                           [[0.0031, 0.0031, 0.0031, 0.0031, 0.0031, 0.0031, 0.0031],
                                            [0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032, 0.0032],
                                            [-0.01, - 0.01, - 0.01, - 0.01, - 0.01, - 0.01, - 0.01],
                                            [0.0034, 0.0034, 0.0034, 0.0034, 0.0034, 0.0034, 0.0034]],
                                           [[0.0041, 0.0041, 0.0041, 0.0041, 0.0041, 0.0041, 0.0041],
                                            [0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.0042, 0.0042],
                                            [0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043],
                                            [-0.013, - 0.013, - 0.013, - 0.013, - 0.013, - 0.013, - 0.013]]]])
    EXPECTED_ION_LOSS_TERMS = numpy.array([[[0.0012, 0.0112, 0.0212, 0.0262, 0.0312, 0.0812, 0.1012],
                                            [0.0022, 0.0122, 0.0222, 0.0272, 0.0322, 0.0822, 0.1022],
                                            [0.0032, 0.0132, 0.0232, 0.0282, 0.0332, 0.0832, 0.1032]],
                                           [[0.0012, 0.0062, 0.0112, 0.0137, 0.0162, 0.0412, 0.0512],
                                            [0.0022, 0.0072, 0.0122, 0.0147, 0.0172, 0.0422, 0.0522],
                                            [0.0032, 0.0082, 0.0132, 0.0157, 0.0182, 0.0432, 0.0532]],
                                           [[0.0012, 0.00453333, 0.00786667, 0.00953333, 0.0112, 0.02786667, 0.03453333],
                                            [0.0022, 0.00553333, 0.00886667, 0.01053333, 0.0122, 0.02886667, 0.03553333],
                                            [0.0032, 0.00653333, 0.00986667, 0.01153333, 0.0132, 0.02986667, 0.03653333]],
                                           [[0.0013, 0.0113, 0.0213, 0.0263, 0.0313, 0.0813, 0.1013],
                                            [0.0023, 0.0123, 0.0223, 0.0273, 0.0323, 0.0823, 0.1023],
                                            [0.0033, 0.0133, 0.0233, 0.0283, 0.0333, 0.0833, 0.1033]],
                                           [[0.0015, 0.0115, 0.0215, 0.0265, 0.0315, 0.0815, 0.1015],
                                            [0.0025, 0.0125, 0.0225, 0.0275, 0.0325, 0.0825, 0.1025],
                                            [0.0035, 0.0135, 0.0235, 0.0285, 0.0335, 0.0835, 0.1035]]])
    EXPECTED_ION_TRANSITION_TERMS = numpy.array([[[[0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0012, 0.0112, 0.0212, 0.0262, 0.0312, 0.0812, 0.1012],
                                                   [0.0013, 0.0113, 0.0213, 0.0263, 0.0313, 0.0813, 0.1013]],
                                                  [[0.0021, 0.0121, 0.0221, 0.0271, 0.0321, 0.0821, 0.1021],
                                                   [0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0023, 0.0123, 0.0223, 0.0273, 0.0323, 0.0823, 0.1023]],
                                                  [[0.0031, 0.0131, 0.0231, 0.0281, 0.0331, 0.0831, 0.1031],
                                                   [0.0032, 0.0132, 0.0232, 0.0282, 0.0332, 0.0832, 0.1032],
                                                   [0., 0., 0., 0., 0., 0., 0.]]],
                                                 [[[0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0012, 0.0062, 0.0112, 0.0137, 0.0162, 0.0412, 0.0512],
                                                   [0.0013, 0.0063, 0.0113, 0.0138, 0.0163, 0.0413, 0.0513]],
                                                  [[0.0021, 0.0071, 0.0121, 0.0146, 0.0171, 0.0421, 0.0521],
                                                   [0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0023, 0.0073, 0.0123, 0.0148, 0.0173, 0.0423, 0.0523]],
                                                  [[0.0031, 0.0081, 0.0131, 0.0156, 0.0181, 0.0431, 0.0531],
                                                   [0.0032, 0.0082, 0.0132, 0.0157, 0.0182, 0.0432, 0.0532],
                                                   [0., 0., 0., 0., 0., 0., 0.]]],
                                                 [[[0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0012, 0.00453333, 0.00786667, 0.00953333, 0.0112, 0.02786667, 0.03453333],
                                                   [0.0013, 0.00463333, 0.00796667, 0.00963333, 0.0113, 0.02796667, 0.03463333]],
                                                  [[0.0021, 0.00543333, 0.00876667, 0.01043333, 0.0121, 0.02876667, 0.03543333],
                                                   [0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0023, 0.00563333, 0.00896667, 0.01063333, 0.0123, 0.02896667, 0.03563333]],
                                                  [[0.0031, 0.00643333, 0.00976667, 0.01143333, 0.0131, 0.02976667, 0.03643333],
                                                   [0.0032, 0.00653333, 0.00986667, 0.01153333, 0.0132, 0.02986667, 0.03653333],
                                                   [0., 0., 0., 0., 0., 0., 0.]]],
                                                 [[[0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0121, 0.1121, 0.2121, 0.2621, 0.3121, 0.8121, 1.0121],
                                                   [0.0131, 0.1131, 0.2131, 0.2631, 0.3131, 0.8131, 1.0131]],
                                                  [[0.0211, 0.1211, 0.2211, 0.2711, 0.3211, 0.8211, 1.0211],
                                                   [0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0231, 0.1231, 0.2231, 0.2731, 0.3231, 0.8231, 1.0231]],
                                                  [[0.0311, 0.1311, 0.2311, 0.2811, 0.3311, 0.8311, 1.0311],
                                                   [0.0321, 0.1321, 0.2321, 0.2821, 0.3321, 0.8321, 1.0321],
                                                   [0., 0., 0., 0., 0., 0., 0.]]],
                                                 [[[0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0123, 0.1123, 0.2123, 0.2623, 0.3123, 0.8123, 1.0123],
                                                   [0.0133, 0.1133, 0.2133, 0.2633, 0.3133, 0.8133, 1.0133]],
                                                  [[0.0213, 0.1213, 0.2213, 0.2713, 0.3213, 0.8213, 1.0213],
                                                   [0., 0., 0., 0., 0., 0., 0.],
                                                   [0.0233, 0.1233, 0.2233, 0.2733, 0.3233, 0.8233, 1.0233]],
                                                  [[0.0313, 0.1313, 0.2313, 0.2813, 0.3313, 0.8313, 1.0313],
                                                   [0.0323, 0.1323, 0.2323, 0.2823, 0.3323, 0.8323, 1.0323],
                                                   [0., 0., 0., 0., 0., 0., 0.]]]])
    EXPECTED_RATE_COEFFICIENT_MATRIX = \
        numpy.array([[[-3.75700000e-01, -3.22570000e+00, -6.07570000e+00, -7.50070000e+00,
                       -8.92570000e+00, -2.31757000e+01, -2.88757000e+01],
                      [+1.67500000e-01, 1.50750000e+00, 2.84750000e+00, 3.51750000e+00,
                       4.18750000e+00, 1.08875000e+01, 1.35675000e+01],
                      [+1.81200000e-01, 1.52120000e+00, 2.86120000e+00, 3.53120000e+00,
                       4.20120000e+00, 1.09012000e+01, 1.35812000e+01]],
                     [[+5.86859823e+02, 5.88199823e+02, 5.89539823e+02, 5.90209823e+02,
                       5.90879823e+02, 5.97579823e+02, 6.00259823e+02],
                      [-5.87225023e+02, -5.90075023e+02, -5.92925023e+02, -5.94350023e+02,
                       -5.95775023e+02, -6.10025023e+02, -6.15725023e+02],
                      [+3.18200000e-01, 1.65820000e+00, 2.99820000e+00, 3.66820000e+00,
                       4.33820000e+00, 1.10382000e+01, 1.37182000e+01]],
                     [[+8.66315406e+02, 8.67655406e+02, 8.68995406e+02, 8.69665406e+02,
                       8.70335406e+02, 8.77035406e+02, 8.79715406e+02],
                      [+8.94260964e+02, 8.95600964e+02, 8.96940964e+02, 8.97610964e+02,
                       8.98280964e+02, 9.04980964e+02, 9.07660964e+02],
                      [-1.76064337e+03, -1.76349337e+03, -1.76634337e+03, -1.76776837e+03,
                       -1.76919337e+03, -1.78344337e+03, -1.78914337e+03]]])
    EXPECTED_PHOTON_TERM = \
        numpy.array([[[0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0.]],
                     [[586.56902347, 586.56902347, 586.56902347, 586.56902347,
                       586.56902347, 586.56902347, 586.56902347],
                      [-586.56902347, -586.56902347, -586.56902347, -586.56902347,
                       -586.56902347, -586.56902347, -586.56902347],
                      [0., 0., 0., 0., 0., 0., 0.]],
                     [[865.88760608, 865.88760608, 865.88760608, 865.88760608,
                       865.88760608, 865.88760608, 865.88760608],
                      [893.81946434, 893.81946434, 893.81946434, 893.81946434,
                       893.81946434, 893.81946434, 893.81946434],
                      [-1759.70707042, -1759.70707042, -1759.70707042, -1759.70707042,
                       -1759.70707042, -1759.70707042, -1759.70707042]]])
    EXPECTED_ELECTRON_TERM = \
        numpy.array([[[-0.0036, -0.0336, -0.0636, -0.0786, -0.0936, -0.2436, -0.3036],
                      [0.0012, 0.0112, 0.0212, 0.0262, 0.0312, 0.0812, 0.1012],
                      [0.0013, 0.0113, 0.0213, 0.0263, 0.0313, 0.0813, 0.1013]],
                     [[0.0021, 0.0121, 0.0221, 0.0271, 0.0321, 0.0821, 0.1021],
                      [-0.0065, -0.0365, -0.0665, -0.0815, -0.0965, -0.2465, -0.3065],
                      [0.0023, 0.0123, 0.0223, 0.0273, 0.0323, 0.0823, 0.1023]],
                     [[0.0031, 0.0131, 0.0231, 0.0281, 0.0331, 0.0831, 0.1031],
                      [0.0032, 0.0132, 0.0232, 0.0282, 0.0332, 0.0832, 0.1032],
                      [-0.0094, -0.0394, -0.0694, -0.0844, -0.0994, -0.2494, -0.3094]]])
    EXPECTED_ION_TERM = \
        numpy.array([[[[-3.70000000e-03, -3.37000000e-02, -6.37000000e-02, -7.87000000e-02, -9.37000000e-02,
                        -2.43700000e-01, -3.03700000e-01],
                       [1.20000000e-03, 1.12000000e-02, 2.12000000e-02, 2.62000000e-02, 3.12000000e-02,
                        8.12000000e-02, 1.01200000e-01],
                       [1.30000000e-03, 1.13000000e-02, 2.13000000e-02, 2.63000000e-02, 3.13000000e-02,
                        8.13000000e-02, 1.01300000e-01]],
                      [[2.10000000e-03, 1.21000000e-02, 2.21000000e-02, 2.71000000e-02, 3.21000000e-02,
                        8.21000000e-02, 1.02100000e-01],
                       [-6.60000000e-03, -3.66000000e-02, -6.66000000e-02, -8.16000000e-02, -9.66000000e-02,
                        -2.46600000e-01, -3.06600000e-01],
                       [2.30000000e-03, 1.23000000e-02, 2.23000000e-02, 2.73000000e-02, 3.23000000e-02,
                        8.23000000e-02, 1.02300000e-01]],
                      [[3.10000000e-03, 1.31000000e-02, 2.31000000e-02, 2.81000000e-02, 3.31000000e-02,
                        8.31000000e-02, 1.03100000e-01],
                       [3.20000000e-03, 1.32000000e-02, 2.32000000e-02, 2.82000000e-02, 3.32000000e-02,
                        8.32000000e-02, 1.03200000e-01],
                       [-9.50000000e-03, -3.95000000e-02, -6.95000000e-02, -8.45000000e-02, -9.95000000e-02,
                        -2.49500000e-01, -3.09500000e-01]]],
                     [[[-3.70000000e-03, -1.87000000e-02, -3.37000000e-02, -4.12000000e-02, -4.87000000e-02,
                        -1.23700000e-01, -1.53700000e-01],
                       [1.20000000e-03, 6.20000000e-03, 1.12000000e-02, 1.37000000e-02, 1.62000000e-02,
                         4.12000000e-02, 5.12000000e-02],
                       [1.30000000e-03, 6.30000000e-03, 1.13000000e-02, 1.38000000e-02, 1.63000000e-02,
                         4.13000000e-02, 5.13000000e-02]],
                      [[2.10000000e-03, 7.10000000e-03, 1.21000000e-02, 1.46000000e-02, 1.71000000e-02,
                        4.21000000e-02, 5.21000000e-02],
                       [-6.60000000e-03, -2.16000000e-02, -3.66000000e-02, -4.41000000e-02, -5.16000000e-02,
                        -1.26600000e-01, -1.56600000e-01],
                       [2.30000000e-03, 7.30000000e-03, 1.23000000e-02, 1.48000000e-02, 1.73000000e-02,
                        4.23000000e-02, 5.23000000e-02]],
                      [[3.10000000e-03, 8.10000000e-03, 1.31000000e-02, 1.56000000e-02, 1.81000000e-02,
                        4.31000000e-02, 5.31000000e-02],
                       [3.20000000e-03, 8.20000000e-03, 1.32000000e-02, 1.57000000e-02, 1.82000000e-02,
                        4.32000000e-02, 5.32000000e-02],
                       [-9.50000000e-03, -2.45000000e-02, -3.95000000e-02, -4.70000000e-02, -5.45000000e-02,
                        -1.29500000e-01, -1.59500000e-01]]],
                     [[[-3.70000000e-03, -1.37000000e-02, -2.37000000e-02, -2.87000000e-02, -3.37000000e-02,
                        -8.37000000e-02, -1.03700000e-01],
                       [1.20000000e-03, 4.53333333e-03, 7.86666667e-03, 9.53333333e-03, 1.12000000e-02,
                         2.78666667e-02, 3.45333333e-02],
                       [1.30000000e-03, 4.63333333e-03, 7.96666667e-03, 9.63333333e-03, 1.13000000e-02,
                         2.79666667e-02, 3.46333333e-02]],
                      [[2.10000000e-03, 5.43333333e-03, 8.76666667e-03, 1.04333333e-02, 1.21000000e-02,
                        2.87666667e-02, 3.54333333e-02],
                       [-6.60000000e-03, -1.66000000e-02, -2.66000000e-02, -3.16000000e-02, -3.66000000e-02,
                        -8.66000000e-02, -1.06600000e-01],
                       [2.30000000e-03, 5.63333333e-03, 8.96666667e-03, 1.06333333e-02, 1.23000000e-02,
                        2.89666667e-02, 3.56333333e-02]],
                      [[3.10000000e-03, 6.43333333e-03, 9.76666667e-03, 1.14333333e-02, 1.31000000e-02,
                        2.97666667e-02, 3.64333333e-02],
                       [3.20000000e-03, 6.53333333e-03, 9.86666667e-03, 1.15333333e-02, 1.32000000e-02,
                        2.98666667e-02, 3.65333333e-02],
                       [-9.50000000e-03, -1.95000000e-02, -2.95000000e-02, -3.45000000e-02, -3.95000000e-02,
                        -8.95000000e-02, -1.09500000e-01]]],
                     [[[-2.65000000e-02, -2.36500000e-01, -4.46500000e-01, -5.51500000e-01, -6.56500000e-01,
                        -1.70650000e+00, -2.12650000e+00],
                       [1.21000000e-02, 1.12100000e-01, 2.12100000e-01, 2.62100000e-01, 3.12100000e-01,
                         8.12100000e-01, 1.01210000e+00],
                       [1.31000000e-02, 1.13100000e-01, 2.13100000e-01, 2.63100000e-01, 3.13100000e-01,
                         8.13100000e-01, 1.01310000e+00]],
                      [[2.11000000e-02, 1.21100000e-01, 2.21100000e-01, 2.71100000e-01, 3.21100000e-01,
                        8.21100000e-01, 1.02110000e+00],
                       [-4.65000000e-02, -2.56500000e-01, -4.66500000e-01, -5.71500000e-01, -6.76500000e-01,
                        -1.72650000e+00, -2.14650000e+00],
                       [2.31000000e-02, 1.23100000e-01, 2.23100000e-01, 2.73100000e-01, 3.23100000e-01,
                        8.23100000e-01, 1.02310000e+00]],
                      [[3.11000000e-02, 1.31100000e-01, 2.31100000e-01, 2.81100000e-01, 3.31100000e-01,
                        8.31100000e-01, 1.03110000e+00],
                       [3.21000000e-02, 1.32100000e-01, 2.32100000e-01, 2.82100000e-01, 3.32100000e-01,
                        8.32100000e-01, 1.03210000e+00],
                       [-6.65000000e-02, -2.76500000e-01, -4.86500000e-01, -5.91500000e-01, -6.96500000e-01,
                        -1.74650000e+00, -2.16650000e+00]]],
                     [[[-2.71000000e-02, -2.37100000e-01, -4.47100000e-01, -5.52100000e-01, -6.57100000e-01,
                        -1.70710000e+00, -2.12710000e+00],
                       [1.23000000e-02, 1.12300000e-01, 2.12300000e-01, 2.62300000e-01, 3.12300000e-01,
                         8.12300000e-01, 1.01230000e+00],
                       [1.33000000e-02, 1.13300000e-01, 2.13300000e-01, 2.63300000e-01, 3.13300000e-01,
                         8.13300000e-01, 1.01330000e+00]],
                      [[2.13000000e-02, 1.21300000e-01, 2.21300000e-01, 2.71300000e-01, 3.21300000e-01,
                        8.21300000e-01, 1.02130000e+00],
                       [-4.71000000e-02, -2.57100000e-01, -4.67100000e-01, -5.72100000e-01, -6.77100000e-01,
                        -1.72710000e+00, -2.14710000e+00],
                       [2.33000000e-02, 1.23300000e-01, 2.23300000e-01, 2.73300000e-01, 3.23300000e-01,
                        8.23300000e-01, 1.02330000e+00]],
                      [[3.13000000e-02, 1.31300000e-01, 2.31300000e-01, 2.81300000e-01, 3.31300000e-01,
                        8.31300000e-01, 1.03130000e+00],
                       [3.23000000e-02, 1.32300000e-01, 2.32300000e-01, 2.82300000e-01, 3.32300000e-01,
                        8.32300000e-01, 1.03230000e+00],
                       [-6.71000000e-02, -2.77100000e-01, -4.87100000e-01, -5.92100000e-01, -6.97100000e-01,
                        -1.74710000e+00, -2.16710000e+00]]]])

    def setUp(self):
        self.BEAMLET_PARAM, self.COMPONENTS, self.PROFILES = self._build_beamlet_input()
        self.ATOMIC_DB = AtomicDB(param=self.BEAMLET_PARAM, components=self.COMPONENTS)
        self.RATE_COEFFICIENT = CoefficientMatrix(self.BEAMLET_PARAM, self.PROFILES, self.COMPONENTS, self.ATOMIC_DB)

    def tearDown(self):
        del self.RATE_COEFFICIENT

    def test_all_attributes(self):
        for attr in self.EXPECTED_ATTRIBUTES:
            assert hasattr(self.RATE_COEFFICIENT, attr)

    def test_electron_impact_transition(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.electron_impact_trans_np, numpy.ndarray,
                              msg='The electron impact transition terms is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.electron_impact_trans_np.shape,
                              (self.ATOMIC_DB.atomic_ceiling, self.ATOMIC_DB.atomic_ceiling,
                               self.PROFILES['beamlet grid'].size), msg='The electron impact transition '
                                                                        'terms is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.electron_impact_trans_np,
                                          self.EXPECTED_ELECTRON_IMPACT_TERMS, self.EXPECTED_DECIMAL_PRECISION_6,
                                          err_msg='Interpolation failure for electron impact transitions.')

    def test_electron_impact_loss(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.electron_impact_loss_np, numpy.ndarray, msg='The electron impact '
                              'ionization terms is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.electron_impact_loss_np.shape, (self.ATOMIC_DB.atomic_ceiling,
                              self.PROFILES['beamlet grid'].size), msg='The electron impact ionization term is '
                                                                       'not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.electron_impact_loss_np,
                                          self.EXPECTED_ELECTRON_LOSS_TERMS,
                                          self.EXPECTED_DECIMAL_PRECISION_6,
                                          err_msg='Interpolation failure for electron impact loss.')

    def test_ion_impact_loss(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.electron_impact_loss_np, numpy.ndarray, msg='The ion impact '
                              'ionization terms is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.ion_impact_loss_np.shape, (len(self.COMPONENTS.T.keys()) - 1,
                              self.ATOMIC_DB.atomic_ceiling, self.PROFILES['beamlet grid'].size), msg='The ion impact '
                              'ionization term is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.ion_impact_loss_np, self.EXPECTED_ION_LOSS_TERMS,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Interpolation failure '
                                                                                     'for ion impact loss.')

    def test_ion_impact_transition(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.ion_impact_trans_np, numpy.ndarray,
                              msg='The ion impact transition terms is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.ion_impact_trans_np.shape, (len(self.COMPONENTS.T.keys()) - 1,
                              self.ATOMIC_DB.atomic_ceiling, self.ATOMIC_DB.atomic_ceiling,
                              self.PROFILES['beamlet grid'].size), msg='The ion impact transition term is not '
                                                                       'dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.ion_impact_trans_np, self.EXPECTED_ION_TRANSITION_TERMS,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Interpolation failure for '
                                                                                     'ion impact transition.')

    def test_rate_matrix(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.matrix, numpy.ndarray,
                              msg='The rate coefficient matrix is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.matrix.shape, (self.ATOMIC_DB.atomic_ceiling,
                              self.ATOMIC_DB.atomic_ceiling, self.PROFILES['beamlet grid'].size),
                              msg='The rate coefficient matrix is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.matrix, self.EXPECTED_RATE_COEFFICIENT_MATRIX,
                                          self.EXPECTED_DECIMAL_PRECISION_4, err_msg='Rate coefficient matrix assembly '
                                                                                     'and generation failed.')

    def test_spontaneous_rate_term_assembly(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.photon_terms, numpy.ndarray,
                              msg='The spontaneous photon term is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.photon_terms.shape, (self.ATOMIC_DB.atomic_ceiling,
                              self.ATOMIC_DB.atomic_ceiling, self.PROFILES['beamlet grid'].size),
                              msg='The photon term is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.photon_terms, self.EXPECTED_PHOTON_TERM,
                                          self.EXPECTED_DECIMAL_PRECISION_4, err_msg='Photon term assembly failed.')

    def test_spontaneous_term_application(self):
        self.RATE_COEFFICIENT.matrix -= self.RATE_COEFFICIENT.matrix
        for step in range(self.PROFILES['beamlet grid'].size):
            self.RATE_COEFFICIENT.apply_photons(step)
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.matrix, self.EXPECTED_PHOTON_TERM,
                                          self.EXPECTED_DECIMAL_PRECISION_4, err_msg='Photon term application failed.')

    def test_electron_rate_term_assembly(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.electron_terms, numpy.ndarray,
                              msg='The electron rate term is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.electron_terms.shape, (self.ATOMIC_DB.atomic_ceiling,
                              self.ATOMIC_DB.atomic_ceiling, self.PROFILES['beamlet grid'].size),
                              msg='The electron term is dimensionally not accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.electron_terms, self.EXPECTED_ELECTRON_TERM,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Electron term assembly failed.')

    def test_electron_term_application(self):
        self.RATE_COEFFICIENT.matrix -= self.RATE_COEFFICIENT.matrix
        actual = numpy.zeros((self.ATOMIC_DB.atomic_ceiling, self.ATOMIC_DB.atomic_ceiling,
                              self.PROFILES['beamlet grid'].size))
        for step in range(self.PROFILES['beamlet grid'].size):
            self.RATE_COEFFICIENT.apply_electron_density(step)
            actual[:, :, step] = self.PROFILES['electron']['density']['m-3'][step] \
                * self.EXPECTED_ELECTRON_TERM[:, :, step]
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.matrix, actual, self.EXPECTED_DECIMAL_PRECISION_6,
                                          err_msg='Electron term and density application failed.')

    def test_ion_rate_term_assembly(self):
        self.assertIsInstance(self.RATE_COEFFICIENT.ion_terms, numpy.ndarray,
                              msg='The ion term is not in the expected format.')
        self.assertTupleEqual(self.RATE_COEFFICIENT.ion_terms.shape, (len(self.COMPONENTS.T.keys()) - 1,
                              self.ATOMIC_DB.atomic_ceiling, self.ATOMIC_DB.atomic_ceiling,
                              self.PROFILES['beamlet grid'].size), msg='The ion term is dimensionally not accurate.')
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.ion_terms, self.EXPECTED_ION_TERM,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Ion rate term assembly failure.')

    def test_ion_rate_term_application(self):
        self.RATE_COEFFICIENT.matrix -= self.RATE_COEFFICIENT.matrix
        actual = numpy.zeros((self.ATOMIC_DB.atomic_ceiling, self.ATOMIC_DB.atomic_ceiling,
                              self.PROFILES['beamlet grid'].size))
        for ion in range(len(self.COMPONENTS.T.keys())-1):
            for step in range(self.PROFILES['beamlet grid'].size):
                self.RATE_COEFFICIENT.apply_ion_density(ion, step)
                actual[:, :, step] += self.PROFILES['ion'+str(ion+1)]['density']['m-3'][step] * \
                    self.EXPECTED_ION_TERM[ion, :, :, step]
        numpy.testing.assert_almost_equal(self.RATE_COEFFICIENT.matrix, actual, self.EXPECTED_DECIMAL_PRECISION_6,
                                          err_msg='Ion term and density application failed.')

    def test_ceiled_electron_impact_loss(self):
        ceiled_atomic_db = AtomicDB(param=self.BEAMLET_PARAM, components=self.COMPONENTS, atomic_ceiling=2)
        ceiled_rate_coefficient = CoefficientMatrix(self.BEAMLET_PARAM, self.PROFILES,
                                                    self.COMPONENTS, ceiled_atomic_db)
        self.assertIsInstance(ceiled_rate_coefficient.electron_impact_loss_np, numpy.ndarray, msg='The electron impact '
                              'ionization terms is not in the expected format.')
        self.assertTupleEqual(ceiled_rate_coefficient.electron_impact_loss_np.shape, (ceiled_atomic_db.atomic_ceiling,
                              self.PROFILES['beamlet grid'].size), msg='The electron impact ionization term is '
                                                                       'not dimensionally accurate.')
        numpy.testing.assert_almost_equal(ceiled_rate_coefficient.electron_impact_loss_np,
                                          self.EXPECTED_ELECTRON_LOSS_TERMS[0:ceiled_atomic_db.atomic_ceiling, :],
                                          self.EXPECTED_DECIMAL_PRECISION_6,
                                          err_msg='Interpolation failure for electron impact loss.')

    def test_neutral_impact_loss(self):
        self.setup_neutral()
        self.assertIsInstance(self.neutral_rate_coefficient.neutral_impact_loss_np, numpy.ndarray, msg='The neutral '
                              'impact ionization terms is not in the expected format.')
        self.assertTupleEqual(self.neutral_rate_coefficient.neutral_impact_loss_np.shape,
                              (self.EXPECTED_NEUTRAL_COUNT, self.EXPECTED_NEUTRAL_ATOMIC_LEVELS,
                               self.PROFILES['beamlet grid'].size), msg='The neutral impact '
                              'ionization term is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.neutral_rate_coefficient.neutral_impact_loss_np,
                                          self.EXPECTED_NEUTRAL_LOSS_TERMS, self.EXPECTED_DECIMAL_PRECISION_4,
                                          err_msg='Interpolation failure for neutral impact loss.')

    def test_neutral_impact_transition(self):
        self.setup_neutral()
        self.assertIsInstance(self.neutral_rate_coefficient.neutral_impact_trans_np, numpy.ndarray, msg='The neutral '
                              'impact transition terms is not in the expected format.')
        self.assertTupleEqual(self.neutral_rate_coefficient.neutral_impact_trans_np.shape,
                              (self.EXPECTED_NEUTRAL_COUNT, self.EXPECTED_NEUTRAL_ATOMIC_LEVELS,
                               self.EXPECTED_NEUTRAL_ATOMIC_LEVELS,
                               self.PROFILES['beamlet grid'].size), msg='The neutral impact '
                              'transition term is not dimensionally accurate.')
        numpy.testing.assert_almost_equal(self.neutral_rate_coefficient.neutral_impact_trans_np,
                                          self.EXPECTED_NEUTRAL_TRANS_TERMS, self.EXPECTED_DECIMAL_PRECISION_4,
                                          err_msg='Interpolation failure for neutral impact transition.')

    def test_neutral_assembly(self):
        self.setup_neutral()
        self.assertIsInstance(self.neutral_rate_coefficient.neutral_terms, numpy.ndarray,
                              msg='The neutral rate term is not in the expected format.')
        self.assertTupleEqual(self.neutral_rate_coefficient.neutral_terms.shape, (self.EXPECTED_NEUTRAL_COUNT,
                              self.EXPECTED_NEUTRAL_ATOMIC_LEVELS, self.EXPECTED_NEUTRAL_ATOMIC_LEVELS,
                              self.PROFILES['beamlet grid'].size),
                              msg='The neutral term is dimensionally not accurate.')
        numpy.testing.assert_almost_equal(self.neutral_rate_coefficient.neutral_terms, self.EXPECTED_NEUTRAL_TERMS,
                                          self.EXPECTED_DECIMAL_PRECISION_6, err_msg='Neutral term assembly failed.')

    def _build_beamlet_input(self):
        input_gen = BeamletInput(energy=60, projectile='dummy', param_name='Coefficient Matrix Test',
                                 source='Unittest', current=0.001)
        input_gen.add_grid(self.INPUT_GRID)
        input_gen.add_target_profiles(charge=-1, atomic_number=0, mass_number=0, molecule_name=None,
                                      density=self.INPUT_DENSITY, temperature=self.INPUT_TEMPERATURE)
        for index in range(len(self.INPUT_q)):
            input_gen.add_target_profiles(charge=self.INPUT_q[index], atomic_number=self.INPUT_z[index],
                                          mass_number=self.INPUT_a[index], molecule_name=self.INPUT_m[index],
                                          density=self.INPUT_DENSITY*self.INPUT_a[index],
                                          temperature=self.INPUT_TEMPERATURE)
        return input_gen.get_beamlet_input()

    def _build_beamlet_neutral_input(self):
        input_gen = BeamletInput(energy=50, projectile='H', param_name='Coefficient Neutral Matrix Test',
                                 source='Unittest', current=0.001)
        input_gen.add_grid(self.INPUT_GRID)
        for index in range(len(self.INPUT_neutral_q)):
            if self.INPUT_neutral_q != 0:
                density = self.INPUT_DENSITY * 0
            else:
                density = self.INPUT_DENSITY * 2
            input_gen.add_target_profiles(charge=self.INPUT_neutral_q[index],
                                          atomic_number=self.INPUT_neutral_z[index],
                                          mass_number=self.INPUT_neutral_a[index],
                                          molecule_name=self.INPUT_neutral_m[index],
                                          density=density,
                                          temperature=self.INPUT_TEMPERATURE)
        return input_gen.get_beamlet_input()

    def setup_neutral(self):
        self.neutral_param, self.neutral_components, self.neutral_profiles = self._build_beamlet_neutral_input()
        self.neutral_atomic = AtomicDB(param=self.neutral_param, components=self.neutral_components, resolution='test')
        self.neutral_rate_coefficient = CoefficientMatrix(beamlet_param=self.neutral_param,
                                                          beamlet_profiles=self.neutral_profiles,
                                                          plasma_components=self.neutral_components,
                                                          atomic_db=self.neutral_atomic)
