import unittest

import numpy
import numpy.testing as npt

from crm_solver.ode import Ode


class OdeTest(unittest.TestCase):

    # General setup
    DECIMALS_2 = 2
    DECIMALS_6 = 6
    START_INTERVAL = 0
    END_INTERVAL = 0.1
    STEP_NUMBER = 100
    STEPS = numpy.linspace(START_INTERVAL, END_INTERVAL, STEP_NUMBER)

    INIT_CONDITION_1D = [numpy.array([0.]), numpy.array([1.])]
    COEFF_MATRIX_1D = [numpy.array([[2.]]), numpy.array([[2.]])]
    COEFF_MATRIX_DIM_ERROR = numpy.array([15, 12])

    EXPECTED_SIZE_2 = 2
    EXPECTED_SIZE_3 = 3
    EXPECTED_SIZE_100 = 100
    EXPECTED_SIZE_200 = 200

    ACCEPTED_TYPES = [int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                      float, numpy.float16, numpy.float32, numpy.float64]

    INIT_CONDITION_GENERAL = numpy.array([1., 2.])

    # Setup CONSTANT DIAGONAL CASE
    COEFF_MATRIX_CONSTANT_DIAGONAL = numpy.array([[-0.2, 0.],
                                                  [0., -1.]])
    EXPECTED_DERIVATIVE_CONSTANT_DIAGONAL = numpy.array([-0.2, -2.])

    # Setup CONSTANT NONDIAGONAL CASE
    INIT_CONDITION_CONSTANT_NONDIAGONAL = numpy.array([3, -11, 11])
    COEFF_MATRIX_CONSTANT_NONDIAGONAL = numpy.array([[1, 3, -4],
                                                     [-1, 1, -2],
                                                     [-1, -3, 1]])
    EXPECTED_DERIVATIVE_CONSTANT_NONDIAGONAL = numpy.array([3, -35, 21])
    EXPECTED_RESULT_CONSTANT_NONDIAGONAL = numpy.array([3.38999231881, -14.9942067896, 13.5292786755])

    # Setup VARYING NONDIAGONAL CASE
    INIT_CONDITION_VARYING_NONDIAGONAL = numpy.array([2, -1])
    STEPS_VARYING_NONDIAGONAL = numpy.array([0, 0.1, 0.2, 0.3])
    START_POSITION = (STEPS[0] + STEPS[1]) / 2.

    EXPECTED_RESULT_VARYING_NONDIAGONAL = numpy.array([[2, -1],
                                                       [2.42465202, -0.4280517],
                                                       [2.82871194, -0.36454059],
                                                       [3.0456828, -0.85070421]])
    EXPECTED_DERIVATIVE_VARYING_NONDIAGONAL = numpy.array([2, 4])

    COEFF_MATRIX_CHANGING = numpy.tensordot(COEFF_MATRIX_CONSTANT_DIAGONAL, STEPS, axes=0)

    COEFF_MATRIX_VARYING_NONDIAGONAL = numpy.zeros((2, 2, STEPS_VARYING_NONDIAGONAL.size))
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 0] = numpy.array([[2, 3], [-1, -2]])
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 1] = numpy.array([[2, 1], [2, -2]])
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 2] = numpy.array([[2, -1], [5, -2]])
    COEFF_MATRIX_VARYING_NONDIAGONAL[:, :, 3] = numpy.array([[2, -3], [8, -2]])


    @classmethod
    def setUpClass(cls):
        # TODO add class level variables or constants
        pass

    def setUp(self):
        # TODO add local variables or constants that should initialize before all functions
        pass

    def tearDown(self):
        # TODO if need to destruct a class level variable
        pass

    @classmethod
    def tearDownClass(cls):
        # TODO if need to destruct a local variable after its run
        pass

    #GENERAL TESTS
    def test_calculate_analytical_solution_1d(self):
        for init in self.INIT_CONDITION_1D:
            for coefficient in self.COEFF_MATRIX_1D:
                ode = Ode(coeff_matrix=coefficient, init_condition=init)
                actual = ode.calculate_analytical_solution(self.STEPS)
                self.assertEqual(type(actual), numpy.ndarray)
                self.assertEqual(actual.size, self.EXPECTED_SIZE_100)
                for index, variable in enumerate(self.STEPS):
                    expected = ode.calculate_exp_solution(init, coefficient, variable)
                    self.assertIn(type(actual[index]), self.ACCEPTED_TYPES)
                    self.assertEqual(actual[index], expected)

    def test_set_derivative_vector_for_dimension_error(self):
        with self.assertRaises(ValueError):
            ode = Ode(coeff_matrix=self.COEFF_MATRIX_DIM_ERROR, init_condition=self.INIT_CONDITION_GENERAL)
            ode.set_derivative_vector(variable_vector=self.INIT_CONDITION_GENERAL, actual_position=self.START_POSITION,
                                      coeff_matrix=self.COEFF_MATRIX_DIM_ERROR, steps=self.STEPS)

    def test_almost_equal(self):
        actual = 1.004
        expected = 1
        self.assertAlmostEqual(actual, expected, self.DECIMALS_2)

    # CONSTANT DIAGONAL TESTS:
    def test_set_derivative_vector_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION_GENERAL)
        actual = ode.set_derivative_vector(variable_vector=self.INIT_CONDITION_GENERAL, actual_position=0,
                                           coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)
        npt.assert_almost_equal(actual, self.EXPECTED_DERIVATIVE_CONSTANT_DIAGONAL, self.DECIMALS_6)

    def test_calculate_numerical_solution_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION_GENERAL)
        actual = ode.calculate_numerical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INIT_CONDITION_GENERAL.size):
                expected = ode.calculate_exp_solution(self.INIT_CONDITION_GENERAL[j],
                                                      self.COEFF_MATRIX_CONSTANT_DIAGONAL[j, j], self.STEPS[i])
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_calculate_analytical_solution_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION_GENERAL)
        actual = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_200)
        for i in range(self.STEP_NUMBER):
            for j in range(self.INIT_CONDITION_GENERAL.size):
                expected = ode.calculate_exp_solution(self.INIT_CONDITION_GENERAL[j],
                                                      self.COEFF_MATRIX_CONSTANT_DIAGONAL[j, j], self.STEPS[i])
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)
                self.assertAlmostEqual(actual[i, j], expected, self.DECIMALS_6)

    def test_benchmark_solvers_for_constant_diagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_DIAGONAL, init_condition=self.INIT_CONDITION_GENERAL)
        numerical = ode.calculate_numerical_solution(self.STEPS)
        analytical = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(numerical.size, self.EXPECTED_SIZE_200)
        self.assertEqual(analytical.size, self.EXPECTED_SIZE_200)
        npt.assert_almost_equal(numerical, analytical, self.DECIMALS_6)

    # CONSTANT NONDIAGONAL TESTS:
    def test_set_derivative_vector_for_constant_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_CONSTANT_NONDIAGONAL)
        actual = ode.set_derivative_vector(variable_vector=self.INIT_CONDITION_CONSTANT_NONDIAGONAL, actual_position=0,
                                           coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL, steps=self.STEPS)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_3)
        npt.assert_almost_equal(actual, self.EXPECTED_DERIVATIVE_CONSTANT_NONDIAGONAL, self.DECIMALS_6)

    def test_calculate_numerical_solution_for_constant_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_CONSTANT_NONDIAGONAL)
        actual = ode.calculate_numerical_solution(self.STEPS)
        self.assertEqual(actual.size, self.STEP_NUMBER * self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size)
        self.assertEqual(actual.shape, (self.STEP_NUMBER, self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size))
        for index in range(self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size):
            self.assertIn(type(actual[-1, index]), self.ACCEPTED_TYPES)
            self.assertAlmostEqual(actual[-1, index], self.EXPECTED_RESULT_CONSTANT_NONDIAGONAL[index], self.DECIMALS_6)

    @unittest.skip('Analytical solver for constant non-diagonal case.')
    def test_calculate_analytical_solution_for_constant_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_CONSTANT_NONDIAGONAL)
        actual = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size, self.STEP_NUMBER * self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size)
        self.assertEqual(actual.shape, (self.STEP_NUMBER, self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size))
        for index in range(self.INIT_CONDITION_CONSTANT_NONDIAGONAL.size):
            self.assertIn(type(actual[-1, index]), self.ACCEPTED_TYPES)
            self.assertAlmostEqual(actual[-1, index], self.EXPECTED_RESULT_CONSTANT_NONDIAGONAL[index], self.DECIMALS_6)

    @unittest.skip('Constant Non-diagonal Solvers Benchmark.')
    def test_benchmark_solvers_for_constant_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_CONSTANT_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_CONSTANT_NONDIAGONAL)
        numerical = ode.calculate_numerical_solution(self.STEPS)
        analytical = ode.calculate_analytical_solution(self.STEPS)
        self.assertEqual(numerical.size, self.EXPECTED_SIZE_200)
        self.assertEqual(analytical.size, self.EXPECTED_SIZE_200)
        npt.assert_almost_equal(numerical, analytical, self.DECIMALS_6)

    #VARYING NONDIAGONAL TESTS:
    def test_set_derivative_vector_for_varying_nondiagonal(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_VARYING_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_VARYING_NONDIAGONAL)
        actual = ode.set_derivative_vector(variable_vector=self.INIT_CONDITION_VARYING_NONDIAGONAL,
                                           actual_position=self.STEPS_VARYING_NONDIAGONAL[1],
                                           coeff_matrix=self.COEFF_MATRIX_VARYING_NONDIAGONAL,
                                           steps=self.STEPS_VARYING_NONDIAGONAL)
        self.assertEqual(actual.size, self.EXPECTED_SIZE_2)
        npt.assert_almost_equal(actual, self.EXPECTED_DERIVATIVE_VARYING_NONDIAGONAL, self.DECIMALS_6)
        #TODO: A more general form is required

    def test_calculate_numerical_solution_for_varying_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_VARYING_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_VARYING_NONDIAGONAL)
        actual = ode.calculate_numerical_solution(self.STEPS_VARYING_NONDIAGONAL)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size,
                         self.STEPS_VARYING_NONDIAGONAL.size * self.INIT_CONDITION_VARYING_NONDIAGONAL.size)
        self.assertEqual(actual.shape,
                         (self.STEPS_VARYING_NONDIAGONAL.size, self.INIT_CONDITION_VARYING_NONDIAGONAL.size))
        npt.assert_almost_equal(actual, self.EXPECTED_RESULT_VARYING_NONDIAGONAL, self.DECIMALS_6)
        for i in range(self.STEPS_VARYING_NONDIAGONAL.size):
            for j in range(self.INIT_CONDITION_VARYING_NONDIAGONAL.size):
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)

    @unittest.skip('Analytical solver for varying non-diagonal case.')
    def test_calculate_analytical_solution_for_varying_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_VARYING_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_VARYING_NONDIAGONAL)
        actual = ode.calculate_analytical_solution(self.STEPS_VARYING_NONDIAGONAL)
        self.assertEqual(type(actual), numpy.ndarray)
        self.assertEqual(actual.size,
                         self.STEPS_VARYING_NONDIAGONAL.size * self.INIT_CONDITION_VARYING_NONDIAGONAL.size)
        self.assertEqual(actual.shape,
                         (self.STEPS_VARYING_NONDIAGONAL.size, self.INIT_CONDITION_VARYING_NONDIAGONAL.size))
        npt.assert_almost_equal(actual, self.EXPECTED_RESULT_VARYING_NONDIAGONAL, self.DECIMALS_6)
        for i in range(self.STEPS_VARYING_NONDIAGONAL.size):
            for j in range(self.INIT_CONDITION_VARYING_NONDIAGONAL.size):
                self.assertIn(type(actual[i, j]), self.ACCEPTED_TYPES)

    @unittest.skip('Varying, non-diagonal solvers Benchmark.')
    def test_benchmark_solvers_for_varying_nondiagonal_case(self):
        ode = Ode(coeff_matrix=self.COEFF_MATRIX_VARYING_NONDIAGONAL,
                  init_condition=self.INIT_CONDITION_VARYING_NONDIAGONAL)
        numerical = ode.calculate_numerical_solution(self.STEPS_VARYING_NONDIAGONAL)
        analytical = ode.calculate_numerical_solution(self.STEPS_VARYING_NONDIAGONAL)
        npt.assert_almost_equal(numerical, analytical, self.DECIMALS_6)


if __name__ == '__main__':
    unittest.main()
