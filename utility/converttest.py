import math
import unittest

import numpy
import scipy.constants as c

from utility import convert as uc


class ConvertTest(unittest.TestCase):

    def test_convert_from_cm2_to_m2_scalar(self):
        self.assertAlmostEqual(uc.convert_from_cm2_to_m2(1.0), 1.0e-4,
                               msg='1 cm2 is expected to convert to 1e-4 m2.')

    def test_convert_from_cm2_to_m2_array(self):
        actual = uc.convert_from_cm2_to_m2(numpy.asarray([1.0, 2.0, 4.0]))
        numpy.testing.assert_allclose(actual, numpy.asarray([1.0e-4, 2.0e-4, 4.0e-4]),
                                      err_msg='Array cross-section conversion is expected to scale every '
                                              'element by 1e-4.')

    def test_convert_from_m2_to_cm2_scalar(self):
        self.assertAlmostEqual(uc.convert_from_m2_to_cm2(1.0), 1.0e4,
                               msg='1 m2 is expected to convert to 1e4 cm2.')

    def test_convert_from_m2_to_cm2_array(self):
        actual = uc.convert_from_m2_to_cm2(numpy.asarray([1.0, 2.0, 4.0]))
        numpy.testing.assert_allclose(actual, numpy.asarray([1.0e4, 2.0e4, 4.0e4]),
                                      err_msg='Array cross-section conversion is expected to scale every '
                                              'element by 1e4.')

    def test_convert_from_cm_to_m_scalar(self):
        self.assertAlmostEqual(uc.convert_from_cm_to_m(1.0), 1.0e-2,
                               msg='1 cm is expected to convert to 1e-2 m.')

    def test_convert_from_cm_to_m_array(self):
        actual = uc.convert_from_cm_to_m(numpy.asarray([1.0, 10.0]))
        numpy.testing.assert_allclose(actual, numpy.asarray([0.01, 0.1]),
                                      err_msg='Array length conversion is expected to scale every element by 1e-2.')

    def test_convert_from_10_19_to_1(self):
        self.assertAlmostEqual(uc.convert_from_10_19_to_1(2.0), 2.0e19,
                               msg='Density given in 10^19 units is expected to convert to SI by *1e19.')

    def test_convert_keV_to_eV(self):
        self.assertAlmostEqual(uc.convert_keV_to_eV(1.5), 1500.0,
                               msg='1.5 keV is expected to convert to 1500 eV.')

    def test_calculate_velocity_from_energy(self):
        energy = 1.0
        mass = c.electron_mass
        expected = (2.0 * energy * c.elementary_charge / mass) ** 0.5
        actual = uc.calculate_velocity_from_energy(energy, mass)
        self.assertAlmostEqual(actual, expected, places=9,
                               msg='Velocity from energy is expected to equal sqrt(2 E e / m) using '
                                   'scipy elementary_charge and electron_mass.')

    def test_distance_cartesian_2d(self):
        self.assertAlmostEqual(uc.distance([0.0, 0.0], [3.0, 4.0]), 5.0,
                               msg='Cartesian distance of (0,0) and (3,4) is expected to be 5.')

    def test_distance_cartesian_3d(self):
        self.assertAlmostEqual(uc.distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0,
                               msg='Distance between identical 3D points is expected to be 0.')
        self.assertAlmostEqual(uc.distance([0.0, 0.0, 0.0], [1.0, 2.0, 2.0]), 3.0,
                               msg='Cartesian distance of (0,0,0) and (1,2,2) is expected to be 3.')

    def test_distance_dimension_mismatch(self):
        with self.assertRaises(AssertionError,
                               msg='distance is expected to reject points of unequal dimension.'):
            uc.distance([0.0, 0.0], [1.0, 2.0, 3.0])

    def test_distance_unsupported_system(self):
        with self.assertRaises(AssertionError,
                               msg='distance is expected to reject coordinate systems other than cartesian.'):
            uc.distance([0.0, 0.0], [1.0, 0.0], system='cylindrical')

    def test_distance_system_type(self):
        with self.assertRaises(AssertionError,
                               msg='distance is expected to reject a non-string coordinate system.'):
            uc.distance([0.0, 0.0], [1.0, 0.0], system=1)

    def test_unit_vector_cartesian(self):
        actual = uc.unit_vector([3.0, 4.0, 0.0], [0.0, 0.0, 0.0])
        numpy.testing.assert_allclose(actual, numpy.asarray([0.6, 0.8, 0.0]),
                                      err_msg='Unit vector of (3,4,0) relative to origin is expected to be (0.6, 0.8, 0).')
        self.assertAlmostEqual(numpy.linalg.norm(actual), 1.0,
                               msg='Returned cartesian unit vector is expected to have length 1.')

    def test_cartesian_to_cylin_axes(self):
        numpy.testing.assert_allclose(uc.cartesian_to_cylin([1.0, 0.0, 0.0]),
                                      numpy.asarray([1.0, 0.0, 0.0]),
                                      err_msg='Cartesian (1,0,0) is expected to map to cylindrical (r=1, z=0, phi=0).')
        numpy.testing.assert_allclose(uc.cartesian_to_cylin([0.0, 1.0, 2.0]),
                                      numpy.asarray([1.0, 2.0, math.pi / 2.0]),
                                      err_msg='Cartesian (0,1,2) is expected to map to cylindrical (r=1, z=2, phi=pi/2).')
        numpy.testing.assert_allclose(uc.cartesian_to_cylin([-1.0, 0.0, 0.0]),
                                      numpy.asarray([1.0, 0.0, math.pi]),
                                      err_msg='Cartesian (-1,0,0) is expected to map to cylindrical (r=1, z=0, phi=pi).')

    def test_cartesian_to_cylin_requires_3d(self):
        with self.assertRaises(AssertionError,
                               msg='cartesian_to_cylin is expected to require a 3-component point.'):
            uc.cartesian_to_cylin([1.0, 0.0])

    def test_cylin_to_cartesian_axes(self):
        numpy.testing.assert_allclose(uc.cylin_to_cartesian([1.0, 0.0, 0.0]),
                                      numpy.asarray([1.0, 0.0, 0.0]),
                                      atol=1e-12,
                                      err_msg='Cylindrical (r=1, z=0, phi=0) is expected to map to cartesian (1,0,0).')
        numpy.testing.assert_allclose(uc.cylin_to_cartesian([1.0, 2.0, math.pi / 2.0]),
                                      numpy.asarray([0.0, 1.0, 2.0]),
                                      atol=1e-12,
                                      err_msg='Cylindrical (r=1, z=2, phi=pi/2) is expected to map to cartesian (0,1,2).')

    def test_cylin_to_cartesian_requires_3d(self):
        with self.assertRaises(AssertionError,
                               msg='cylin_to_cartesian is expected to require a 3-component point.'):
            uc.cylin_to_cartesian([1.0, 0.0])

    def test_cartesian_cylindrical_roundtrip(self):
        original = numpy.asarray([1.5, -2.0, 3.0])
        recovered = uc.cylin_to_cartesian(uc.cartesian_to_cylin(original))
        numpy.testing.assert_allclose(recovered, original, atol=1e-12,
                                      err_msg='cartesian_to_cylin followed by cylin_to_cartesian is expected '
                                              'to recover the original point.')


if __name__ == '__main__':
    unittest.main()
