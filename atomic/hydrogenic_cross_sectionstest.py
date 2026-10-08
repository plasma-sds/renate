import random
import unittest

import numpy
import scipy.interpolate

from atomic.hydrogenic_cross_sections import HydrogenicData
from utility.particle import Particle
from utility.transition import Transition


class HydrogenicDataTest(unittest.TestCase):

    RANDOM_SEED = 0
    N_SAMPLES = 12
    ENERGY_GRID = numpy.logspace(2.0, 5.0, 32)

    TARGETS = (
        Particle(label='e', charge=-1, atomic_number=0, mass_number=0),
        Particle(label='p', charge=1, atomic_number=1, mass_number=1),
        Particle(label='He', charge=2, atomic_number=2, mass_number=4),
        Particle(label='Li', charge=3, atomic_number=3, mass_number=7),
        Particle(label='Be', charge=4, atomic_number=4, mass_number=9),
        Particle(label='B', charge=5, atomic_number=5, mass_number=11),
        Particle(label='C', charge=6, atomic_number=6, mass_number=12),
        Particle(label='O', charge=8, atomic_number=8, mass_number=16),
        Particle(label='Ne', charge=10, atomic_number=10, mass_number=20),
    )

    def setUp(self):
        self.data = HydrogenicData()
        self.projectile = Particle(label='H', atomic_number=1, mass_number=1)
        self.rng = random.Random(self.RANDOM_SEED)

    def tearDown(self):
        del self.data
        del self.projectile
        del self.rng

    def _allowed_trans_types(self, target):
        types = ['ex', 'de-ex', 'ion', 'eloss']
        if target.charge != -1:
            types.append('cx')
        return types

    def _random_levels(self, trans_name):
        from_n = self.rng.randint(1, 3)
        to_n = self.rng.randint(from_n + 1, 6)
        if trans_name in ['ex']:
            return str(from_n), str(to_n)
        if trans_name == 'de-ex':
            return str(to_n), str(from_n)
        return str(from_n), None

    def _make_transition(self, target, trans_name):
        from_level, to_level = self._random_levels(trans_name)
        return Transition(projectile=self.projectile, target=target,
                          from_level=from_level, to_level=to_level, trans=trans_name)

    def _interpolator_from_transition(self, transition):
        cross_section = self.data.get_cross_section(transition, self.ENERGY_GRID)
        return scipy.interpolate.interp1d(self.ENERGY_GRID, cross_section, fill_value='extrapolate')

    def test_random_targets_and_transitions_yield_callable_interpolators(self):
        for sample in range(self.N_SAMPLES):
            target = self.rng.choice(self.TARGETS)
            trans_name = self.rng.choice(self._allowed_trans_types(target))
            transition = self._make_transition(target, trans_name)
            interpolator = self._interpolator_from_transition(transition)
            self.assertIsInstance(interpolator, scipy.interpolate.interp1d,
                                  msg='HydrogenicData is expected to yield an interp1d interpolator for '
                                      'target ' + target.label + ' and trans ' + trans_name +
                                      ' (sample ' + str(sample) + ').')
            self.assertTrue(callable(interpolator),
                            msg='Generated interpolator is expected to be callable for target ' +
                                target.label + ' and trans ' + trans_name +
                                ' (sample ' + str(sample) + ').')
            sample_energy = self.ENERGY_GRID[len(self.ENERGY_GRID) // 2]
            value = interpolator(sample_energy)
            self.assertIsNotNone(value,
                                 msg='Calling the interpolator is expected to return a value for target ' +
                                     target.label + ' and trans ' + trans_name +
                                     ' (sample ' + str(sample) + ').')

    def test_spontaneous_trans_matrix_exists(self):
        self.assertTrue(hasattr(self.data, 'spontaneous_trans'),
                        msg='HydrogenicData is expected to provide a spontaneous_trans matrix.')
        self.assertIsInstance(self.data.spontaneous_trans, numpy.ndarray,
                              msg='spontaneous_trans is expected to be stored as a numpy ndarray.')
        self.assertEqual(self.data.spontaneous_trans.ndim, 2,
                         msg='spontaneous_trans is expected to be a 2D matrix.')
        n_rows, n_cols = self.data.spontaneous_trans.shape
        self.assertEqual(n_rows, n_cols,
                         msg='spontaneous_trans is expected to be a square matrix.')


if __name__ == '__main__':
    unittest.main()
