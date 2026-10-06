import unittest

from utility.exceptions import InputError
from utility.particle import Particle
from utility.transition import Transition


class TransitionTest(unittest.TestCase):

    def setUp(self):
        self.projectile = Particle(label='H', atomic_number=1, mass_number=1)
        self.target = Particle(label='e', charge=-1, atomic_number=0, mass_number=0)

    def tearDown(self):
        del self.projectile
        del self.target

    def _make(self, from_level='1n', to_level='3n', trans='ex'):
        return Transition(projectile=self.projectile, target=self.target,
                          from_level=from_level, to_level=to_level, trans=trans)

    def test_stores_projectile_and_target(self):
        transition = self._make()
        self.assertIs(transition.projectile, self.projectile,
                      msg='Transition.projectile is expected to be the provided Particle.')
        self.assertIs(transition.target, self.target,
                      msg='Transition.target is expected to be the provided Particle.')

    def test_supported_transition_names(self):
        for name in ['ex', 'de-ex', 'eloss', 'ion', 'cx']:
            transition = self._make(trans=name)
            self.assertEqual(transition.name, name,
                             msg='Transition.name is expected to equal the supported trans value: ' + name)

    def test_from_and_to_levels_are_stored(self):
        transition = self._make(from_level='1n', to_level='3n', trans='ex')
        self.assertEqual(transition.from_level, '1n',
                         msg='Transition.from_level is expected to equal the provided from_level.')
        self.assertEqual(transition.to_level, '3n',
                         msg='Transition.to_level is expected to equal the provided to_level.')

    def test_to_level_none_is_allowed(self):
        transition = self._make(from_level='1n', to_level=None, trans='ion')
        self.assertIsNone(transition.to_level,
                          msg='Transition.to_level is expected to remain None when None is provided.')

    def test_unsupported_trans_raises_input_error(self):
        with self.assertRaises(InputError,
                               msg='Transition is expected to raise InputError for an unsupported trans name.'):
            self._make(trans='not-a-transition')

    def test_non_string_to_level_raises_input_error(self):
        with self.assertRaises(InputError,
                               msg='Transition is expected to raise InputError when to_level is not str or None.'):
            self._make(to_level=3)

    def test_non_string_from_level_raises_input_error(self):
        with self.assertRaises(InputError,
                               msg='Transition is expected to raise InputError when from_level is not a string.'):
            self._make(from_level=4)

    def test_non_string_trans_raises_input_error(self):
        with self.assertRaises(InputError,
                               msg='Transition is expected to raise InputError when trans is not a string.'):
            self._make(trans=1)

    def test_default_arguments_raise_input_error(self):
        with self.assertRaises(InputError,
                               msg='Transition() with default from_level/trans type objects is expected to '
                                   'raise InputError.'):
            Transition()

    def test_str_for_excitation(self):
        transition = self._make(from_level='2s', to_level='2p', trans='ex')
        self.assertEqual(str(transition), '2s-2p',
                         msg='str of an excitation Transition is expected to be from_level-to_level.')

    def test_str_for_deexcitation(self):
        transition = self._make(from_level='2p', to_level='2s', trans='de-ex')
        self.assertEqual(str(transition), '2p-2s',
                         msg='str of a de-excitation Transition is expected to be from_level-to_level.')

    def test_str_for_cx_eloss_ion(self):
        self.assertEqual(str(self._make(from_level='2s', to_level=None, trans='cx')), '2s-cx',
                         msg='str of a charge-exchange Transition is expected to be from_level-cx.')
        self.assertEqual(str(self._make(from_level='2s', to_level=None, trans='eloss')), '2s-eloss',
                         msg='str of an electron-loss Transition is expected to be from_level-eloss.')
        self.assertEqual(str(self._make(from_level='2s', to_level=None, trans='ion')), '2s-ion',
                         msg='str of an ionization Transition is expected to be from_level-ion.')

    def test_repr_contains_collision_and_levels(self):
        transition = self._make(from_level='2s', to_level='2p', trans='ex')
        representation = repr(transition)
        self.assertIn('Collision of: H + e', representation,
                      msg='repr(Transition) is expected to include projectile and target labels.')
        self.assertIn('with transition: ex', representation,
                      msg='repr(Transition) is expected to include the transition name.')
        self.assertIn('from level: 2s', representation,
                      msg='repr(Transition) is expected to include from_level.')
        self.assertIn('to_level: 2p', representation,
                      msg='repr(Transition) is expected to include to_level.')


if __name__ == '__main__':
    unittest.main()
