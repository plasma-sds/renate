import unittest
from utility.input import AtomicInput
from utility.exceptions import RenateNotValidTransitionError
from atomic.atomic_db import NeutralDB


class NeutralDBTest(unittest.TestCase):

    INPUT_ENERGY = 50
    INPUT_PROJECTILE = 'H'
    INPUT_CURRENT = 0.001
    INPUT_SOURCE = 'Internal'
    INPUT_PARAM_NAME = 'NeutralDB test'
    INPUT_H = [0, 1, 1, None]
    INPUT_H2 = [0, 1, 1, 'H2']
    INPUT_TEST = 'test'

    EXPECTED_KEY_2 = 2
    EXPECTED_KEY_1 = 1
    EXPECTED_KEY_NAME_1 = 'neutral1'
    EXPECTED_KEY_NAME_2 = ['neutral1', 'neutral2']
    EXPECTED_TEST_ATOMIC_LEVELS = 4

    INPUT_FROM_LEVELS = 0
    INPUT_TO_LEVELS = [1, 2, 3]

    EXPECTED_LOSS = [0.0001, 0.0002, 0.0003, 0.0004]
    EXPECTED_TRANS = [0.0012, 0.0013, 0.0014]

    def setUp(self):
        self.input = AtomicInput(energy=self.INPUT_ENERGY, projectile=self.INPUT_PROJECTILE, current=self.INPUT_CURRENT,
                                 param_name=self.INPUT_PARAM_NAME, source=self.INPUT_SOURCE)

    def tearDown(self):
        del self.input

    def test_neutral_atom_addition(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components)
        self.assertEqual(actual_db.neutral_target_count, self.EXPECTED_KEY_1,
                         msg='Only a single neutral component was expected.')
        self.assertEqual(list(actual_db.neutral_cross_sections.keys())[0], self.EXPECTED_KEY_NAME_1,
                         msg='Expected <neutral1> for single neutral component addition to the Neutral DB.')

    def test_neutral_molecule_addition(self):
        self.input.add_target_component(charge=self.INPUT_H2[0], atomic_number=self.INPUT_H2[1],
                                        mass_number=self.INPUT_H2[2], molecule_name=self.INPUT_H2[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components)
        self.assertEqual(actual_db.neutral_target_count, self.EXPECTED_KEY_1,
                         msg='Only a single neutral component was expected.')
        self.assertEqual(list(actual_db.neutral_cross_sections.keys())[0], self.EXPECTED_KEY_NAME_1,
                         msg='Expected <neutral1> for single neutral component addition to the Neutral DB.')

    def test_multiple_neutral_addition(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        self.input.add_target_component(charge=self.INPUT_H2[0], atomic_number=self.INPUT_H2[1],
                                        mass_number=self.INPUT_H2[2], molecule_name=self.INPUT_H2[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components)
        self.assertEqual(actual_db.neutral_target_count, self.EXPECTED_KEY_2,
                         msg='Only a single neutral component was expected.')
        self.assertListEqual(list(actual_db.neutral_cross_sections.keys()), self.EXPECTED_KEY_NAME_2,
                             msg='Expected <neutral1> for single neutral component addition to the Neutral DB.')

    def test_atomic_levels(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components, resolved=self.INPUT_TEST)
        self.assertEqual(actual_db.atomic_levels, self.EXPECTED_TEST_ATOMIC_LEVELS, msg='The actual atomic level '
                         'number does not match with the expected atomic level number.')

    def test_electron_loss_getter(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components, resolved=self.INPUT_TEST)
        actual_loss = []
        for index in range(actual_db.atomic_levels):
            actual_loss.append(actual_db.get_neutral_impact_loss(target=self.EXPECTED_KEY_NAME_1, from_level=index))
        self.assertListEqual(self.EXPECTED_LOSS, actual_loss, msg='Expected electron loss cross-sections do '
                             'not match to actual electron loss cross-sections.')

    def test_transition_loss_getter(self):
        self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                        mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
        actual_param, actual_components = self.input.get_atomic_db_input()
        actual_db = NeutralDB(param=actual_param, components=actual_components, resolved=self.INPUT_TEST)
        actual_transition = []
        for to_level in self.INPUT_TO_LEVELS:
            actual_transition.append(actual_db.get_neutral_impact_transition(target=self.EXPECTED_KEY_NAME_1,
                                                                             from_level=self.INPUT_FROM_LEVELS,
                                                                             to_level=to_level))
        self.assertListEqual(actual_transition, self.EXPECTED_TRANS, msg='Expected electron transition cross-sections '
                             'do not match to actual electron loss cross-sections.')

    def test_transition_error(self):
        with self.assertRaises(RenateNotValidTransitionError, msg='Expected error was not raised.'):
            self.input.add_target_component(charge=self.INPUT_H[0], atomic_number=self.INPUT_H[1],
                                            mass_number=self.INPUT_H[2], molecule_name=self.INPUT_H[3])
            actual_param, actual_components = self.input.get_atomic_db_input()
            actual_db = NeutralDB(param=actual_param, components=actual_components, resolved=self.INPUT_TEST)
            val = actual_db.get_neutral_impact_transition(target=self.EXPECTED_KEY_NAME_1,
                                                          from_level=self.INPUT_FROM_LEVELS,
                                                          to_level=self.INPUT_FROM_LEVELS)
