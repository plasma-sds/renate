from crm_solver.beamlet import Beamlet
import unittest
from lxml import etree
from atomic.atomic_db import AtomicDB
import pandas
import numpy


class BeamletTest(unittest.TestCase):
    EXPECTED_ATTR = ['param', 'components', 'profiles', 'coefficient_matrix', 'atomic_db', 'initial_condition']
    EXPECTED_INITIAL_CONDITION = [4832583046.75342, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    EXPECTED_PARAM_ATTR = ['beamlet_source', 'beamlet_energy', 'beamlet_species', 'beamlet_current']
    EXPECTED_COMPONENTS_KEYS = ['q', 'Z', 'A']
    EXPECTED_COMPONENTS_SPECIES = ['electron', 'ion1', 'ion2']
    EXPECTED_PROFILES_LENGTH = 101
    EXPECTED_PROFILES_KEYS = [('beamlet grid', 'distance', 'm'),
                              ('electron', 'density', 'm-3'),
                              ('electron', 'temperature', 'eV'),
                              ('ion1', 'density', 'm-3'),
                              ('ion1', 'temperature', 'eV'),
                              ('ion2', 'density', 'm-3'),
                              ('ion2', 'temperature', 'eV')]
    EXPECTED_ATTENUATION_KEY = 'linear_density_attenuation'
    INPUT_TRANSITION = ['2s', '2p', '5s', '5p']
    EXPECTED_ELEMENTS_3 = 3

    def setUp(self):
        self.beamlet = Beamlet()

    def tearDown(self):
        del self.beamlet

    def test_attributes(self):
        for attr in self.EXPECTED_ATTR:
            assert hasattr(self.beamlet, attr)

    def test_initial_conditions(self):
        self.assertIsInstance(self.beamlet.initial_condition, list, msg='Initial condition is of wrong type. '
                                                                        'Expected type: list')
        self.assertEqual(len(self.beamlet.initial_condition), self.beamlet.atomic_db.atomic_ceiling,
                         msg='Initial conditions must match number of atomic levels.')
        for element in range(len(self.beamlet.initial_condition)):
            self.assertIsInstance(self.beamlet.initial_condition[element], float, msg='Expected type for initial'
                                                                                      ' conditions is float.')
            self.assertAlmostEqual(self.beamlet.initial_condition[element], self.EXPECTED_INITIAL_CONDITION[element],
                             delta=1e-1, msg='Computed Init conditions do not match expected init conditions.')

    def test_param_xml(self):
        self.assertIsInstance(self.beamlet.param, etree._ElementTree,
                              msg='Expected type for param input is xml elementtree.')
        for param_attribute in self.EXPECTED_PARAM_ATTR:
            self.assertIsInstance(self.beamlet.param.getroot().find('body').find(param_attribute).text, str,
                                  msg='Failed to load or find attribute '+param_attribute+' in xml file.')

    def test_atomic_db(self):
        self.assertIsInstance(self.beamlet.atomic_db, AtomicDB,
                              msg='Expected data type for the Atomic database is AtomicDB.')

    def test_components(self):
        self.assertIsInstance(self.beamlet.components, pandas.core.frame.DataFrame,
                              msg='Expected data type of components is: pandas DataFrame.')
        description = self.beamlet.components.keys()
        species = self.beamlet.components.T.keys()
        self.assertEqual(len(description), self.EXPECTED_ELEMENTS_3,
                         msg='Three keys are expected for species description: q,Z,A.')
        for key in range(len(description)):
            self.assertEqual(description[key], self.EXPECTED_COMPONENTS_KEYS[key], msg='The index: ' +
                             description[key] + ' is a mismatch for '+self.EXPECTED_COMPONENTS_KEYS[key])
        self.assertEqual(len(species), len(self.beamlet.atomic_db.ion_impact_loss[0])+1,
                         msg='The same amount of species description have to be present in components as in profiles.')
        for key in range(len(species)):
            self.assertEqual(species[key], self.EXPECTED_COMPONENTS_SPECIES[key],
                             msg='Mismatch of expected plasma species.')
        for component in species:
            for coordinate in description:
                self.assertIsInstance(self.beamlet.components[coordinate][component], numpy.int64,
                                      msg='Type mismatch of pandas components content.')

    def test_profiles(self):
        actual = Beamlet(solver='disregard')
        self.assertIsInstance(actual.profiles, pandas.core.frame.DataFrame,
                              msg='Expected data type of profiles is: pandas DataFrame.')
        self.assertEqual(len(actual.profiles), self.EXPECTED_PROFILES_LENGTH,
                         msg='Expected profile length for test case does not match.')
        self.assertTupleEqual(actual.profiles.shape, (self.EXPECTED_PROFILES_LENGTH,
                              2*len(self.EXPECTED_COMPONENTS_KEYS)+1), msg='Plasma description lacks necessary'
                              ' density and/or temperature profiles for all components.')
        self.assertIsInstance(actual.profiles.axes[0], pandas.Int64Index,
                              msg='Expected data type of X - axis for profiles is Int64Index.')
        self.assertIsInstance(actual.profiles.axes[1], pandas.MultiIndex,
                              msg='Expected data type of Y - axis for profiles is MultiIndex.')
        for key in range(len(actual.profiles.keys())):
            self.assertTupleEqual(actual.profiles.keys()[key], self.EXPECTED_PROFILES_KEYS[key],
                                  msg='Profiles key description fails for test case.')

    def test_analytical_solver(self):
        with self.assertRaises(NotImplementedError):
            actual = Beamlet(solver='analytical')

    def test_numerical_solver(self):
        self.assertTupleEqual(self.beamlet.profiles.shape, (self.EXPECTED_PROFILES_LENGTH,
                              2*len(self.EXPECTED_COMPONENTS_KEYS)+1+self.beamlet.atomic_db.atomic_ceiling),
                              msg='Numerical solver failed to provide expected output into plasma profiles.')
        for key_index in range(2*len(self.EXPECTED_COMPONENTS_KEYS)+1,
                               len(self.beamlet.profiles.keys())):
            self.assertEqual(self.beamlet.profiles.keys()[key_index][0], 'level '+self.beamlet.atomic_db.inv_atomic_dict
                             [key_index - 2*len(self.EXPECTED_COMPONENTS_KEYS)-1],
                             msg='Pandas keys for labeling electron population evolution on atomic levels fails.')
        for level in range(self.beamlet.atomic_db.atomic_ceiling):
            self.assertIsInstance(self.beamlet.profiles['level '+self.beamlet.atomic_db.inv_atomic_dict[level]],
                                  pandas.core.series.Series, msg='Expected data type of beam evolution '
                                                                 'process are pandas series.')
            self.assertEqual(len(self.beamlet.profiles['level '+self.beamlet.atomic_db.inv_atomic_dict[level]]),
                             self.EXPECTED_PROFILES_LENGTH, msg='Beam evolution calculation are expected to return '
                                                                'atomic state evolution on the input grid.')

    def test_not_supported_solver(self):
        with self.assertRaises(Exception):
            actual = Beamlet(solver='not-supported')

    def test_attenuation_calculator(self):
        self.beamlet.compute_linear_density_attenuation()
        self.assertEqual(self.beamlet.profiles.keys()[-1][0], self.EXPECTED_ATTENUATION_KEY,
                         msg='Beam attenuation key mismatch within pandas data frame.')
        self.assertIsInstance(self.beamlet.profiles[self.EXPECTED_ATTENUATION_KEY], pandas.core.series.Series,
                              msg='Beam attenuation output expected to be stored in pandas series.')
        self.assertEqual(len(self.beamlet.profiles[self.EXPECTED_ATTENUATION_KEY]), self.EXPECTED_PROFILES_LENGTH,
                         msg='Beam attenuation calculation is expected to be returned on the input grid.')
        test = self.beamlet.profiles['level 2s']
        for level in range(1, self.beamlet.atomic_db.atomic_ceiling):
            test += self.beamlet.profiles['level ' + self.beamlet.atomic_db.inv_atomic_dict[level]]
        for index in range(self.EXPECTED_PROFILES_LENGTH):
            self.assertEqual(test[index], self.beamlet.profiles[self.EXPECTED_ATTENUATION_KEY][index],
                             msg='Beam attenuation calculation fails for test case.')

    def test_spontaneous_emission_fail(self):
        with self.assertRaises(Exception):
            self.beamlet.profiles[self.EXPECTED_ATTENUATION_KEY].\
                compute_linear_emission_density(to_level=self.INPUT_TRANSITION[1], from_level=self.INPUT_TRANSITION[0])

    def test_not_supported_atomic_level_fail(self):
        with self.assertRaises(Exception):
            self.beamlet.profiles[self.EXPECTED_ATTENUATION_KEY].\
                compute_linear_emission_density(to_level=self.INPUT_TRANSITION[2], from_level=self.INPUT_TRANSITION[3])

    def test_emission_calculator(self):
        self.beamlet.compute_linear_emission_density(to_level=self.INPUT_TRANSITION[0],
                                                     from_level=self.INPUT_TRANSITION[1])
        self.assertEqual(self.beamlet.profiles.keys()[-1][0],
                         self.INPUT_TRANSITION[1] + '-->' + self.INPUT_TRANSITION[0],
                         msg='Beam emission calculation key mismatch within pandas data frame.')
        self.assertIsInstance(self.beamlet.profiles[self.INPUT_TRANSITION[1] + '-->' + self.INPUT_TRANSITION[0]],
                              pandas.core.series.Series, msg='Beam emission output expected to '
                                                             'be stored in pandas series.')
        self.assertEqual(len(self.beamlet.profiles[self.INPUT_TRANSITION[1] + '-->' + self.INPUT_TRANSITION[0]]),
                         self.EXPECTED_PROFILES_LENGTH, msg='Beam emission calculation is expected to be returned '
                                                            'on the input grid.')
        test = self.beamlet.profiles['level '+self.INPUT_TRANSITION[1]] * self.beamlet.atomic_db.spontaneous_trans[
            self.beamlet.atomic_db.atomic_dict[self.INPUT_TRANSITION[0]],
            self.beamlet.atomic_db.atomic_dict[self.INPUT_TRANSITION[1]]]
        for index in range(len(test)):
            self.assertEqual(test[index], self.beamlet.profiles[self.INPUT_TRANSITION[1]
                             + '-->' + self.INPUT_TRANSITION[0]][index],
                             msg='Beam emission calculation fails for test case.')

    def test_relative_population_calculator(self):
        self.beamlet.compute_relative_populations(reference_level=self.INPUT_TRANSITION[0])
        self.assertTupleEqual(self.beamlet.profiles.filter(like='rel.pop').shape,
                              (self.EXPECTED_PROFILES_LENGTH, self.beamlet.atomic_db.atomic_ceiling),
                              msg='Relative populations calculated do not match input level number.')
        for level in range(self.beamlet.atomic_db.atomic_ceiling):
            for index in range(len(self.beamlet.profiles)):
                if self.beamlet.atomic_db.inv_atomic_dict[level] == self.INPUT_TRANSITION[0]:
                    self.assertEqual(self.beamlet.profiles['rel.pop ' +
                                                           self.beamlet.atomic_db.inv_atomic_dict[level]][index], 1.0,
                                     msg='Values on reference level are expected to be 1.')
                else:
                    self.assertLess(self.beamlet.profiles['rel.pop ' +
                                                          self.beamlet.atomic_db.inv_atomic_dict[level]][index], 1.0,
                                    msg='Values on comparative levels are expected to be less than 1.')

    def test_beamlet_pandas_copy(self):
        actual = self.beamlet.copy(object_copy='full')
        self.assertTupleEqual(actual.components.shape, self.beamlet.components.shape,
                              msg='Actual and copy Beamlet object components are expected to have same shape.')
        logic_components = actual.components == self.beamlet.components
        self.assertTrue(logic_components.values.all(), msg='Content of actual and reference Beamlet components '
                                                           'objects is required to be equal.')
        self.assertTupleEqual(actual.profiles.shape, self.beamlet.profiles.shape,
                              msg='Actual and copy Beamlet object profiles are expected to have the same shape.')
        logic_profiles = actual.profiles == self.beamlet.profiles
        self.assertTrue(logic_profiles.values.all(), msg='Content of actual and reference Beamlet profiles '
                                                         'objects is required to be equal.')

    def test_beamlet_pandas_copy_without_results(self):
        actual = self.beamlet.copy(object_copy='without-results')
        self.assertTupleEqual(actual.components.shape, self.beamlet.components.shape,
                              msg='Copy and Actual Beamlet object components are expected to have same shape.')
        logic_components = actual.components == self.beamlet.components
        self.assertTrue(logic_components.values.all(), msg='Content of Copy and Actual Beamlet components '
                                                           'objects is required to be equal.')
        self.assertEqual(actual.profiles.shape[0], self.beamlet.profiles.shape[0],
                         msg='Copy and Actual of Beamlet profiles are expected to have the same number of elements.')
        self.assertEqual(actual.profiles.shape[1], self.beamlet.profiles.shape[1]-self.beamlet.atomic_db.atomic_ceiling,
                         msg='Copy of Actual Beamlet object profiles is expected to have nr of atomic levels: ' +
                             str(self.beamlet.atomic_db.atomic_ceiling) + ' less columns.')
        self.assertEqual(actual.profiles.filter(like='level').shape[1], 0,
                         msg='The copy without results is expected NOT to contain any columns labeled <level>.')
