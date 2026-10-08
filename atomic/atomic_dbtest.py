from atomic.atomic_db import AtomicDB
from atomic.atomic_db import RenateDB
from atomic.neutral_db import NeutralDB
from utility.input import AtomicInput
import unittest
import numpy
import scipy
import utility.convert as uc


class RenateDBTest(unittest.TestCase):
    EXPECTED_ATTR = ['energy', 'param', 'species', 'mass', 'atomic_dict', 'rate_type', 'velocity',
                     'atomic_levels', 'inv_atomic_dict', 'impurity_mass_normalization', 'charged_states']
    EXPECTED_ATOM = ['dummy', 'Li', 'Na', 'T', 'H', 'D']
    EXPECTED_ATOMIC_LEVELS = [3, 9, 8, 6, 6, 6]
    EXPECTED_ENERGY = 60
    EXPECTED_VELOCITY = 1291547.19
    EXPECTED_MASS = 1.15258e-26
    EXPECTED_ATOMIC_DICT = [{'1': 1, '0': 0, '2': 2},
                            {'2s': 0, '2p': 1, '3s': 2, '3p': 3, '3d': 4, '4s': 5, '4p': 6, '4d': 7, '4f': 8},
                            {'3s': 0, '3p': 1, '3d': 2, '4s': 3, '4p': 4, '4d': 5, '4f': 6, '5s': 7},
                            {'1n': 0, '2n': 1, '3n': 2, '4n': 3, '5n': 4, '6n': 5},
                            {'1n': 0, '2n': 1, '3n': 2, '4n': 3, '5n': 4, '6n': 5},
                            {'1n': 0, '2n': 1, '3n': 2, '4n': 3, '5n': 4, '6n': 5}]
    EXPECTED_MASS_CORRECTION_DICT = {'charge-1': 1, 'charge-2': 4, 'charge-3': 7, 'charge-4': 9, 'charge-5': 11,
                                     'charge-6': 12, 'charge-7': 14, 'charge-8': 16, 'charge-9': 19, 'charge-10': 20,
                                     'charge-11': 23}
    EXPECTED_DEFAULT_ATOMIC_STATES = [['1', '0', '0', '1-->0'],
                                      ['2p', '2s', '2s', '2p-->2s'],
                                      ['3p', '3s', '3s', '3p-->3s'],
                                      ['3n', '2n', '1n', '3n-->2n'],
                                      ['3n', '2n', '1n', '3n-->2n'],
                                      ['3n', '2n', '1n', '3n-->2n']]
    EXPECTED_DECIMAL_PRECISION_2 = 2
    EXPECTED_DECIMAL_PRECISION_4 = 4
    EXPECTED_DIMENSION_1 = 1
    EXPECTED_DIMENSION_2 = 2
    EXPECTED_DIMENSION_3 = 3
    EXPECTED_DIMENSION_4 = 4
    INPUT_PATH = 'beamlet/testimp0001.xml'
    INPUT_DATA_GETTING = ['temperature', 'spontaneous_transition', 'ionization_terms',
                          'impurity_transition', 'ion_transition', 'electron_transition']

    def setUp(self):
        self.renate_db = RenateDB(None, 'default', self.INPUT_PATH)

    def tearDown(self):
        del self.renate_db

    def test_all_attributes(self):
        for attr in self.EXPECTED_ATTR:
            assert hasattr(self.renate_db, attr)

    def test_projectile_energy(self):
        self.assertIsInstance(self.renate_db.energy, str, msg='Energy input is expected to be in str format.')
        self.assertEqual(int(self.renate_db.energy), self.EXPECTED_ENERGY,
                         msg='Test energy value is expected to be convertable to int and 60.')

    def test_impurity_mass_correction_dictionary(self):
        self.assertIsInstance(self.renate_db.impurity_mass_normalization, dict,
                              msg='The impurity mass correction DB is expected to be of dict type.')
        self.assertDictEqual(self.renate_db.impurity_mass_normalization, self.EXPECTED_MASS_CORRECTION_DICT,
                             msg='The implemented mass correction dictionary fails.')

    def test_projectile_mass(self):
        self.assertIsInstance(self.renate_db.mass, float, msg='Projectile mass is expected to be of float type.')
        self.assertEqual(self.renate_db.mass, self.EXPECTED_MASS,
                         msg='Weight of projectile is expected to match Li for the testcase.')

    def test_projectile_velocity(self):
        self.assertIsInstance(self.renate_db.velocity, float,
                              msg='Projectile velocity is expected to be given as float.')
        self.assertAlmostEqual(self.renate_db.velocity, self.EXPECTED_VELOCITY, self.EXPECTED_DECIMAL_PRECISION_2,
                               msg='Projectile velocity is expected to match the velocity of Li @ 60 keV.')

    def test_atomic_species(self):
        self.assertIsInstance(self.renate_db.species, str, msg='Atomic species is expected to be given in type of str.')
        self.assertIn(self.renate_db.species, self.EXPECTED_ATOM,
                      msg='Projectile atomic species currently supported has to be among: H, D, T, Li and Na.')

    def test_atomic_levels(self):
        for index in range(len(self.EXPECTED_ATOM)):
            self.renate_db.param.getroot().find('body').find('beamlet_species').text = self.EXPECTED_ATOM[index]
            atom = RenateDB(self.renate_db.param, 'default', None)
            self.assertIsInstance(atom.atomic_levels, int,
                                  msg='Number of atomic levels is expected to be given in type int')
            self.assertEqual(atom.atomic_levels, self.EXPECTED_ATOMIC_LEVELS[index])
            self.assertIsInstance(atom.atomic_dict, dict,
                                  msg='Data structure labeling atomic levels is expected to be given in type dict.')
            self.assertDictEqual(atom.atomic_dict, self.EXPECTED_ATOMIC_DICT[index],
                                 msg='Atomic label dict for '+self.EXPECTED_ATOM[index]+' fails.')
            self.assertIsInstance(atom.inv_atomic_dict, dict,
                                  msg='Data structure linking inverting atomic labels is expected to be of type dict.')

    def test_default_atomic_levels(self):
        for index in range(len(self.EXPECTED_ATOM)):
            self.renate_db.param.getroot().find('body').find('beamlet_species').text = self.EXPECTED_ATOM[index]
            atom = RenateDB(self.renate_db.param, 'default', None)
            fr, to, ground, trans = atom.set_default_atomic_levels()
            self.assertIsInstance(fr, str, msg='From level label is expected to be of type str.')
            self.assertEqual(fr, self.EXPECTED_DEFAULT_ATOMIC_STATES[index][0],
                             msg='Returned <from> level label fails for projectile type: ' + self.EXPECTED_ATOM[index])
            self.assertIsInstance(to, str, msg='To level label is expected to be of type str.')
            self.assertEqual(to, self.EXPECTED_DEFAULT_ATOMIC_STATES[index][1],
                             msg='Returned <to> level label fails for projectile type: ' + self.EXPECTED_ATOM[index])
            self.assertIsInstance(ground, str, msg='Ground level label is expected to be of type str.')
            self.assertEqual(ground, self.EXPECTED_DEFAULT_ATOMIC_STATES[index][2],
                             msg='Returned <ground> level label fails for projectile type: '
                                 + self.EXPECTED_ATOM[index])
            self.assertIsInstance(trans, str, msg='Transition label is expected to be of type str.')
            self.assertEqual(trans, self.EXPECTED_DEFAULT_ATOMIC_STATES[index][3],
                             msg='Returned <transition> label fails for projectile type: ' + self.EXPECTED_ATOM[index])

    def test_atomic_data_getter(self):
        for index in range(len(self.INPUT_DATA_GETTING)):
            data = self.renate_db.get_from_renate_atomic(self.INPUT_DATA_GETTING[index])
            self.assertIsInstance(data, numpy.ndarray, msg='Data type to be returned by getter function is'
                                                           ' expected to be of type numpy ndarray.')

    def test_charged_state_library(self):
        self.assertIsInstance(self.renate_db.charged_states, tuple, msg='Data structure containing availability for '
                              'rates in various charged states is of type tuple.')
        for state in range(len(self.renate_db.charged_states)):
            self.assertIsInstance(self.renate_db.charged_states[state], str, msg='Charge state label of plasma '
                                  'component supported for rate calculations is expected to be of type str.')
            self.assertEqual(self.renate_db.charged_states[state], 'charge-'+str(state+1),
                             msg='Data structures containing the maximum number of charged states supported fails.')

    def test_renate_temperature(self):
        data = self.renate_db.get_from_renate_atomic('temperature')
        self.assertEqual(data.ndim, self.EXPECTED_DIMENSION_1, msg='Expected dimension of temperature array is 1.')

    def test_renate_spontaneous(self):
        data = self.renate_db.get_from_renate_atomic('spontaneous_transition')
        self.assertEqual(data.ndim, self.EXPECTED_DIMENSION_2,
                         msg='Expected dimension for spontaneous transition data is 2.')
        self.assertTupleEqual(data.shape, (self.renate_db.atomic_levels, self.renate_db.atomic_levels),
                              msg='Data structure size is not in accordance with atomic physics specifications.')

    def test_renate_electron_transitions(self):
        temp = self.renate_db.get_from_renate_atomic('temperature')
        data = self.renate_db.get_from_renate_atomic('electron_transition')
        self.assertEqual(data.ndim, self.EXPECTED_DIMENSION_3,
                         msg='Expected dimension for electron transition data is 3.')
        self.assertTupleEqual(data.shape, (self.renate_db.atomic_levels, self.renate_db.atomic_levels, len(temp)),
                              msg='Data structure size is not in accordance with atomic physics specifications.')

    def test_renate_ion_transitions(self):
        temp = self.renate_db.get_from_renate_atomic('temperature')
        data = self.renate_db.get_from_renate_atomic('ion_transition')
        self.assertEqual(data.ndim, self.EXPECTED_DIMENSION_3, msg='Expected dimension for ion transition data is 3.')
        self.assertTupleEqual(data.shape, (self.renate_db.atomic_levels, self.renate_db.atomic_levels, len(temp)),
                              msg='Data structure size is not in accordance with atomic physics specifications.')

    def test_renate_impurity_transitions(self):
        temp = self.renate_db.get_from_renate_atomic('temperature')
        data = self.renate_db.get_from_renate_atomic('impurity_transition')
        self.assertEqual(data.ndim, self.EXPECTED_DIMENSION_4,
                         msg='Expected dimension for impurity transition data is 4.')
        self.assertTupleEqual(data.shape, (len(self.renate_db.charged_states)-1, self.renate_db.atomic_levels,
                              self.renate_db.atomic_levels, len(temp)), msg='Data structure size is not in accordance'
                                                                            ' with atomic physics specifications.')

    def test_renate_ionization_terms(self):
        temp = self.renate_db.get_from_renate_atomic('temperature')
        data = self.renate_db.get_from_renate_atomic('ionization_terms')
        self.assertEqual(data.ndim, self.EXPECTED_DIMENSION_3, msg='Expected dimension for ionization data is 3.')
        self.assertTupleEqual(data.shape, (len(self.renate_db.charged_states)+1,
                              self.renate_db.atomic_levels, len(temp)), msg='Data structure size is not in accordance '
                                                                            'with atomic physics specifications.')


class AtomicDBTest(unittest.TestCase):
    EXPECTED_ATTR = ['temperature_axis', 'spontaneous_trans', 'electron_impact_loss',
                     'ion_impact_loss', 'electron_impact_trans', 'ion_impact_trans']
    INPUT_q = [1, 1, 1, 2, 4]
    INPUT_z = [1, 1, 2, 2, 4]
    INPUT_a = [1, 2, 3, 4, 9]
    INPUT_m = [None, None, None, None, None]
    INPUT_neutral_q = [-1, 0, 1, 1]
    INPUT_neutral_z = [0, 1, 1, 1]
    INPUT_neutral_a = [0, 1, 1, 2]
    INPUT_neutral_m = [None, None, None, None]
    INTERPOLATION_TEST_TEMPERATURE = [0, 1, 2, 2.5, 3, 8, 10]
    EXPECTED_DECIMAL_PRECISION_5 = 5
    EXPECTED_ELECTRON_IMPACT_LOSS = uc.convert_from_cm2_to_m2(numpy.asarray(
                                                              [[11., 111., 211., 261., 311., 811., 1011.],
                                                               [21., 121., 221., 271., 321., 821., 1021.],
                                                               [31., 131., 231., 281., 331., 831., 1031.]]))
    EXPECTED_ELECTRON_IMPACT_TRANS = uc.convert_from_cm2_to_m2(numpy.asarray(
                                                               [[[0., 0., 0., 0., 0., 0., 0.],
                                                                 [12., 112., 212., 262., 312., 812., 1012.],
                                                                 [13., 113., 213., 263., 313., 813., 1013.]],
                                                                [[21., 121., 221., 271., 321., 821., 1021.],
                                                                 [0., 0., 0., 0., 0., 0., 0.],
                                                                 [23., 123., 223., 273., 323., 823., 1023.]],
                                                                [[31., 131., 231., 281., 331., 831., 1031.],
                                                                 [32., 132., 232., 282., 332., 832., 1032.],
                                                                 [0., 0., 0., 0., 0., 0., 0.]]]))
    EXPECTED_ION_IMPACT_LOSS = uc.convert_from_cm2_to_m2(numpy.asarray(
                                                         [[[12., 112., 212., 262., 312., 812., 1012.],
                                                           [12., 62., 112., 137., 162., 412., 512.],
                                                           [12., 45.333, 78.666, 95.333, 112., 278.666, 345.333],
                                                           [13., 113., 213., 263., 313., 813., 1013.],
                                                           [15., 115., 215., 265., 315., 815., 1015.]],
                                                          [[22., 122., 222., 272., 322., 822., 1022.],
                                                           [22., 72., 122., 147., 172., 422., 522.],
                                                           [22., 55.333, 88.666, 105.333, 122., 288.666, 355.333],
                                                           [23., 123., 223., 273., 323., 823., 1023.],
                                                           [25., 125., 225., 275., 325., 825., 1025.]],
                                                          [[32., 132., 232., 282., 332., 832., 1032.],
                                                           [32., 82., 132., 157., 182., 432., 532.],
                                                           [32., 65.333, 98.666, 115.333, 132., 298.666, 365.333],
                                                           [33., 133., 233., 283., 333., 833., 1033.],
                                                           [35., 135., 235., 285., 335., 835., 1035.]]]))
    EXPECTED_ION_IMPACT_TRANS = uc.convert_from_cm2_to_m2(numpy.asarray(
                                                          [[[[0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.]],
                                                            [[12., 112., 212., 262., 312., 812., 1012.],
                                                             [12., 62., 112., 137., 162., 412., 512.],
                                                             [12., 45.333, 78.666, 95.333, 112., 278.666, 345.333],
                                                             [121., 1121., 2121., 2621., 3121., 8121., 10121.],
                                                             [123., 1123., 2123., 2623., 3123., 8123., 10123.]],
                                                            [[13., 113., 213., 263., 313., 813., 1013.],
                                                             [13., 63., 113., 138., 163., 413., 513.],
                                                             [13., 46.333, 79.666, 96.333, 113., 279.666, 346.333],
                                                             [131., 1131., 2131., 2631., 3131., 8131., 10131.],
                                                             [133., 1133., 2133., 2633., 3133., 8133., 10133.]]],
                                                           [[[21., 121., 221., 271., 321., 821., 1021.],
                                                             [21., 71., 121., 146., 171., 421., 521.],
                                                             [21., 54.333, 87.666, 104.333, 121., 287.666, 354.333],
                                                             [211., 1211., 2211., 2711., 3211., 8211., 10211.],
                                                             [213., 1213., 2213., 2713., 3213., 8213., 10213.]],
                                                            [[0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.]],
                                                            [[23., 123., 223., 273., 323., 823., 1023.],
                                                             [23., 73., 123., 148., 173., 423., 523.],
                                                             [23., 56.333, 89.666, 106.333, 123., 289.666, 356.333],
                                                             [231., 1231., 2231., 2731., 3231., 8231., 10231.],
                                                             [233., 1233., 2233., 2733., 3233., 8233., 10233.]]],
                                                           [[[31., 131., 231., 281., 331., 831., 1031.],
                                                             [31., 81., 131., 156., 181., 431., 531.],
                                                             [31., 64.333, 97.666, 114.333, 131., 297.666, 364.333],
                                                             [311., 1311., 2311., 2811., 3311., 8311., 10311.],
                                                             [313., 1313., 2313., 2813., 3313., 8313., 10313.]],
                                                            [[32., 132., 232., 282., 332., 832., 1032.],
                                                             [32., 82., 132., 157., 182., 432., 532.],
                                                             [32., 65.333, 98.666, 115.333, 132., 298.666, 365.333],
                                                             [321., 1321., 2321., 2821., 3321., 8321., 10321.],
                                                             [323., 1323., 2323., 2823., 3323., 8323., 10323.]],
                                                            [[0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.],
                                                             [0., 0., 0., 0., 0., 0., 0.]]]]))

    def setUp(self):
        param, components = self.build_atomic_input()
        self.atomic_db = AtomicDB(param=param, components=components)

    def tearDown(self):
        del self.atomic_db

    def test_all_attributes(self):
        for attr in self.EXPECTED_ATTR:
            assert hasattr(self.atomic_db, attr)

    def test_spontaneous_trans(self):
        self.assertIsInstance(self.atomic_db.spontaneous_trans, numpy.ndarray,
                              msg='Spontaneous transition data stored in wrong format.')
        self.assertEqual(self.atomic_db.spontaneous_trans.ndim, 2,
                         msg='Dimensions of spontaneous transition data is expected to be 2D.')
        self.assertEqual(self.atomic_db.atomic_ceiling, int(self.atomic_db.spontaneous_trans.size ** 0.5),
                         msg='Number of elements in spontaneous transition array is '
                             'inconsistent with number of atomic levels.')
        for to_level in range(self.atomic_db.atomic_ceiling):
            for from_level in range(self.atomic_db.atomic_ceiling):
                if from_level <= to_level:
                    self.assertEqual(self.atomic_db.spontaneous_trans[to_level, from_level],
                                     0.0, msg='Spontaneous transition levels set wrong!!')

    def test_electron_impact_loss_terms(self):
        self.assertIsInstance(self.atomic_db.electron_impact_loss, tuple,
                              msg='Electron impact loss functions are stored in wrong data format.')
        self.assertEqual(len(self.atomic_db.electron_impact_loss), self.atomic_db.atomic_ceiling,
                         msg='Number of expected interpolator functions is inconsistent with number of atomic levels.')
        for index in range(self.atomic_db.atomic_ceiling):
            self.assertIsInstance(self.atomic_db.electron_impact_loss[index], scipy.interpolate.interp1d,
                                  msg='Provided electron impact loss functions are of wrong type.')

    def test_electron_impact_transition_terms(self):
        self.assertIsInstance(self.atomic_db.electron_impact_trans, tuple,
                              msg='Electron impact transition functions are stored in wrong data format.')
        self.assertEqual(len(self.atomic_db.electron_impact_trans), self.atomic_db.atomic_ceiling,
                         msg='Number of expected interpolator functions is inconsistent with number of atomic levels.')
        for from_level in range(self.atomic_db.atomic_ceiling):
            self.assertIsInstance(self.atomic_db.electron_impact_trans[from_level], tuple,
                                  msg='Electron impact transition functions are stored in wrong data format.')
            self.assertEqual(len(self.atomic_db.electron_impact_trans[from_level]), self.atomic_db.atomic_ceiling,
                             msg='Number of expected interpolator functions is '
                                 'inconsistent with number of atomic levels.')
            for to_level in range(self.atomic_db.atomic_ceiling):
                self.assertIsInstance(self.atomic_db.electron_impact_trans[from_level][to_level],
                                      scipy.interpolate.interp1d, msg='Provided electron impact transition '
                                                                      'functions are of wrong type.')

    def test_ion_impact_loss_terms(self):
        self.assertIsInstance(self.atomic_db.ion_impact_loss, tuple,
                              msg='Ion impact loss functions are stored in wrong data format.')
        self.assertEqual(len(self.atomic_db.ion_impact_loss), self.atomic_db.atomic_ceiling,
                         msg='Number of expected interpolator functions is inconsistent with number of atomic levels.')
        for from_level in range(self.atomic_db.atomic_ceiling):
            self.assertIsInstance(self.atomic_db.ion_impact_loss[from_level], tuple,
                                  msg='Ion impact loss functions are stored in wrong data format.')
            self.assertEqual(len(self.atomic_db.ion_impact_loss[from_level]), len(self.atomic_db.components.T.keys())-1,
                             msg='Number of expected interpolator functions is '
                                 'inconsistent with the number of plasma ions.')
            for charge in range(len(self.atomic_db.components.T.keys())-1):
                self.assertIsInstance(self.atomic_db.ion_impact_loss[from_level][charge], scipy.interpolate.interp1d,
                                      msg='Provided electron impact loss functions are of wrong type.')

    def test_ion_impact_transition_terms(self):
        self.assertIsInstance(self.atomic_db.ion_impact_trans, tuple, msg='Ion impact transition functions '
                                                                          'are stored in wrong data format.')
        self.assertEqual(len(self.atomic_db.ion_impact_trans), self.atomic_db.atomic_ceiling,
                         msg='Number of expected interpolator functions is inconsistent with number of atomic levels.')
        for from_level in range(self.atomic_db.atomic_ceiling):
            self.assertIsInstance(self.atomic_db.ion_impact_trans[from_level], tuple,
                                  msg='Ion impact transition functions are stored in wrong data format.')
            self.assertEqual(len(self.atomic_db.ion_impact_trans[from_level]), self.atomic_db.atomic_ceiling,
                             msg='Number of expected interpolator functions is '
                                 'inconsistent with number of atomic levels.')
            for to_level in range(self.atomic_db.atomic_ceiling):
                self.assertIsInstance(self.atomic_db.ion_impact_trans[from_level][to_level], tuple,
                                      msg='Ion impact transition functions are stored in wrong data format.')
                self.assertEqual(len(self.atomic_db.ion_impact_trans[from_level][to_level]),
                                 len(self.atomic_db.components.T.keys())-1,
                                 msg='Number of expected interpolator functions is '
                                 'inconsistent with the number of plasma ions.')
                for charge in range(len(self.atomic_db.components.T.keys())-1):
                    self.assertIsInstance(self.atomic_db.ion_impact_trans[from_level][to_level][charge],
                                          scipy.interpolate.interp1d, msg='Provided electron impact loss '
                                                                          'functions are of wrong type.')

    def test_electron_impact_loss_interpolator(self):
        for level in range(self.atomic_db.atomic_ceiling):
            rates = self.atomic_db.electron_impact_loss[level](self.INTERPOLATION_TEST_TEMPERATURE)
            self.assertIsInstance(rates, numpy.ndarray, msg='Interpolator output expected to be numpy.')
            numpy.testing.assert_almost_equal(self.EXPECTED_ELECTRON_IMPACT_LOSS[level], rates,
                                              self.EXPECTED_DECIMAL_PRECISION_5, err_msg='Electron impact loss '
                                                                                         'interpolator failure.')

    def test_electron_impact_transition_interpolator(self):
        for from_level in range(self.atomic_db.atomic_ceiling):
            for to_level in range(self.atomic_db.atomic_ceiling):
                rates = self.atomic_db.electron_impact_trans[from_level][to_level](self.INTERPOLATION_TEST_TEMPERATURE)
                self.assertIsInstance(rates, numpy.ndarray, msg='Interpolator output expected to be numpy.')
                numpy.testing.assert_almost_equal(self.EXPECTED_ELECTRON_IMPACT_TRANS[from_level][to_level], rates,
                                                  self.EXPECTED_DECIMAL_PRECISION_5,
                                                  err_msg='Electron impact transition interpolator failure.')

    def test_ion_impact_loss_interpolator(self):
        for level in range(self.atomic_db.atomic_ceiling):
            for target in range(len(self.atomic_db.components.T.keys())-1):
                rates = self.atomic_db.ion_impact_loss[level][target](self.INTERPOLATION_TEST_TEMPERATURE)
                self.assertIsInstance(rates, numpy.ndarray, msg='Interpolator output expected to be numpy.')
                numpy.testing.assert_almost_equal(self.EXPECTED_ION_IMPACT_LOSS[level][target], rates,
                                                  self.EXPECTED_DECIMAL_PRECISION_5,
                                                  err_msg='Ion impact loss interpolator failure.')

    def test_ion_impact_transition_interpolator(self):
        for from_level in range(self.atomic_db.atomic_ceiling):
            for to_level in range(self.atomic_db.atomic_ceiling):
                for target in range(len(self.atomic_db.components.T.keys())-1):
                    rates = self.atomic_db.ion_impact_trans[from_level][to_level][target](self.INTERPOLATION_TEST_TEMPERATURE)
                    self.assertIsInstance(rates, numpy.ndarray, msg='Interpolator output expected to be numpy.')
                    numpy.testing.assert_almost_equal(self.EXPECTED_ION_IMPACT_TRANS[from_level][to_level][target],
                                                      rates, self.EXPECTED_DECIMAL_PRECISION_5,
                                                      err_msg='Ion impact transition interpolator failure.')

    def test_ceiled_electron_impact_loss_terms(self):
        ceiled_db = AtomicDB(param=self.atomic_db.param, components=self.atomic_db.components, atomic_ceiling=2)
        self.assertIsInstance(ceiled_db.electron_impact_loss, tuple,
                              msg='Electron impact loss functions are stored in wrong data format.')
        self.assertEqual(len(ceiled_db.electron_impact_loss), ceiled_db.atomic_ceiling,
                         msg='Number of expected interpolator functions is inconsistent with number of atomic levels.')
        for index in range(ceiled_db.atomic_ceiling):
            self.assertIsInstance(ceiled_db.electron_impact_loss[index], scipy.interpolate.interp1d,
                                  msg='Provided electron impact loss functions are of wrong type.')

    def test_ceiled_electron_impact_loss_interpolator(self):
        ceiled_db = AtomicDB(param=self.atomic_db.param, components=self.atomic_db.components, atomic_ceiling=2)
        for level in range(ceiled_db.atomic_ceiling):
            rates = ceiled_db.electron_impact_loss[level](self.INTERPOLATION_TEST_TEMPERATURE)
            self.assertIsInstance(rates, numpy.ndarray, msg='Interpolator output expected to be numpy.')
            numpy.testing.assert_almost_equal(self.EXPECTED_ELECTRON_IMPACT_LOSS[level], rates,
                                              self.EXPECTED_DECIMAL_PRECISION_5, err_msg='Electron impact loss '
                                                                                         'interpolator failure.')

    def test_neutral_db_init(self):
        actual_param, actual_components = self.build_atomic_neutral_input()
        actual_db = AtomicDB(param=actual_param, components=actual_components)
        self.assertTrue(actual_db.are_neutrals, msg='The atomic_db is expected to have neutral cross-section data.')

    def test_atomic_ceiling(self):
        actual_param, actual_components = self.build_atomic_neutral_input()
        actual_db = AtomicDB(param=actual_param, components=actual_components)
        self.assertLessEqual(actual_db.atomic_ceiling, actual_db.atomic_levels, msg='The atomic ceiling is expected to '
                             'be less or equal to the atomic levels from beam with plasma interaction.')
        self.assertLessEqual(actual_db.atomic_ceiling, actual_db.neutral_db.atomic_levels, msg='The atomic ceiling is '
                             'expected to be less or equal to the atomic levels from beam with neutral interaction.')

    def build_atomic_input(self):
        input_gen = AtomicInput(energy=60, projectile='dummy', param_name='AtomicDB_test',
                                source='Unittest', current=0.001)
        input_gen.add_target_component(charge=-1, atomic_number=0, mass_number=0, molecule_name=None)
        for index in range(len(self.INPUT_q)):
            input_gen.add_target_component(charge=self.INPUT_q[index], atomic_number=self.INPUT_z[index],
                                           mass_number=self.INPUT_a[index], molecule_name=self.INPUT_m[index])
        return input_gen.get_atomic_db_input()

    def build_atomic_neutral_input(self):
        input_gen = AtomicInput(energy=50, projectile='H', param_name='AtomicDB_test',
                                source='Unittest', current=0.001)
        for index in range(len(self.INPUT_neutral_q)):
            input_gen.add_target_component(charge=self.INPUT_neutral_q[index],
                                           atomic_number=self.INPUT_neutral_z[index],
                                           mass_number=self.INPUT_neutral_a[index],
                                           molecule_name=self.INPUT_neutral_m[index])
        return input_gen.get_atomic_db_input()
