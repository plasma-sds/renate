import unittest
import warnings
import numpy
import pandas
from crm_solver.beamlet import Beamlet
from observation.observation import Obs1D
from utility.exceptions import RenateNotValidTransitionError


class _StubAtomicDB:
    DEFAULT_TRANSITION = ('9x', '8x', '1x', '9x-->8x')

    def set_default_atomic_levels(self):
        return self.DEFAULT_TRANSITION


class _StubBeamlet:
    GRID = [0., 0.5, 1.]
    EMISSION = [1., 3., 2.]

    def __init__(self, emission_stored=False, solved=True):
        self.atomic_db = _StubAtomicDB()
        self.solved = solved
        self.compute_calls = []
        self.profiles = pandas.DataFrame({('beamlet grid', 'distance', 'm'): self.GRID})
        if emission_stored:
            self.profiles[('9x-->8x', '', '')] = self.EMISSION

    def compute_linear_emission_density(self, to_level=None, from_level=None):
        self.compute_calls.append({'from_level': from_level, 'to_level': to_level})
        if self.solved:
            self.profiles[(from_level + '-->' + to_level, '', '')] = self.EMISSION


class Obs1DInitTest(unittest.TestCase):
    INPUT_POSITIONS = [0.25, 0.5, 1.0]
    INPUT_SIZE_SCALAR = 0.125
    INPUT_SIZE_ARRAY = [0.125, 0.25, 0.5]
    INPUT_INVALID_POSITIONS = [[[0.25, 0.5]], [], [0.25, numpy.nan], [0.25, numpy.inf], [-0.25, 0.5], [0.5, 0.25],
                               [0.25, 0.25]]
    INPUT_INVALID_SIZES = [[0.125, 0.25], [[0.125, 0.25, 0.5]], 0., -0.125, numpy.nan, numpy.inf, [0.125, 0., 0.5]]
    INPUT_INVALID_TRANSITIONS = [(('3n', None), ValueError), ((None, '2n'), ValueError), ((3, 2), TypeError),
                                 (('3n', 2), TypeError)]
    EXPECTED_ATTRIBUTES = ['detector_positions', 'detector_size', 'detector_ranges']

    def test_positions(self):
        obs = Obs1D(self.INPUT_POSITIONS)
        numpy.testing.assert_array_equal(obs.detector_positions, self.INPUT_POSITIONS,
                                         err_msg='Detector positions are not stored as given.')
        self.assertEqual(obs.detector_positions.dtype, numpy.float64, msg='Detector positions are expected as floats.')

    def test_invalid_positions(self):
        for positions in self.INPUT_INVALID_POSITIONS:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError, msg='Invalid detector positions are expected to be rejected.'):
                    Obs1D(positions)

    def test_size_none_single_position(self):
        with self.assertRaises(ValueError, msg='A single position without size is expected to be rejected.'):
            Obs1D(self.INPUT_POSITIONS[:1])

    def test_size_modes(self):
        n = len(self.INPUT_POSITIONS)
        for size in (None, self.INPUT_SIZE_SCALAR, self.INPUT_SIZE_ARRAY):
            with self.subTest(size=size):
                obs = Obs1D(self.INPUT_POSITIONS, size)
                self.assertEqual(obs.detector_positions.shape, (n,), msg='Wrong detector position shape.')
                for attribute, shape in (('detector_size', (n,)), ('detector_ranges', (n, 2))):
                    value = getattr(obs, attribute)
                    self.assertIsInstance(value, numpy.ndarray, msg='<' + attribute + '> is expected as numpy array.')
                    self.assertEqual(value.dtype, numpy.float64, msg='<' + attribute + '> is expected as floats.')
                    self.assertEqual(value.shape, shape, msg='Wrong <' + attribute + '> shape.')

    def test_invalid_sizes(self):
        for size in self.INPUT_INVALID_SIZES:
            with self.subTest(size=size):
                with self.assertRaises(ValueError, msg='Invalid detector sizes are expected to be rejected.'):
                    Obs1D(self.INPUT_POSITIONS, size)

    def test_read_only(self):
        obs = Obs1D(self.INPUT_POSITIONS, self.INPUT_SIZE_ARRAY)
        for attribute in self.EXPECTED_ATTRIBUTES:
            with self.subTest(attribute=attribute):
                with self.assertRaises(ValueError, msg='<' + attribute + '> is expected to be read-only.'):
                    getattr(obs, attribute)[0] = 0.

    def test_input_copied(self):
        positions = numpy.array(self.INPUT_POSITIONS)
        sizes = numpy.array(self.INPUT_SIZE_ARRAY)
        obs = Obs1D(positions, sizes)
        stored_sizes = obs.detector_size.copy()
        self.assertTrue(positions.flags.writeable and sizes.flags.writeable,
                        msg='Input arrays of the caller are expected to remain writeable.')
        positions[0], sizes[0] = 0., 1.
        numpy.testing.assert_array_equal(obs.detector_positions, self.INPUT_POSITIONS,
                                         err_msg='Detector positions are expected to be a copy of the input.')
        numpy.testing.assert_array_equal(obs.detector_size, stored_sizes,
                                         err_msg='Detector sizes are expected to be independent of the input.')

    def test_default_transition_not_given(self):
        obs = Obs1D(self.INPUT_POSITIONS)
        self.assertIsNone(obs.from_level, msg='Default <from_level> is expected to be None.')
        self.assertIsNone(obs.to_level, msg='Default <to_level> is expected to be None.')

    def test_default_transition_given(self):
        obs = Obs1D(self.INPUT_POSITIONS, from_level='3n', to_level='2n')
        self.assertEqual((obs.from_level, obs.to_level), ('3n', '2n'), msg='Default transition is not stored.')

    def test_invalid_default_transition(self):
        for (from_level, to_level), error in self.INPUT_INVALID_TRANSITIONS:
            with self.subTest(from_level=from_level, to_level=to_level):
                with self.assertRaises(error, msg='Invalid default transition is expected to be rejected.'):
                    Obs1D(self.INPUT_POSITIONS, from_level=from_level, to_level=to_level)


class Obs1DTransitionTest(unittest.TestCase):
    INPUT_POSITIONS = [0.25, 0.5, 1.0]
    INPUT_INSTANCE_DEFAULT = ('3n', '2n')
    INPUT_CALL_OVERRIDE = ('4n', '2n')
    INPUT_PRIORITY_CASES = [((None, None), (None, None), ('9x', '8x')),
                            ((None, None), ('4n', '2n'), ('4n', '2n')),
                            (('3n', '2n'), (None, None), ('3n', '2n')),
                            (('3n', '2n'), ('4n', '2n'), ('4n', '2n'))]
    INPUT_SINGLE_LEVEL_OVERRIDES = [('4n', None), (None, '2n')]

    def setUp(self):
        self.beamlet = _StubBeamlet()

    def tearDown(self):
        del self.beamlet

    def test_priority(self):
        for instance_default, call_levels, expected in self.INPUT_PRIORITY_CASES:
            with self.subTest(instance_default=instance_default, call_levels=call_levels):
                obs = Obs1D(self.INPUT_POSITIONS, from_level=instance_default[0], to_level=instance_default[1])
                actual = obs._resolve_transition(self.beamlet, *call_levels)
                self.assertEqual(actual[:2], expected, msg='Transition levels resolved from the wrong source.')
                self.assertEqual(actual[2], expected[0] + '-->' + expected[1],
                                 msg='Transition label is expected in the <from>--><to> format.')

    def test_override_keeps_instance_default(self):
        obs = Obs1D(self.INPUT_POSITIONS, from_level=self.INPUT_INSTANCE_DEFAULT[0],
                    to_level=self.INPUT_INSTANCE_DEFAULT[1])
        obs._resolve_transition(self.beamlet, *self.INPUT_CALL_OVERRIDE)
        self.assertEqual((obs.from_level, obs.to_level), self.INPUT_INSTANCE_DEFAULT,
                         msg='A call override is not expected to change the instance default transition.')

    def test_single_level_override(self):
        obs = Obs1D(self.INPUT_POSITIONS)
        for from_level, to_level in self.INPUT_SINGLE_LEVEL_OVERRIDES:
            with self.subTest(from_level=from_level, to_level=to_level):
                with self.assertRaises(ValueError, msg='Overriding a single level is expected to be rejected.'):
                    obs._resolve_transition(self.beamlet, from_level, to_level)


class Obs1DEmissionDensityTest(unittest.TestCase):
    INPUT_TRANSITION = ('9x', '8x', '9x-->8x')
    EXPECTED_COMPUTE_CALL = {'from_level': '9x', 'to_level': '8x'}

    def test_stored_emission_reused(self):
        beamlet = _StubBeamlet(emission_stored=True)
        _, emission = Obs1D._get_linear_emission_density(beamlet, *self.INPUT_TRANSITION)
        self.assertEqual(beamlet.compute_calls, [], msg='Stored emission is not expected to be recomputed.')
        numpy.testing.assert_array_equal(emission, _StubBeamlet.EMISSION,
                                         err_msg='Stored emission is expected to be returned.')

    def test_missing_emission_computed(self):
        beamlet = _StubBeamlet()
        _, emission = Obs1D._get_linear_emission_density(beamlet, *self.INPUT_TRANSITION)
        self.assertEqual(beamlet.compute_calls, [self.EXPECTED_COMPUTE_CALL],
                         msg='Missing emission is expected to be computed once with matching levels.')
        numpy.testing.assert_array_equal(emission, _StubBeamlet.EMISSION,
                                         err_msg='Computed emission is expected to be returned.')

    def test_unsolved_beamlet(self):
        with self.assertRaises(RuntimeError, msg='Unsolved beamlet is expected to raise RuntimeError.'):
            Obs1D._get_linear_emission_density(_StubBeamlet(solved=False), *self.INPUT_TRANSITION)

    def test_return_types(self):
        grid, emission = Obs1D._get_linear_emission_density(_StubBeamlet(emission_stored=True),
                                                            *self.INPUT_TRANSITION)
        for name, value, reference in (('grid', grid, _StubBeamlet.GRID),
                                       ('emission', emission, _StubBeamlet.EMISSION)):
            with self.subTest(output=name):
                self.assertIsInstance(value, numpy.ndarray, msg='<' + name + '> is expected as numpy array.')
                numpy.testing.assert_array_equal(value, reference, err_msg='<' + name + '> is returned incorrectly.')


class Obs1DIntegratorTest(unittest.TestCase):
    INPUT_LINEAR_GRID = numpy.array([0., 0.1, 0.35, 0.4, 1.])
    INPUT_LINEAR_COEFFICIENTS = (2., 3.)
    INPUT_LINEAR_RANGES = numpy.array([[0.05, 0.07], [0.05, 0.9], [0.1, 0.4], [0., 0.2], [0.6, 1.], [0., 1.],
                                       [0.35, 0.35]])
    INPUT_ADJACENT_RANGES = numpy.array([[0., 0.2], [0.2, 0.37], [0.37, 1.]])
    INPUT_KINK_GRID = numpy.array([0., 1., 2., 4.])
    INPUT_KINK_EMISSION = numpy.array([0., 2., 1., 5.])
    INPUT_KINK_RANGES = numpy.array([[0.5, 1.5], [1.5, 3.], [0., 1.], [1., 2.], [2., 4.], [0., 4.]])
    EXPECTED_KINK_INTEGRALS = [1.625, 2.625, 1., 1.5, 6., 8.5]
    PRECISION = {'rtol': 1e-12, 'atol': 1e-14}

    def _linear_emission(self):
        offset, slope = self.INPUT_LINEAR_COEFFICIENTS
        return offset + slope * self.INPUT_LINEAR_GRID

    def _linear_integral(self, ranges):
        offset, slope = self.INPUT_LINEAR_COEFFICIENTS
        lower, upper = ranges[:, 0], ranges[:, 1]
        return offset * (upper - lower) + slope / 2. * (upper ** 2 - lower ** 2)

    def test_linear_emission(self):
        actual = Obs1D._integrate_over_detectors(self.INPUT_LINEAR_GRID, self._linear_emission(),
                                                 self.INPUT_LINEAR_RANGES)
        numpy.testing.assert_allclose(actual, self._linear_integral(self.INPUT_LINEAR_RANGES), **self.PRECISION,
                                      err_msg='Linear emission is expected to be integrated exactly.')

    def test_piecewise_linear_emission(self):
        actual = Obs1D._integrate_over_detectors(self.INPUT_KINK_GRID, self.INPUT_KINK_EMISSION,
                                                 self.INPUT_KINK_RANGES)
        numpy.testing.assert_allclose(actual, self.EXPECTED_KINK_INTEGRALS, **self.PRECISION,
                                      err_msg='Piecewise linear emission is expected to be integrated exactly.')

    def test_additivity(self):
        emission = self._linear_emission()
        parts = Obs1D._integrate_over_detectors(self.INPUT_LINEAR_GRID, emission, self.INPUT_ADJACENT_RANGES)
        whole = Obs1D._integrate_over_detectors(self.INPUT_LINEAR_GRID, emission,
                                                self.INPUT_ADJACENT_RANGES[[0, -1], [0, 1]].reshape(1, 2))
        numpy.testing.assert_allclose(parts.sum(), whole[0], **self.PRECISION,
                                      err_msg='Integrals over adjacent ranges are expected to add up.')

    def test_output(self):
        actual = Obs1D._integrate_over_detectors(self.INPUT_LINEAR_GRID, self._linear_emission(),
                                                 self.INPUT_LINEAR_RANGES)
        self.assertIsInstance(actual, numpy.ndarray, msg='Integrals are expected as numpy array.')
        self.assertEqual(actual.dtype, numpy.float64, msg='Integrals are expected as floats.')
        self.assertEqual(actual.shape, (len(self.INPUT_LINEAR_RANGES),), msg='One integral per range is expected.')


class Obs1DOutsideGridTest(unittest.TestCase):
    INPUT_GRID = numpy.array([0., 0.5, 1.])
    INPUT_POSITIONS = [0., 0.0625, 0.5, 0.9375, 1., 1.5]
    INPUT_SIZE = 0.125
    INPUT_FLUX = [1., 2., 3., 4., 5., 6.]
    INPUT_INSIDE_POSITIONS = [0.0625, 0.5, 0.9375]
    EXPECTED_FLUX = [numpy.nan, 2., 3., 4., numpy.nan, numpy.nan]
    EXPECTED_INDICES = '[0, 4, 5]'

    def _flag(self, positions, flux):
        obs = Obs1D(positions, self.INPUT_SIZE)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            actual = obs._flag_detectors_outside_grid(self.INPUT_GRID, numpy.array(flux))
        return actual, caught

    def test_outside_detectors_flagged(self):
        actual, _ = self._flag(self.INPUT_POSITIONS, self.INPUT_FLUX)
        numpy.testing.assert_array_equal(actual, self.EXPECTED_FLUX,
                                         err_msg='Only detectors extending beyond the grid are expected to be NaN.')

    def test_warning(self):
        _, caught = self._flag(self.INPUT_POSITIONS, self.INPUT_FLUX)
        self.assertEqual(len(caught), 1, msg='Exactly one warning is expected.')
        self.assertTrue(issubclass(caught[0].category, UserWarning), msg='A UserWarning is expected.')
        self.assertIn(self.EXPECTED_INDICES, str(caught[0].message),
                      msg='The warning is expected to list the detectors outside the grid.')

    def test_all_inside(self):
        flux = self.INPUT_FLUX[:len(self.INPUT_INSIDE_POSITIONS)]
        actual, caught = self._flag(self.INPUT_INSIDE_POSITIONS, flux)
        numpy.testing.assert_array_equal(actual, flux, err_msg='Detectors inside the grid are expected unchanged.')
        self.assertEqual(len(caught), 0, msg='No warning is expected when all detectors are inside the grid.')


class Obs1DPhotonFluxTest(unittest.TestCase):
    INPUT_OVERRIDE = ('3p', '2s')
    INPUT_REVERSED = ('2s', '2p')
    INPUT_DETECTOR_NUMBER = 5

    @classmethod
    def setUpClass(cls):
        cls.REFERENCE = Beamlet()
        grid = cls.REFERENCE.profiles[('beamlet grid', 'distance', 'm')].to_numpy()
        span = grid[-1] - grid[0]
        cls.GRID = grid
        cls.POSITIONS = numpy.linspace(grid[0] + 0.2 * span, grid[-1] - 0.2 * span, cls.INPUT_DETECTOR_NUMBER)
        cls.DEFAULT = cls.REFERENCE.atomic_db.set_default_atomic_levels()

    def setUp(self):
        self.beamlet = self.REFERENCE.copy()
        self.obs = Obs1D(self.POSITIONS)

    def tearDown(self):
        del self.beamlet, self.obs

    def test_single_beamlet(self):
        positions, flux = self.obs.compute_photon_flux(self.beamlet)
        self.assertIs(positions, self.obs.detector_positions, msg='Detector positions are expected to be returned.')
        self.assertIsInstance(flux, numpy.ndarray, msg='Photon flux is expected as numpy array.')
        self.assertEqual(flux.shape, (self.INPUT_DETECTOR_NUMBER,), msg='One flux value per detector is expected.')
        self.assertTrue(numpy.all(numpy.isfinite(flux)) and numpy.all(flux > 0),
                        msg='Photon flux inside the grid is expected to be finite and positive.')

    def test_beamlet_sequence(self):
        for beamlets in ([self.beamlet, self.beamlet], (self.beamlet, self.beamlet)):
            with self.subTest(sequence=type(beamlets).__name__):
                _, flux = self.obs.compute_photon_flux(beamlets)
                self.assertEqual(flux.shape, (2, self.INPUT_DETECTOR_NUMBER), msg='One row per beamlet is expected.')
                numpy.testing.assert_array_equal(flux[0], flux[1], err_msg='Identical beamlets are expected to give '
                                                                           'identical flux.')

    def test_sequence_of_one(self):
        _, flux = self.obs.compute_photon_flux([self.beamlet])
        self.assertEqual(flux.shape, (1, self.INPUT_DETECTOR_NUMBER),
                         msg='A sequence of one beamlet is expected to keep the beamlet dimension.')

    def test_matches_integrated_emission(self):
        _, flux = self.obs.compute_photon_flux(self.beamlet)
        emission = self.beamlet.profiles[self.DEFAULT[3]].to_numpy()
        expected = Obs1D._integrate_over_detectors(self.GRID, emission, self.obs.detector_ranges)
        numpy.testing.assert_array_equal(flux, expected, err_msg='Photon flux is expected to be the integrated '
                                                                 'emission of the beamlet over the detector ranges.')

    def test_species_default(self):
        _, flux = self.obs.compute_photon_flux(self.beamlet)
        explicit = Obs1D(self.POSITIONS, from_level=self.DEFAULT[0], to_level=self.DEFAULT[1])
        _, flux_explicit = explicit.compute_photon_flux(self.beamlet)
        numpy.testing.assert_array_equal(flux, flux_explicit,
                                         err_msg='Species default transition is expected without class defaults.')

    def test_call_override(self):
        _, flux_default = self.obs.compute_photon_flux(self.beamlet)
        _, flux_override = self.obs.compute_photon_flux(self.beamlet, *self.INPUT_OVERRIDE)
        instance = Obs1D(self.POSITIONS, from_level=self.INPUT_OVERRIDE[0], to_level=self.INPUT_OVERRIDE[1])
        _, flux_instance = instance.compute_photon_flux(self.beamlet)
        numpy.testing.assert_array_equal(flux_override, flux_instance,
                                         err_msg='Call override and class default are expected to give equal flux.')
        self.assertFalse(numpy.array_equal(flux_override, flux_default),
                         msg='A different transition is expected to give a different flux.')
        self.assertIsNone(self.obs.from_level, msg='A call override is not expected to change class defaults.')

    def test_emission_added_to_beamlet(self):
        label = self.INPUT_OVERRIDE[0] + '-->' + self.INPUT_OVERRIDE[1]
        self.obs.compute_photon_flux(self.beamlet, *self.INPUT_OVERRIDE)
        self.assertIn(label, self.beamlet.profiles, msg='Computed emission is expected in the beamlet profiles.')

    def test_store_off(self):
        self.obs.compute_photon_flux(self.beamlet)
        for attribute in ('photon_flux', 'observed_transitions', 'observed_beamlets'):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(self.obs, attribute), msg='Nothing is expected to be stored by default.')

    def test_store_on(self):
        source = self._beamlet_source(self.beamlet)
        _, flux = self.obs.compute_photon_flux(self.beamlet, store=True)
        numpy.testing.assert_array_equal(self.obs.photon_flux, flux, err_msg='Returned flux is expected stored.')
        self.assertEqual(self.obs.observed_transitions, [self.DEFAULT[3]], msg='Transition label is expected stored.')
        self.assertEqual(self.obs.observed_beamlets, [source], msg='Beamlet source is expected stored.')
        _, flux = self.obs.compute_photon_flux(self.beamlet, *self.INPUT_OVERRIDE, store=True)
        numpy.testing.assert_array_equal(self.obs.photon_flux, flux, err_msg='Stored flux is expected overwritten.')
        self.assertEqual(self.obs.observed_transitions, [self.INPUT_OVERRIDE[0] + '-->' + self.INPUT_OVERRIDE[1]],
                         msg='Stored transition label is expected overwritten.')
        self.assertEqual(self.obs.observed_beamlets, [source], msg='Stored beamlet source is expected overwritten.')

    def test_store_beamlet_sequence(self):
        source = self._beamlet_source(self.beamlet)
        self.obs.compute_photon_flux([self.beamlet, self.beamlet], store=True)
        self.assertEqual(self.obs.observed_beamlets, [source, source],
                         msg='One beamlet source per beamlet is expected stored in order.')

    def test_store_missing_beamlet_source(self):
        body = self.beamlet.param.getroot().find('body')
        body.remove(body.find('beamlet_source'))
        _, flux = self.obs.compute_photon_flux(self.beamlet, store=True)
        self.assertEqual(self.obs.observed_beamlets, [None], msg='A missing beamlet source is expected as None.')
        self.assertTrue(numpy.all(numpy.isfinite(flux)), msg='Photon flux is expected without beamlet source.')

    @staticmethod
    def _beamlet_source(beamlet):
        return beamlet.param.getroot().find('body').find('beamlet_source').text

    def test_outside_grid(self):
        obs = Obs1D([self.POSITIONS[0], self.GRID[-1]], (self.POSITIONS[1] - self.POSITIONS[0]) / 2.)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, flux = obs.compute_photon_flux(self.beamlet)
        self.assertTrue(numpy.isfinite(flux[0]) and numpy.isnan(flux[1]),
                        msg='Only the detector beyond the grid is expected to be NaN.')
        self.assertEqual(len(caught), 1, msg='Exactly one warning is expected.')

    def test_errors(self):
        cases = [('single level override', ValueError, lambda: self.obs.compute_photon_flux(self.beamlet, '3p')),
                 ('reversed transition', RenateNotValidTransitionError,
                  lambda: self.obs.compute_photon_flux(self.beamlet, *self.INPUT_REVERSED)),
                 ('unsolved beamlet', RuntimeError,
                  lambda: self.obs.compute_photon_flux(Beamlet(solver='disregard')))]
        for name, error, call in cases:
            with self.subTest(case=name):
                with self.assertRaises(error, msg='<' + name + '> is expected to raise ' + error.__name__ + '.'):
                    call()


if __name__ == '__main__':
    unittest.main()
