import os
import shutil
import tempfile
import unittest
import h5py
import numpy
import pandas

from utility import writedata as writedata_module
from utility.writedata import WriteData
from crm_solver.beamlet import Beamlet
from observation.observation import Obs1D


class WriteDataTest(unittest.TestCase):

    SUBDIR = 'subtest'

    @classmethod
    def setUpClass(cls):
        # Construct the default-dummy beamlet once (expensive: reads XML, loads HDF5,
        # builds AtomicDB). Each test clones it so XML mutations stay isolated.
        # solver='disregard' skips the ODE solve - WriteData does not need its results.
        cls._template_beamlet = Beamlet(solver='disregard')

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='writedata_test_')
        # Trailing separator keeps internal path composition consistent with
        # the module default and avoids coupling the test to os.path.join quirks.
        self.writer = WriteData(root_path=self.tmp_root + os.sep)
        # Fresh deepcopy so each test can safely mutate .param / .profiles.
        self.beamlet = self._template_beamlet.copy(object_copy='full')
        self.output_id = self.beamlet.param.getroot().find('head').find('id').text
        self.initial_source = self.beamlet.param.getroot().find('body').find('beamlet_source').text

    def tearDown(self):
        if os.path.isdir(self.tmp_root):
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        del self.writer
        del self.beamlet

    # ------------------------------------------------------------------ init

    def test_default_root_path_is_relative_to_module(self):
        default_writer = WriteData()
        expected = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(writedata_module.__file__)),
            '..', 'data', 'output'))
        self.assertEqual(os.path.normpath(default_writer.root_path.rstrip(os.sep)),
                         expected,
                         msg='Default root_path is expected to resolve relative to '
                             'utility/writedata.py (<module_dir>/../data/output).')

    # ------------------------------------------------- write_beamlet_profiles

    def test_write_beamlet_profiles_creates_files(self):
        h5_full, xml_full = self.writer.write_beamlet_profiles(self.beamlet)
        self.assertTrue(os.path.isfile(h5_full),
                        msg='HDF5 output file is expected at: ' + h5_full)
        self.assertTrue(os.path.isfile(xml_full),
                        msg='XML output file is expected at: ' + xml_full)

    def test_write_beamlet_profiles_returns_expected_paths(self):
        h5_full, xml_full = self.writer.write_beamlet_profiles(self.beamlet)
        self.assertEqual(h5_full,
                         os.path.join(self.writer.root_path, self.output_id + '.h5'),
                         msg='Returned HDF5 path is expected to equal '
                             'os.path.join(root_path, <id>.h5).')
        self.assertEqual(xml_full,
                         os.path.join(self.writer.root_path, self.output_id + '.xml'),
                         msg='Returned XML path is expected to equal '
                             'os.path.join(root_path, <id>.xml).')

    def test_write_beamlet_profiles_uses_subdir(self):
        h5_full, xml_full = self.writer.write_beamlet_profiles(self.beamlet, subdir=self.SUBDIR)
        expected_dir = os.path.join(self.writer.root_path.rstrip(os.sep), self.SUBDIR)
        self.assertTrue(os.path.isfile(h5_full),
                        msg='HDF5 file in subdir is expected at: ' + h5_full)
        self.assertTrue(os.path.isfile(xml_full),
                        msg='XML file in subdir is expected at: ' + xml_full)
        self.assertEqual(os.path.normpath(os.path.dirname(h5_full)),
                         os.path.normpath(expected_dir),
                         msg='subdir is expected to be composed under root_path '
                             'with os.path.join.')

    def test_write_beamlet_profiles_appends_history(self):
        self.writer.write_beamlet_profiles(self.beamlet)
        history_element = self.beamlet.param.getroot().find('body').find('beamlet_history')
        self.assertIsNotNone(history_element,
                             msg='beamlet_history element is expected to be appended '
                                 'to the XML on the first write.')
        self.assertEqual(history_element.text, self.initial_source,
                         msg='beamlet_history is expected to record the previous '
                             'beamlet_source value.')
        self.assertEqual(history_element.get('unit'), '-',
                         msg="beamlet_history element is expected to have unit='-'.")

    def test_write_beamlet_profiles_updates_beamlet_source(self):
        self.writer.write_beamlet_profiles(self.beamlet)
        source = self.beamlet.param.getroot().find('body').find('beamlet_source').text
        self.assertEqual(source, self.output_id + '.h5',
                         msg='beamlet_source is expected to be set to the new '
                             'relative HDF5 filename after writing.')

    def test_write_beamlet_profiles_data_roundtrip(self):
        expected_profiles = self.beamlet.profiles.copy()
        expected_components = self.beamlet.components.copy()
        h5_full, _ = self.writer.write_beamlet_profiles(self.beamlet)
        profiles_rt = pandas.read_hdf(h5_full, key='profiles')
        components_rt = pandas.read_hdf(h5_full, key='components')
        self.assertTrue(profiles_rt.equals(expected_profiles),
                        msg='Profiles DataFrame is expected to round-trip through HDF5 unchanged.')
        self.assertTrue(components_rt.equals(expected_components),
                        msg='Components DataFrame is expected to round-trip through HDF5 unchanged.')

    def test_write_beamlet_profiles_raises_on_bad_input(self):
        self.beamlet.profiles = None  # triggers AttributeError inside the try block
        with self.assertRaises(Exception,
                               msg='write_beamlet_profiles is expected to re-raise '
                                   'as Exception when the inner write fails.'):
            self.writer.write_beamlet_profiles(self.beamlet)


class WriteObservationTest(unittest.TestCase):
    OUTPUT_ID = 'observation_test'
    SUBDIR = 'subtest'
    INPUT_POSITIONS = [0.25, 0.5, 1.0]
    INPUT_SIZE = 0.125
    INPUT_FLUX = [[1., 2., numpy.nan], [4., 5., 6.]]
    INPUT_TRANSITIONS = ['2p-->2s', '3p-->2s']
    INPUT_BEAMLETS = ['beamlet/test_impurity.h5', None]
    EXPECTED_BEAMLETS = ['beamlet/test_impurity.h5', '']
    EXPECTED_DATASETS = {'detector_positions', 'detector_size', 'detector_ranges', 'photon_flux',
                         'observed_transitions', 'observed_beamlets'}
    EXPECTED_UNITS = {'detector_positions': 'm', 'detector_size': 'm', 'detector_ranges': 'm', 'photon_flux': 'ph/s'}

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix='writeobservation_test_')
        self.writer = WriteData(root_path=self.tmp_root + os.sep)
        self.obs = Obs1D(self.INPUT_POSITIONS, self.INPUT_SIZE)
        self._store(self.INPUT_FLUX, self.INPUT_TRANSITIONS, self.INPUT_BEAMLETS)

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        del self.writer, self.obs

    def _store(self, flux, transitions, beamlets):
        self.obs.photon_flux = numpy.array(flux)
        self.obs.observed_transitions = list(transitions)
        self.obs.observed_beamlets = list(beamlets)

    def test_returns_path(self):
        path = self.writer.write_observation(self.obs, self.OUTPUT_ID)
        self.assertEqual(path, os.path.join(self.writer.root_path, self.OUTPUT_ID + '.h5'),
                         msg='Returned path is expected to equal os.path.join(root_path, <output_id>.h5).')
        self.assertTrue(os.path.isfile(path), msg='HDF5 output file is expected at: ' + path)

    def test_subdir(self):
        path = self.writer.write_observation(self.obs, self.OUTPUT_ID, subdir=self.SUBDIR)
        self.assertTrue(os.path.isfile(path), msg='HDF5 file in subdir is expected at: ' + path)
        self.assertEqual(os.path.normpath(os.path.dirname(path)),
                         os.path.normpath(os.path.join(self.writer.root_path, self.SUBDIR)),
                         msg='subdir is expected to be composed under root_path.')

    def test_contents(self):
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            self.assertEqual(set(h5_file.keys()), self.EXPECTED_DATASETS, msg='Unexpected datasets in the file.')

    def test_detector_arrays(self):
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            for name in ('detector_positions', 'detector_size', 'detector_ranges'):
                with self.subTest(dataset=name):
                    numpy.testing.assert_array_equal(h5_file[name][()], getattr(self.obs, name),
                                                     err_msg='<' + name + '> is expected to round-trip unchanged.')

    def test_photon_flux(self):
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            numpy.testing.assert_array_equal(h5_file['photon_flux'][()], self.INPUT_FLUX,
                                             err_msg='Photon flux is expected to round-trip unchanged.')

    def test_single_result(self):
        self._store(self.INPUT_FLUX[1], self.INPUT_TRANSITIONS[:1], self.INPUT_BEAMLETS[:1])
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            self.assertEqual(h5_file['photon_flux'].shape, (1, len(self.INPUT_POSITIONS)),
                             msg='A single photon flux profile is expected to be written as one row.')

    def test_labels(self):
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            self.assertEqual(list(h5_file['observed_transitions'].asstr()[()]), self.INPUT_TRANSITIONS,
                             msg='Transition labels are expected to round-trip unchanged.')
            self.assertEqual(list(h5_file['observed_beamlets'].asstr()[()]), self.EXPECTED_BEAMLETS,
                             msg='Beamlet sources are expected to round-trip, with None written as empty string.')

    def test_units(self):
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            for name, unit in self.EXPECTED_UNITS.items():
                with self.subTest(dataset=name):
                    self.assertEqual(h5_file[name].attrs['unit'], unit, msg='Wrong unit for <' + name + '>.')

    def test_overwrite(self):
        self.writer.write_observation(self.obs, self.OUTPUT_ID)
        self._store(self.INPUT_FLUX[1], self.INPUT_TRANSITIONS[:1], self.INPUT_BEAMLETS[:1])
        with h5py.File(self.writer.write_observation(self.obs, self.OUTPUT_ID), 'r') as h5_file:
            self.assertEqual(set(h5_file.keys()), self.EXPECTED_DATASETS, msg='Unexpected datasets after overwrite.')
            self.assertEqual(h5_file['photon_flux'].shape, (1, len(self.INPUT_POSITIONS)),
                             msg='A second write is expected to replace the file contents.')

    def test_nothing_stored(self):
        obs = Obs1D(self.INPUT_POSITIONS, self.INPUT_SIZE)
        with self.assertRaises(ValueError, msg='Writing without stored photon flux is expected to raise ValueError.'):
            self.writer.write_observation(obs, self.OUTPUT_ID)
        self.assertFalse(os.path.exists(os.path.join(self.writer.root_path, self.OUTPUT_ID + '.h5')),
                         msg='No file is expected when nothing is stored.')

    def test_write_failure(self):
        self.obs.photon_flux = [[object()]]
        with self.assertRaises(OSError, msg='A failed write is expected to raise OSError.'):
            self.writer.write_observation(self.obs, self.OUTPUT_ID)


if __name__ == '__main__':
    unittest.main()
