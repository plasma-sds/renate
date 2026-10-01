import os
import shutil
import tempfile
import unittest
import pandas

from utility import writedata as writedata_module
from utility.writedata import WriteData
from crm_solver.beamlet import Beamlet


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


if __name__ == '__main__':
    unittest.main()
