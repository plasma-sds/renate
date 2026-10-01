import os
import shutil
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
