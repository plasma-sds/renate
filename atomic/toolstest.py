import os
import shutil
import tempfile
import unittest

from atomic.atomic_dbtest import AtomicDBTest
from atomic.tools import AtomicDB_to_HDF_Writer


class AtomicDB_to_HDF_WriterTest(unittest.TestCase):

    def setUp(self):
        atomicdb_test = AtomicDBTest()
        atomicdb_test.setUp()
        self.atomicdb = atomicdb_test.atomic_db
        self.writer = AtomicDB_to_HDF_Writer(self.atomicdb)
        self.tmp_root = tempfile.mkdtemp(prefix='atomicdbwriter_test_')

    def tearDown(self):
        del self.writer
        del self.atomicdb
        if os.path.isdir(self.tmp_root):
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_atomicdb_to_hdf_writer(self):
        self.writer.write_to(self.tmp_root)
        self.assertTrue(os.path.isfile(self.writer.path),
                        msg='AtomicDB HDF output file is expected at: ' + self.writer.path)