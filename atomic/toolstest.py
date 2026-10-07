import os
import shutil
import unittest
import numpy as np

from utility.input import BeamletInput
from crm_solver.beamlet import Beamlet
from atomic.atomic_db import AtomicDB
from atomic.atomic_dbtest import AtomicDBTest
from atomic.tools import AtomicDB_to_HDF_Writer


class AtomicDB_to_HDF_WriterTest(unittest.TestCase):

    def setUp(self):
        atomicdb_test = AtomicDBTest()
        atomicdb_test.setUp()
        self.atomicdb = atomicdb_test.atomic_db
        self.writer = AtomicDB_to_HDF_Writer(self.atomicdb)
        self.out_path = os.path.join(os.path.dirname(__file__), '..',
                                    'data', 'dummy', 'atomic_data', 'dummy', 'rates', 'writer_test')
        os.mkdir(self.out_path)
        self.beam = None

    def tearDown(self):
        del self.beam
        if os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path, ignore_errors=True)
        del self.out_path
        del self.writer
        del self.atomicdb

    def test_atomicdb_to_hdf_writer(self):
        self.writer.write_to(self.out_path)
        self.assertTrue(os.path.isfile(self.writer.path),
                        msg='AtomicDB HDF output file is expected at: ' + self.writer.path)

    def test_atomicdb_to_hdf_writer_output(self):
        grid = np.linspace(0,1,100)
        density = np.full_like(grid, 1e19)
        temperature = np.full_like(grid, 1e3)

        input_gen = BeamletInput(energy=60, projectile='dummy', current=0.001,
                                 source='AtomicDB_to_HDF_WriterTest', param_name='AtomicDB_to_HDF_WriterTest')
        input_gen.add_grid(grid)
        
        for _, component in self.atomicdb.components.iterrows():
            input_gen.add_target_profiles(charge=component['q'],
                                atomic_number=component['Z'],
                                mass_number=component['A'],
                                molecule_name=None,
                                density=density,
                                temperature=temperature)
            
        param, comp, profiles = input_gen.get_beamlet_input()
        atomic = AtomicDB(param=param, components=comp, rate_type='writer_test')
        self.beam = Beamlet(param=param, profiles=profiles, components=comp, atomic_db=atomic, solver='numerical')
        self.beam.compute_linear_emission_density()