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

    def setUpClass(self):
        self.atomicdb_test = AtomicDBTest()
        self.atomicdb_test.setUp()
        self.atomicdb = self.atomicdb_test.atomic_db
        self.writer = AtomicDB_to_HDF_Writer(self.atomicdb)
        self.out_path = os.path.join(os.path.dirname(__file__), '..',
                                    'data', 'dummy', 'atomic_data', 'dummy', 'rates', 'writer_test')
        os.mkdir(self.out_path)
        self.writer.write_to(self.out_path)
        self.assertTrue(os.path.isfile(self.writer.path),
                        msg='AtomicDB HDF output file is expected at: ' + self.writer.path)

    def tearDownClass(self):
        if os.path.isdir(self.out_path):
            shutil.rmtree(self.out_path, ignore_errors=True)
        del self.out_path
        del self.writer
        del self.atomicdb
        del self.atomicdb_test

    def setUp(self):
        self.beam = None

    def tearDown(self):
        del self.beam

    def test_beam_evolution_with_output(self):
        grid = np.linspace(0,1,100)
        density = np.full_like(grid, 1e19)
        temperature = np.full_like(grid, 1e3)

        # energy and projectile here have to be the same as the ones used by the test instance of AtomicDB
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

    def test_values_of_output(self):
        param, comp = self.atomicdb_test.build_atomic_input()
        new_atomic = AtomicDB(param=param, components=comp, rate_type='writer_test')

        T_min = np.min(self.atomicdb.temperature_axis)
        T_max = np.max(self.atomicdb.temperature_axis)
        T_values = np.linspace(T_min, T_max, 10)

        N_levels = len(self.atomicdb.electron_impact_loss)
        self.assertEqual(N_levels, len(new_atomic.electron_impact_loss))
        N_ions = len(self.atomicdb.ion_impact_loss[0])
        self.assertEqual(N_ions, len(new_atomic.ion_impact_loss[0]))

        for i in range(N_levels):
            np.testing.assert_allclose(new_atomic.electron_impact_loss[i](T_values),
                                       self.atomicdb.electron_impact_loss[i](T_values))

        for i in range(N_levels):
            for j in range(N_levels):
                np.testing.assert_allclose(new_atomic.electron_impact_trans[i][j](T_values),
                                           self.atomicdb.electron_impact_trans[i][j](T_values))

        for i in range(N_levels):
            for j in range(N_ions):
                np.testing.assert_allclose(new_atomic.ion_impact_loss[i][j](T_values),
                                           self.atomicdb.ion_impact_loss[i][j](T_values))

        for i in range(N_levels):
            for j in range(N_levels):
                for k in range(N_ions):
                    np.testing.assert_allclose(new_atomic.ion_impact_trans[i][j][k](T_values),
                                               self.atomicdb.ion_impact_trans[i][j][k](T_values))

        np.testing.assert_allclose(new_atomic.spontaneous_trans, self.atomicdb.spontaneous_trans)