from atomic.atomic_db import AtomicDB, RenateDB, InternalDB
from utility.convert import convert_from_m2_to_cm2
import numpy as np
import h5py
import os


class AtomicDB_to_HDF_Writer:

    #   Usage:
    #   writer = AtomicDB_to_HDF_Writer(AtomicDB_instance)
    #   writer.write_to(output_directory_str_or_path)

    def __init__(self, atomicdb:AtomicDB):
        self.atomicdb = atomicdb
        self.atomic_levels = list(atomicdb.atomic_dict.keys())
        self.level_num = atomicdb.atomic_levels
        self.einsteins = atomicdb.spontaneous_trans
        self.temperature_axis = atomicdb.temperature_axis
        self.impurities = atomicdb.components[atomicdb.components['q'] > 1]
        self.ions = atomicdb.components[atomicdb.components['q'] > 0]
        proton = self.ions[(self.ions['q']==1) &
                                     (self.ions['Z']==1) &
                                     (self.ions['A']==1)]
        self.proton_ind = self.ions.index.get_indexer(proton.index)[0]

        self.excitation_interpolator_dict = {'electron': atomicdb.electron_impact_trans}
        self.loss_interpolator_dict = {'electron': atomicdb.electron_impact_loss}

        ion_impact_trans_array = np.array(atomicdb.ion_impact_trans)
        ion_impact_loss_array = np.array(atomicdb.ion_impact_loss)

        for i,ion in enumerate(self.ions.index):
            self.excitation_interpolator_dict[ion] = ion_impact_trans_array[:,:,i]
            self.loss_interpolator_dict[ion] = ion_impact_loss_array[:,i]

        self.beam_energy_keV = float(atomicdb.param.xpath('//beamlet_energy/text()')[0])
        if atomicdb.param.xpath('//beamlet_energy/@unit')[0] == 'keV':
            pass
        elif atomicdb.param.xpath('//beamlet_energy/@unit')[0] == 'eV':
            self.beam_energy_keV /= 1000
        else:
            print('Unknown beam energy unit in AtomicDB.param!')

        if isinstance(atomicdb.provider, RenateDB):
            self.beam_type = atomicdb.provider.species
        elif isinstance(atomicdb.provider, InternalDB):
            self.beam_type = atomicdb.provider.projectile
        else:
            print('Unkown atomic data AtomicDB.provider!')

        self.electron_excitation = self.__build_rate_matrix('excitation', 'electron')
        self.ion_excitation = self.__build_ion_rate_matrix('excitation')

        self.electron_impact_loss = self.__build_rate_matrix('loss', 'electron')
        self.ion_impact_loss = self.__build_ion_rate_matrix('loss')

        self.impurity_excitation = np.full((10, self.level_num, self.level_num, self.temperature_axis.shape[0]), np.nan)
        self.impurity_loss = np.full((10, self.level_num, self.temperature_axis.shape[0]), np.nan)
        for i,q in enumerate(self.impurities['q']):
            imp_ind = self.ions[self.ions['q']==q].index
            self.impurity_excitation[q-2] = self.ion_excitation[int(self.ions.index.get_indexer(imp_ind))]
            self.impurity_loss[q-2] = self.ion_impact_loss[int(self.ions.index.get_indexer(imp_ind))]

        self.all_loss = np.vstack((np.expand_dims(self.electron_impact_loss, 0),
                                   np.expand_dims(self.ion_impact_loss[self.proton_ind], 0),
                                   self.impurity_loss))

    def __build_rate_matrix(self, mx_type, target):
        print(mx_type+' '+str(self.beam_type)+'-->'+str(target))
        if mx_type == 'excitation':
            matrix = np.zeros((self.level_num, self.level_num, len(self.temperature_axis)), dtype=float)
            for i in range(self.level_num):
                for j in range(self.level_num):
                    matrix[i,j,:] = self.excitation_interpolator_dict[target][i][j](self.temperature_axis)
            return convert_from_m2_to_cm2(matrix)
        if mx_type == 'loss':
            matrix = np.zeros((self.level_num, len(self.temperature_axis)), dtype = float)
            for i in range(self.level_num):
                matrix[i,:] = self.loss_interpolator_dict[target][i](self.temperature_axis)
            return convert_from_m2_to_cm2(matrix)

    def __build_ion_rate_matrix(self, mx_type):
        matrix = []
        for ion in self.ions.index:
            matrix.append(self.__build_rate_matrix(mx_type, ion))
        return np.array(matrix)

    def write_to(self, output_directory):
        filename = f'rate_coeffs_{int(self.beam_energy_keV)}_{self.beam_type}.h5'
        self.path = os.path.join(output_directory, filename)
        rate_data = h5py.File(self.path, "w")

        rate_data.create_dataset('Beam energy', (), dtype='<i2', data=self.beam_energy_keV)
        rate_data.create_dataset('Atomic Levels', dtype='|S3', data=self.atomic_levels)
        rate_data.create_dataset('Beam type', dtype='|S2', data=self.beam_type)
        rate_data.create_dataset('Einstein Coeffs', dtype='<f4', data=self.einsteins)
        rate_data.create_dataset('Temperature axis', dtype='<f8', data=self.temperature_axis)
        rate_data.create_dataset('Impurity Collisions', dtype='<i2', data=list(self.impurities['q']))

        rate_data.create_dataset('Collisional Coeffs/Electron Neutral Collisions', dtype='<f8',
                                 data=self.electron_excitation)
        rate_data.create_dataset('Collisional Coeffs/Proton Neutral Collisions', dtype='<f8',
                                 data=self.ion_excitation[self.proton_ind])
        rate_data.create_dataset('Collisional Coeffs/Impurity Neutral Collisions', dtype='<f8',
                                 data=self.impurity_excitation)
        rate_data.create_dataset('Collisional Coeffs/Electron Loss Collisions', dtype='<f8',
                                 data=self.all_loss)
        rate_data.close()

        print(f'File written to {self.path}')
        