import utility
from utility.constants import Constants
from utility.exceptions import RenateNotValidTransitionError
import pandas
import numpy
from copy import deepcopy
from lxml import etree
from crm_solver.coefficientmatrix import CoefficientMatrix
from crm_solver.ode import Ode
from atomic.atomic_db import AtomicDB


class Beamlet:
    def __init__(self, param=None, profiles=None, components=None, atomic_db=None,
                 solver='numerical', data_path="beamlet/testimp0001.xml"):
        self.param = param
        if not isinstance(self.param, etree._ElementTree):
            self.__read_beamlet_param(data_path)
        self.profiles = profiles
        self.components = components
        self.atomic_db = atomic_db
        if not (isinstance(self.components, pandas.DataFrame) and isinstance(self.profiles, pandas.DataFrame)):
            self.__read_beamlet_profiles()
        if atomic_db is None:
            self.atomic_db = AtomicDB(param=self.param, components=self.components)
        self.const = Constants()
        self.coefficient_matrix = None
        self.initial_condition = None
        self.calculate_beamevolution(solver)

    def __read_beamlet_param(self, data_path):
        self.param = utility.getdata.GetData(data_path_name=data_path).data
        assert isinstance(self.param, etree._ElementTree)
        print('Beamlet.param read from file: ' + data_path)

    def __read_beamlet_profiles(self):
        hdf5_path = self.param.getroot().find('body').find('beamlet_source').text
        self.components = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['components']).data
        assert isinstance(self.components, pandas.DataFrame)
        print('Beamlet.imp_components read from file: ' + hdf5_path)
        self.profiles = utility.getdata.GetData(data_path_name=hdf5_path, data_key=['profiles']).data
        assert isinstance(self.profiles, pandas.DataFrame)
        print('Beamlet.imp_profiles read from file: ' + hdf5_path)

    def __initialize_ode(self):
        self.coefficient_matrix = CoefficientMatrix(self.param, self.profiles, self.components, self.atomic_db)
        self.initial_condition = [self.__get_linear_density()] + [0.] * (self.atomic_db.atomic_ceiling - 1)

    def __get_linear_density(self):
        current = float(self.param.getroot().find('body').find('beamlet_current').text)
        return current / (self.atomic_db.velocity * self.const.charge_electron)

    def __solve_numerically(self):
        if self.coefficient_matrix is None or self.initial_condition is None:
            self.__initialize_ode()

        ode = Ode(coeff_matrix=self.coefficient_matrix.matrix, init_condition=self.initial_condition)
        numerical = ode.calculate_numerical_solution(self.profiles['beamlet grid']['distance']['m'])

        for level in range(self.atomic_db.atomic_ceiling):
            label = 'level ' + self.atomic_db.inv_atomic_dict[level]
            self.profiles[label] = numerical[:, level]
        return

    def calculate_beamevolution(self, solver):
        assert isinstance(solver, str)
        if solver == 'numerical':
            self.__solve_numerically()
        elif solver == 'analytical':
            raise NotImplementedError('Analytical solver not yet implemented.')
        elif solver == 'disregard':
            print('Beam evolution not calculated.')
            return
        else:
            raise ValueError('The numerical solver: ' + solver + ' is not supported. '
                            'Supported solvers are: numerical, analytical, disregard.')

    def __was_beamevolution_performed(self):
        try:
            dummy = self.profiles['level ' + self.atomic_db.set_default_atomic_levels()[2]]
            return True
        except KeyError:
            return False

    def compute_linear_emission_density(self, to_level=None, from_level=None):
        if to_level is None or from_level is None:
            from_level, to_level, ground_level, transition_label = self.atomic_db.set_default_atomic_levels()
        if isinstance(to_level, str) and isinstance(from_level, str):
            if self.atomic_db.atomic_dict[to_level] >= self.atomic_db.atomic_dict[from_level]:
                raise RenateNotValidTransitionError(
                    'Electrons spontaneously transit from higher to lower atomic states. '
                    'The requested transition from level ' + from_level + ' to level ' + to_level +
                    ' is not valid.')
        else:
            raise TypeError('The expected input for atomic transitions are strings. '
                            'Bundled-n for H,D,T beam species ex:[1, 2, ... 6]. '
                            'l-n resolved labels for Li ex: [2s, 2p, ... 4f] and Na ex: [3s, 3p, ... 5s]')
        if self.__was_beamevolution_performed():
            transition_label = from_level + '-->' + to_level
            self.profiles[transition_label] = \
                self.profiles['level '+from_level] * self.atomic_db.spontaneous_trans[self.atomic_db.atomic_dict
                                                                                      [to_level], self.atomic_db.atomic_dict[from_level]]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def compute_linear_density_attenuation(self):
        if self.__was_beamevolution_performed():
            self.profiles['linear_density_attenuation'] = self.profiles['level ' + self.atomic_db.inv_atomic_dict[0]]
            for level in range(1, self.atomic_db.atomic_ceiling):
                self.profiles['linear_density_attenuation'] += self.profiles['level ' +
                                                                             self.atomic_db.inv_atomic_dict[level]]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def compute_relative_populations(self, reference_level=None):
        if self.__was_beamevolution_performed():
            if reference_level is None:
                reference_level = self.atomic_db.set_default_atomic_levels()[2]
            assert isinstance(reference_level, str)
            for level in range(self.atomic_db.atomic_ceiling):
                self.profiles['rel.pop ' + self.atomic_db.inv_atomic_dict[level]] = \
                    self.profiles['level ' + self.atomic_db.inv_atomic_dict[level]] / \
                    self.profiles['level ' + reference_level]
        else:
            print('Beam evolution calculations were not performed. Execute solver first.')

    def copy(self, object_copy='full'):
        if not isinstance(object_copy, str):
            raise TypeError('The expected data type for <object_copy> is str.')
        if object_copy == 'full':
            return deepcopy(self)
        elif object_copy == 'without-results':
            beamlet = deepcopy(self)
            beamlet.profiles = self._copy_profiles_input()
            return beamlet
        else:
            raise ValueError('The <object_copy> variable does not support ' + object_copy)

    def _copy_profiles_input(self):
        profiles = numpy.zeros((1 + len(self.components) * 2, len(self.profiles)))
        type_labels = []
        property_labels = []
        unit_labels = []

        profiles[0, :] = self.profiles['beamlet grid']['distance']['m']
        type_labels.append('beamlet grid')
        property_labels.append('distance')
        unit_labels.append('m')
        count = 1

        for component in self.components.T:
            type_labels.append(str(component))
            property_labels.append('density')
            unit_labels.append('m-3')
            profiles[count, :] = self.profiles[str(component)]['density']['m-3']

            type_labels.append(str(component))
            property_labels.append('temperature')
            unit_labels.append('eV')
            profiles[count+1, :] = self.profiles[str(component)]['temperature']['eV']

            count += 2

        profiles = numpy.swapaxes(profiles, 0, 1)
        row_index = [i for i in range(len(self.profiles))]
        column_index = pandas.MultiIndex.from_arrays([type_labels, property_labels, unit_labels],
                                                     names=['type', 'property', 'unit'])
        return pandas.DataFrame(data=profiles, columns=column_index, index=row_index)
