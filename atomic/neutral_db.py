import os
import utility.convert as uc
from utility.getdata import GetData
from utility.exceptions import RenateNotValidTransitionError


class NeutralDB(object):
    def __init__(self, param=None, components=None, resolved=None):
        self.param = param
        self.__set_atomic_resolution(resolved=resolved)
        self.__load_neutral_cross_section(components=components)
        self.__get_atomic_levels()

    def __set_atomic_resolution(self, resolved):
        projectile = self.param.getroot().find('body').find('beamlet_species').text
        if resolved is None:
            if projectile == 'H' or projectile == 'D' or projectile == 'T':
                self.resolved = 'bundled_n'
            elif projectile == 'Li' or projectile == 'Na':
                self.resolved = 'nl'
        else:
            self.resolved = resolved

    def __create_neutral_cross_section_db(self, components):
        self.neutral_cross_sections = {}
        self.neutral_target_count = len([comp for comp in components['q'] if int(comp) == 0])
        for index in range(self.neutral_target_count):
            self.neutral_cross_sections.update({'neutral'+str(index+1): None})

    def __set_neutral_cross_section_path(self, target='H'):
        file_name = target + '_' + self.resolved + '_' + \
                    self.param.getroot().find('body').find('beamlet_energy').text + '.txt'

        return os.path.join('atomic_data', self.param.getroot().find('body').find('beamlet_species').text,
                            'cross_sections', 'neutral', file_name)

    def __identify_neutral_target(self, component):
        if isinstance(component['Molecule'], str):
            return component['Molecule']
        else:
            if component['Z'] == 1 and component['A'] == 1:
                return 'H'
            elif component['Z'] == 1 and component['A'] == 2:
                return 'D'
            elif component['Z'] == 1 and component['A'] == 3:
                return 'T'
            else:
                raise NotImplementedError('The requested atomic gas is not supported yet.')

    def __load_neutral_cross_section(self, components):
        self.__create_neutral_cross_section_db(components=components)
        for index in range(self.neutral_target_count):
            name = 'neutral'+str(index+1)
            target = self.__identify_neutral_target(component=components.T[name])
            path = self.__set_neutral_cross_section_path(target=target)
            self.neutral_cross_sections[name] = GetData(data_path_name=path, data_format="array").data

    def __get_atomic_levels(self):
        for index in range(self.neutral_target_count):
            atomic_levels = self.neutral_cross_sections['neutral'+str(index+1)].shape[0]
            if not hasattr(self, 'atomic_level'):
                self.atomic_levels = atomic_levels
            elif self.atomic_levels > atomic_levels:
                self.atomic_levels = atomic_levels

    def get_neutral_impact_loss(self, target, from_level):
        return uc.convert_from_cm2_to_m2(self.neutral_cross_sections[target][from_level, from_level])

    def get_neutral_impact_transition(self, target, from_level, to_level):
        if from_level == to_level:
            raise RenateNotValidTransitionError('The requested atomic transition is not valid. '
                                                'Requested transition is an ionization.')
        return uc.convert_from_cm2_to_m2(self.neutral_cross_sections[target][from_level, to_level])
