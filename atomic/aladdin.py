import pycollisiondb
from pycollisiondb.pycollisiondb import PyCollision
from scipy.interpolate import interp1d

class FitFunction(object):
    def __init__(self, data):
        self.coefficients = self.assign_from_fit(data, 'coeffs')
        self.elo = self.assign_from_fit(data, 'Elo')
        self.ehi = self.assign_from_fit(data, 'Ehi')
        self.name = self.assign_from_fit(data, 'func')
        self.precision = self.assign_from_fit(data, 'fit_unc_perc')

    @staticmethod
    def assign_from_fit(data, tag):
        try:
            return data[tag]
        except KeyError:
            return None

class AladdinCrossSection(object):
    def __init__(self, data):
        if not isinstance(data, pycollisiondb.pycoll_ds.PyCollDataSet):
            raise TypeError('The expected type from aladdin API is PyCollision')
        self.energy = data.x
        self.cross_section = data.y
        self.len = data.n
        self.reaction = self.assign_data(data, tag='reaction')
        self.type = self.assign_data(data, tag='process_types')
        self.uncertainty = float(self.assign_from_json(data, tag='unc_perc'))
        #self.threshold = float(self.assign_from_json(data, tag='threshold'))
        self.function = self.assign_from_json(data, tag='data_from_fit')
        if self.function:
            self.fit = FitFunction(data.metadata['json_data']['fit'])

    @staticmethod
    def assign_from_json(data, tag):
        try:
            return data.metadata['json_data'][tag]
        except KeyError:
            return None

    @staticmethod
    def assign_data(data, tag):
        try:
            return data.metadata[tag]
        except KeyError:
            return None

class AladdinData(object):
    def __init__(self, url_source = 'aladdin'):
        if isinstance(url_source, str):
            if url_source == 'aladdin':
                self.api_url = 'https://db-amdis.org/aladdin2/api/'
            elif url_source == 'collisiondb':
                self.api_url = 'https://db-amdis.org/collisiondb/api/'
            else:
                raise ValueError('The requested data source is not supported: ' + url_source)
        else:
            raise TypeError('The required data type for url specification is <str>.')

    @staticmethod
    def _set_sign(particle):
        if particle.charge >= 1:
            return '+'
        else:
            return '-'

    def rod_to_pycoll(self, transition):
        reactants = str(transition.projectile) + ' ' + transition.from_level + ' + ' + \
                    str(transition.target) + self._set_sign(transition.target)

        if transition.name == 'ex' or transition.name == 'de-ex':
            products = str(transition.projectile) + ' ' + transition.to_level + ' + ' + \
                       str(transition.target) + self._set_sign(transition.target)
        elif transition.name == 'ion':
            products = str(transition.projectile) + '+ + ' + \
                       str(transition.target) + self._set_sign(transition.target) + ' + ' + 'e-'
        elif transition.name == 'cx':
            products = str(transition.projectile) + '+ + ' + \
                       str(transition.target)
        else:
            products = str(transition.projectile) + '+ + '
        return reactants + ' -> ' + products

    def get_data_from_iaea(self, pycoll_reaction):
        pycoll_raw = PyCollision.get_datasets(query = {'reaction_text':pycoll_reaction, 'data_type':'cross section'},
                                              API_URL=self.api_url)
        return AladdinCrossSection(pycoll_raw.datasets[list(pycoll_raw.datasets.keys())[0]])

    def get_cross_section(self, transition, energy_grid):
        reaction = self.rod_to_pycoll(transition)
        aladdin_data = self.get_data_from_iaea(pycoll_reaction=reaction)
        cross_function = interp1d(aladdin_data.energy, aladdin_data.cross_section,
                                  kind='cubic', fill_value='extrapolate')
        return cross_function(energy_grid)
