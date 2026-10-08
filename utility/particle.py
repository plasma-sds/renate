from utility.constants import Constants
from utility.exceptions import InputError

CONST = Constants()


class Particle(object):
    def __init__(self, label='', charge=0, mass_number=0, atomic_number=0, mass=None):
        if isinstance(label, str):
            self.label = label
        else:
            raise InputError('Label is expected to be of string type.')
        try:
            atomic_number = int(atomic_number)
            mass_number = int(mass_number)
            charge = int(charge)
        except:
            raise InputError('The atomic number, mass number and charge are expected to be integers.')
        self.atomic_number = atomic_number
        self.charge = charge
        if (mass_number - self.atomic_number) >= 0:
            self.mass_number = mass_number
            self.neutron_number = self.mass_number - self.atomic_number
        if mass is None:
            if tuple(self) == (-1, 0, 0):
                self.mass = CONST.electron_mass
            else:
                self.mass = self.neutron_number*CONST.neutron_mass +\
                    self.atomic_number*CONST.proton_mass
        else:
            self.mass = mass

    def update_mass(self, mass):
        self.mass = mass

    def update_atomic_number(self, atomic_number):
        self.atomic_number = atomic_number

    def update_mass_number(self, mass_number):
        self.mass_number = mass_number
        self.neutron_number = self.mass_number - self.atomic_number

    def update_charge(self, charge):
        self.charge = charge

    def __str__(self):
        return str(self.label)

    def __iter__(self):
        for x in [self.charge, self.atomic_number, self.mass_number]:
            yield x

    def __repr__(self):
        return 'Particle: ' + str(self.label) + '\t (q,Z,A) =  (' + str(self.charge) + ',' + str(self.atomic_number) + \
               ',' + str(self.mass_number) + ')'
