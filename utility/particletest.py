import unittest

import scipy.constants as c

from utility.exceptions import InputError
from utility.particle import Particle


class ParticleTest(unittest.TestCase):

    def test_default_particle(self):
        particle = Particle()
        self.assertEqual(particle.label, '',
                         msg='Default Particle label is expected to be an empty string.')
        self.assertEqual(particle.charge, 0,
                         msg='Default Particle charge is expected to be 0.')
        self.assertEqual(particle.atomic_number, 0,
                         msg='Default Particle atomic_number is expected to be 0.')
        self.assertEqual(particle.mass_number, 0,
                         msg='Default Particle mass_number is expected to be 0.')
        self.assertEqual(particle.neutron_number, 0,
                         msg='Default Particle neutron_number is expected to be 0.')
        self.assertAlmostEqual(particle.mass, 0.0,
                               msg='Default Particle mass is expected to be 0 (no nucleons).')

    def test_label_is_stored(self):
        particle = Particle(label='H', atomic_number=1, mass_number=1)
        self.assertEqual(particle.label, 'H',
                         msg='Particle.label is expected to equal the provided label string.')

    def test_non_string_label_raises_input_error(self):
        with self.assertRaises(InputError,
                               msg='Particle is expected to raise InputError when label is not a string.'):
            Particle(label=1)

    def test_non_integer_quantum_numbers_raise_input_error(self):
        with self.assertRaises(InputError,
                               msg='Particle is expected to raise InputError when atomic_number cannot '
                                   'be converted to int.'):
            Particle(label='H', atomic_number='one', mass_number=1)
        with self.assertRaises(InputError,
                               msg='Particle is expected to raise InputError when mass_number cannot '
                                   'be converted to int.'):
            Particle(label='H', atomic_number=1, mass_number='two')
        with self.assertRaises(InputError,
                               msg='Particle is expected to raise InputError when charge cannot '
                                   'be converted to int.'):
            Particle(label='H', atomic_number=1, mass_number=1, charge='plus')

    def test_numeric_strings_are_accepted(self):
        particle = Particle(label='Li', charge='0', mass_number='7', atomic_number='3')
        self.assertEqual(particle.charge, 0,
                         msg='Numeric string charge is expected to be stored as int 0.')
        self.assertEqual(particle.mass_number, 7,
                         msg='Numeric string mass_number is expected to be stored as int 7.')
        self.assertEqual(particle.atomic_number, 3,
                         msg='Numeric string atomic_number is expected to be stored as int 3.')

    def test_proton_mass_from_nucleons(self):
        particle = Particle(label='p', charge=1, atomic_number=1, mass_number=1)
        self.assertEqual(particle.neutron_number, 0,
                         msg='A proton (Z=1, A=1) is expected to have neutron_number 0.')
        self.assertAlmostEqual(particle.mass, c.proton_mass,
                               msg='Proton mass is expected to equal scipy proton_mass.')


    def test_electron_uses_electron_mass(self):
        particle = Particle(label='e', charge=-1, atomic_number=0, mass_number=0)
        self.assertAlmostEqual(particle.mass, c.electron_mass,
                               msg='Particle with (q,Z,A)=(-1,0,0) is expected to use the electron mass.')

    def test_explicit_mass_overrides_formula(self):
        particle = Particle(label='H', atomic_number=1, mass_number=1, mass=1.67e-27)
        self.assertAlmostEqual(particle.mass, 1.67e-27,
                               msg='Explicit mass argument is expected to override the nucleon-sum formula.')

    def test_explicit_mass_overrides_electron_mass(self):
        particle = Particle(label='e', charge=-1, atomic_number=0, mass_number=0, mass=1.0)
        self.assertAlmostEqual(particle.mass, 1.0,
                               msg='Explicit mass argument is expected to override the electron-mass shortcut.')

    def test_iter_yields_charge_atomic_number_mass_number(self):
        particle = Particle(label='Na', charge=0, atomic_number=11, mass_number=23)
        self.assertEqual(tuple(particle), (0, 11, 23),
                         msg='iter(Particle) is expected to yield (charge, atomic_number, mass_number).')

    def test_str_returns_label(self):
        particle = Particle(label='T', atomic_number=1, mass_number=3)
        self.assertEqual(str(particle), 'T',
                         msg='str(Particle) is expected to equal the particle label.')

    def test_repr_contains_label_and_quantum_numbers(self):
        particle = Particle(label='Li', charge=0, atomic_number=3, mass_number=7)
        representation = repr(particle)
        self.assertIn('Li', representation,
                      msg='repr(Particle) is expected to contain the particle label.')
        self.assertIn('(q,Z,A) =  (0,3,7)', representation,
                      msg='repr(Particle) is expected to contain (q,Z,A) =  (charge,Z,A).')

    def test_update_mass(self):
        particle = Particle(label='H', atomic_number=1, mass_number=1)
        particle.update_mass(9.9e-27)
        self.assertAlmostEqual(particle.mass, 9.9e-27,
                               msg='update_mass is expected to overwrite Particle.mass.')

    def test_update_atomic_number(self):
        particle = Particle(label='H', atomic_number=1, mass_number=1)
        particle.update_atomic_number(2)
        self.assertEqual(particle.atomic_number, 2,
                         msg='update_atomic_number is expected to overwrite Particle.atomic_number.')

    def test_update_mass_number_updates_neutron_number(self):
        particle = Particle(label='H', atomic_number=1, mass_number=1)
        particle.update_mass_number(3)
        self.assertEqual(particle.mass_number, 3,
                         msg='update_mass_number is expected to overwrite Particle.mass_number.')
        self.assertEqual(particle.neutron_number, 2,
                         msg='update_mass_number is expected to set neutron_number to A-Z.')

    def test_update_charge(self):
        particle = Particle(label='H', atomic_number=1, mass_number=1, charge=0)
        particle.update_charge(1)
        self.assertEqual(particle.charge, 1,
                         msg='update_charge is expected to overwrite Particle.charge.')


if __name__ == '__main__':
    unittest.main()
