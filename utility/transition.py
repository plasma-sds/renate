from utility.particle import Particle
from utility.exceptions import InputError


class Transition(object):
    def __init__(self, projectile=Particle, target=Particle, from_level=str, to_level=None, trans=str):
        self.projectile = projectile
        self.target = target
        if isinstance(from_level, str) and isinstance(trans, str):
            self.from_level = from_level
            if trans in ['ex', 'de-ex', 'eloss', 'ion', 'cx']:
                self.name = trans
            else:
                raise InputError('The provided transition is not supported. '
                                 'Supported transitions are: ex, de-ex, eloss, ion and cx.')
            if (to_level is None) or isinstance(to_level, str):
                self.to_level = to_level
            else:
                raise InputError('The provided end state for the electron transition is not valid. Str or None is expected.')
        else:
            raise InputError('Expected input data format for <from_level>, <to_level> and <trans> to be of str type.')

    def __str__(self):
        if self.name in ['cx', 'eloss', 'ion']:
            return self.from_level + '-' + self.name
        else:
            return self.from_level + '-' + self.to_level

    def __repr__(self):
        return 'Collision of: '+str(self.projectile)+' + '+str(self.target)+' with transition: ' + \
               self.name+' | from level: '+self.from_level+' | to_level: '+str(self.to_level)
