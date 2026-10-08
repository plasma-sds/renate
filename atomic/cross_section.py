import numpy as np
import scipy.constants as sc
from atomic.aladdin import AladdinData
from atomic.hydrogenic_cross_sections import HydrogenicData


class CrossSection:

    CROSSEC_SOURCES = {'aladdin': [], 'collisiondb':[], 'internal': ['hydrogenic']}

    def __new__(cls, source, projectile=None):
        if not isinstance(source, str):
            raise TypeError('A string is expected to decide which data source to pursue.\n'+CrossSection.GetAvailableData())
        if source == 'aladdin' or source == 'collisiondb':
            return AladdinData(url_source=source)
        elif source == 'internal':
            if not isinstance(projectile, str):
                raise TypeError('The data source requires the projectile parameter to be a valid string.\n'+CrossSection.GetAvailableData())
            if projectile == 'hydrogenic':
                return HydrogenicData()
            else:
                raise ValueError('The given projectile is not available in the data source.\n'+CrossSection.GetAvailableData())
        else:
            raise ValueError('The data source requested is not supported.\n'+CrossSection.GetAvailableData())

    @staticmethod
    def GetAvailableData():
        return('Available sources and projectiles:\n'+str(CrossSection.CROSSEC_SOURCES))
