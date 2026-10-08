import numpy as np
import scipy.constants as sc


'''
Sources:
[1] Janev et.al. IAEA-APID-4 (1993) https://inis.iaea.org/search/search.aspx?orig_q=RN:25024275
[2] O’Mullane, Martin. "Review of proton impact driven ionisation from the excited levels in neutral hydrogen beams." (2009).
    https://www.adas.ac.uk/notes/adas_c09-01.pdf
[3] Johnson, L. C. "Approximations for collisional and radiative transition rates in atomic hydrogen."
    The Astrophysical Journal 174 (1972): 227. https://adsabs.harvard.edu/pdf/1972ApJ...174..227J
[4] Delabie, E., et al. "Consistency of atomic data for the interpretation of beam emission spectra."
    Plasma Physics and Controlled Fusion 52.12 (2010): 125008.
[5] Marchuk, O., et al. "Review of atomic data needs for active charge-exchange spectroscopy on ITER."
    Review of scientific instruments 79.10 (2008): 10F532.
'''


class HydrogenicData:

    def __init__(self):
        self.levels = ['1n', '2n', '3n', '4n', '5n', '6n', '7n', '8n', '9n', '10n']
        self.description = 'This class provides cross-sections for n-resolved H \
                            interactions with electrons, protons, He, Li, Be, B, C, O and \
                            ions with arbitrary charge and mass. Based primarily \
                            on Janev et.al. IAEA-APID-4 (1993), with some openly \
                            available corrections from ADAS.'

        self.__excitation = {-1: self.__ex_e,
                             1: self.__ex_p,
                             2: self.__ex_He,
                             4: self.__ex_Be,
                             'Z': self.__ex_Z}
        self.__ionization = {-1: self.__ion_e,
                             1: self.__ion_p,
                             2: self.__ion_He,
                             4: self.__ion_Be,
                             5: self.__ion_B,
                             6: self.__ion_C,
                             8: self.__ion_O,
                             'Z': self.__ion_Z}
        self.__charge_ex = {1: self.__cx_p,
                            2: self.__cx_He,
                            4: self.__cx_Be,
                            5: self.__cx_B,
                            6: self.__cx_C,
                            8: self.__cx_O,
                            'Z': self.__cx_Z}
        self.__eloss = {-1: self.__ion_e,
                        'Z': self.__eloss_Z}
        self.__trans_type = {'ex': self.__excitation,
                             'ion': self.__ionization,
                             'cx': self.__charge_ex,
                             'eloss': self.__eloss}

        self.spontaneous_trans = np.array([[0.0000e+00, 4.6986e+08, 5.5751e+07, 1.2785e+07, 4.1250e+06, 1.6440e+06],
                                           [0.0000e+00, 0.0000e+00, 4.4101e+07, 8.4193e+06, 2.5304e+06, 9.7320e+05],
                                           [0.0000e+00, 0.0000e+00, 0.0000e+00, 8.9860e+06, 2.2008e+06, 7.7829e+05],
                                           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 2.6993e+06, 7.7110e+05],
                                           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0254e+06],
                                           [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00]],
                                          dtype='float')

        self.__CROSS_EQ = {

            '10': lambda e, par: 5.984e-16/e*(par[0]+par[1]/(e/10.2)+par[2]/(e/10.2)**2 +
                                              par[3]/(e/10.2)**3+par[4]/(e/10.2)**4+par[5]*np.log(e/10.2)) *
                                              (0.5*np.tanh(1e4*(e-12.23))+0.5) +                                        #swithching bettween two functions with the help of
                                              (0.5*np.tanh(-1e4*(e-12.23))+0.5) *                                       #tanh weighing functions
                                              1e-16*(0.141 + 0.129*(e-10.2)),

            '11': lambda e, par: 5.984e-16/e*((e-par[6])/e)**par[5]*(par[0]+par[1]/(e/par[6])+par[2]/(e/par[6])**2 +
                                                                     par[3]/(e/par[6])**3+par[4]*np.log(e/par[6])),

            '12': lambda e, par: 1.76e-16*par[0]**2/(par[2]*e/par[6])*(1-np.exp(-1*par[3]*par[2]*e/par[6])) *
                                        (par[4]*(np.log(e/par[6])+1/(2*e/par[6])) +
                                         (par[5]-par[4] * np.log(2*par[0]**2/par[2]))*(1-1/(e/par[6]))),

            '13': lambda e, par: 1e-13/(e*par[6])*(par[0]*np.log(e/par[6])+par[1]*(1-par[6]/e) +
                                                   par[2]*(1-par[6]/e)**2+par[3]*(1-par[6]/e)**3 +
                                                   par[4]*(1-par[6]/e)**4+par[5]*(1-par[6]/e)**5),

            '14': lambda e, par: 1.76e-16*par[0]**2/(e/par[4])*(1-np.exp(-par[1]*e/par[4])) *
                                (par[2]*np.log(e/par[4])+(par[3]-par[2]*np.log(2*par[0]**2)) * (1-1/(e/par[4]))**2),

            '15': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e+par[3] *
                                               np.exp(-par[4]*e)/e**par[5]+par[6]*np.exp(-par[7]/e) /
                                               (1+par[8]*e**par[9])),

            '16': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                               par[3]*np.exp(-par[4]*e)/(e**par[5]+par[6]*e**par[7])),

            '17': lambda e, par: (6/par[8])**3*1e-16*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                                             par[3]*np.exp(-par[4]*e)/(e**par[5]+par[6]*e**par[7])),

            '18': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                               par[3]*np.exp(-par[4]*e)/e**par[5]),

            '19': lambda e, par: par[6]*1e-16*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                                      par[3]*np.exp(-par[4]*e)/e**par[5]),

            '110': lambda e, par: par[0]*par[1]**4/e*(par[2]*par[3]*par[4]+par[5]*par[6]*par[7]),

            '111': lambda e, par: (par[8]/3)**4*1e-16*par[0]*(np.exp(-par[1]/e) *
                                                                        np.log(1+par[2]*e)/e +
                                                                        par[3]*np.exp(-par[4]*e) /
                                                              (e**par[5]+par[6]*e**par[7])),

            '112': lambda e, par: 1e-16*par[0]*np.log(par[1]/e+par[5])/(1+par[2]*e+par[3]*e**3.5 + par[4]*e**5.4),

            '113': lambda e, par: par[4]**4*1e-16*par[0] * np.log(par[1]/e+par[3]) / (1+par[2] * e+3.0842e-6*e**3.5+1.1832e-10*e**5.4),

            '115': lambda e, par: par[7]*1e-16*par[6]*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                                              par[3]*np.exp(-par[4]*e)/e**par[5]),

            '116': lambda e, par: (5/par[8])**3*par[7]*1e-16*par[6]*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                                                            par[3]*np.exp(-par[4]*e)/e**par[5]),

            '117': lambda e, par: par[8]*par[7]*1e-16*par[6]*par[0]*(np.exp(-par[1]/e)*np.log(1+par[2]*e)/e +
                                                                     par[3]*np.exp(-par[4]*e)/e**par[5]),

            '118': lambda e, par: par[8]*par[7]*8.86e-17*par[0]**4/e*(par[1]*par[2]*par[3]+par[4]*par[5]*par[6]),

            '119': lambda e, par: np.exp(-par[1]*par[6]/e)*par[0]**4*3.52e-16*par[6]**2/e*(par[2]*(np.log(e/(par[5]**2-e)) -
                                                                                                   e/par[5]**2)+par[3]-par[4]/e),

            '120': lambda e, par: par[7]*1e-16*par[0]*np.log(par[1]/par[6]+par[2])/(1+par[3]*par[6]+par[4]*par[6]**3.5+par[7]*par[6]**5.4),

            '121': lambda e, par: par[2]**4*par[3]*7.04e-16*par[0]/(par[4]**3.5*(1+par[1]*par[4]**2)
                                                                    )*(1-np.exp(-2*par[4]**3.5*(1+par[1]*par[4]**2)/3*par[0])),

            '122': lambda e, par: par[7]**4*par[6]*1e-16*par[0]*np.log(par[1]/e+par[2])/(1+par[3]*e+par[4]*e**3.5+par[6]*e**5.4),

            '129': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/e)/(1+par[2]*e**par[3]+par[4]*e**3.5 +
                                                                   par[5]*e**5.4)+par[6]*np.exp(-par[7]*e)/e**par[8]),

            '130': lambda e, par: 7.04e-16*par[4]**4*par[0]*(1-np.exp(-4/(3*par[0])*(1+e**par[1]+par[2]*e**3.5 +
                                                                                     par[3]*e**5.4)))/(1+e**par[1]+par[2]*e**3.5+par[3]*e**5.4),

            '131': lambda e, par: 1e-16*par[0]*(np.exp(-par[1]/e**par[7])/(1+par[2]*e**2+par[3]*e**par[4]+par[5]*e**par[6])
                                                + par[8]*np.exp(-par[9]*e)/e**par[10]),

            '132': lambda e, par: par[8]**4*1e-16*par[0]*(e**par[1]*np.exp(-par[2]*e)/(1+par[3]*e**par[4]) +
                                                          par[5]*np.exp(-par[6]/e)*np.log(1+par[7]*e)/e)
        }

    def get_cross_section(self, transition, energy_grid):
        trans = {'name': transition.name,
                 'target': (transition.target.charge, transition.target.atomic_number, transition.target.mass_number),
                 'from_level': transition.from_level, 'to_level': transition.to_level,
                 'target_mass': transition.target.mass_number,
                 'target_charge': transition.target.charge}
        if trans['name'] == 'de-ex':
            trans['name'] = 'ex'
            trans['from_level'] = transition.to_level
            trans['to_level'] = transition.from_level
        if trans['target_charge'] in self.__trans_type[trans['name']]:
            cross_function = self.__trans_type[trans['name']][trans['target_charge']]
        else:
            cross_function = self.__trans_type[trans['name']]['Z']
        return cross_function(trans, energy_grid)

    def __get_Johnson_osc_coeff(self, n):  # [g0, g1, g2]
        if n == 1:
            return [1.1330, -0.4059, 0.0714]
        if n == 2:
            return [1.0785, -0.2319, 0.02947]
        if n >= 3:
            return [0.9935+0.2328/n-0.1296/n**2, -1/n*(0.6282-0.5598/n+0.5299/n**2), -1/n**2*(0.3887-1.181/n+1.470/n**2)]

    def __get_Johnson_osc_str(self, n, m):
        x = 1-(n/m)**2
        g0, g1, g2 = self.__get_Johnson_osc_coeff(n)
        g = g0+g1/x+g2/x**2
        return 32/(3*3**0.5*np.pi)*n/m**3/x**3*g

    def __get_A(self, n):
        g0, g1, g2 = self.__get_Johnson_osc_coeff(n)
        g = g0/3+g1/4+g2/5
        return 32*n*g/(3*3**0.5*np.pi)

    def __get_chi(self, q):
        return 2**(0.5238*(1-(2/q)**0.5))

    def get_deex_rate(self, ex_rate, T, transition):
        g1 = int(transition.to_level)**2
        g2 = int(transition.from_level)**2
        if tuple(transition.target) == (-1, 0, 0):
            deltaE = 13.605693122994*(1/g1-1/g2)
            return ex_rate*g1/g2*np.exp(deltaE/T)
        else:
            return ex_rate*g1/g2

    def __C(self, z, z1):
        return z**2*np.log(1+2*z/3)/(2*z1+3*z/2)

###################################################################################################################

    def __ex_e(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        m = int(trans['to_level'])

        if n == 1 and m == 2:
            par = [1.4182, -20.877, 49.735, -46.249, 17.442, 4.4979]
            eq = '10'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 1 and m == 3:
            par = [0.42956, -0.58288, 1.0693, 0.0, 0.75448, 0.38277, 12.09]
            eq = '11'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 1 and m == 4:
            par = [0.24846, 0.19701, 0.0, 0.0, 0.243, 0.41844, 12.75]
            eq = '11'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 1 and m == 5:
            par = [0.13092, 0.23581, 0.0, 0.0, 0.11508, 0.45929, 13.06]
            eq = '11'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 1 and m > 5:  # Corrected according to [3]
            y = 1-(1/m)**2
            f = self.__get_Johnson_osc_str(1, m)
            b = -0.603
            B = 4/(m**3*y**2)*(1+4/(3*y)+b/y**2)
            par = [1, m, y, 0.45, 2*f/y, B, 13.6*(1-1/m**2)]
            eq = '12'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 2 and m == 3:
            par = [5.2373, 119.25, -595.39, 816.71, 38.906, 1.3196, 1.889]
            eq = '11'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n > 1 and m > 3:
            y = 1-(n/m)**2
            f = self.__get_Johnson_osc_str(n, m)
            b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
            par = [n, m, y, 1.94*n**(-1.57), 2*n**2*f/y, 4*n**4/(m**3*y**2)*(1+4/(3*y)+b/y**2), 13.6*(1/n**2-1/m**2)]
            eq = '12'
            return self.__CROSS_EQ[eq](energy_grid, par)

    def __ex_p(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        m = int(trans['to_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1 and m == 2:
            par = [34.433, 44.507, 0.56870, 8.5476, 7.8501, -9.2217, 1.8020e-2, 1.6931, 1.9422e-3, 2.9068]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 3:
            par = [6.1950, 35.773, 0.54818, 5.5162e-3, 0.291114, -4.5264, 6.0311, -2.0679]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 4:
            par = [2.0661, 34.975, 0.91213, 5.133e-4, 0.28953, -2.2849, 0.11528, -4.8970]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 5:
            par = [1.2449, 32.291, 0.21176, 3.0826e-4, 0.31063, -2.4161, 0.024664, -6.3726]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 6:
            par = [0.63771, 37.174, 0.39265, 3.2949e-4, 0.25757, -2.2950, 0.050796, -5.5986]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m > 6:
            par = [0.63771, 37.174, 0.39265, 3.2949e-4, 0.25757, -2.2950, 0.050796, -5.5986, m]
            eq = '17'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 3:
            par = [394.51, 21.606, 0.62426, 0.013597, 0.16565, -0.8949]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 4:
            par = [50.744, 19.416, 4.0262, 0.014398, 0.31584, -1.4799]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 5:
            par = [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m > 5 and m < 11:
            ratio = {6: 0.4610, 7: 0.2475, 8: 0.1465, 9: 0.0920, 10: 0.0605}
            par = [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, ratio[m]]
            eq = '19'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 4:
            par = [1247.5, 11.319, 2.6235, 0.068781, 0.521176, -1.2722]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 5:
            par = [190.59, 11.096, 2.9098, 0.073307, 0.54177, -1.2894]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 6:
            par = [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m > 6 and m < 11:
            ratio = {7: 0.4670, 8: 0.2545, 9: 0.1540, 10: 0.10}
            par = [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, ratio[m]]
            eq = '19'
            return self.__CROSS_EQ[eq](e, par)

        if n > 3:
            eps = energy_grid/25e3
            s = m-n
            D = np.exp(-1/(n*m*eps**2))
            zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
            zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
            y = 1/(1-D*np.log(18*s)/(4*s))
            par = [8.8e-17, n, 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s),
                   D, np.log((1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps)),
                   (1-0.3*s*D/(n*m))**(1+2*s), 0.5*(eps*n**2/(m-1/m))**3,
                   self.__C(zm, y)-self.__C(zp, y)]
            eq = '110'
            return self.__CROSS_EQ[eq](eps, par)

    def __ex_He(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        m = int(trans['to_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1 and m == 2:
            par = [177.69, 64.506, 0.10807, 2.1398e-4, 0.73358, -2.9773, 7.5603e-2, 18.997, 2.4352e-3, 3.4085]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 3:
            par = [18.775, 73.938, 3.2231, 1.2879e-4, 0.75301, -4.1638, 2.366e-1, 20.927, 1.6636e-3, 3.6319]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 4:
            par = [5.5094, 68.504, 12.621, 7.7669e-5, 0.53813, -4.1788, 4.0349e-2, 16.213, 5.4493e-9, 9.5011]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 5:
            par = [4.9796, 64.582, 0.10588, 2.8878e-5, 0.15531, -2.4161, 1.6389e-3, -6.3726]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 6:
            par = [2.55080, 74.348, 0.19625, 3.357e-5, 0.12878, -2.295, 5.144e-3, -5.5986]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m > 6:
            par = [2.55080, 74.348, 0.19625, 3.357e-5, 0.12878, -2.295, 5.144e-3, -5.5986, m]
            eq = '17'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 3:
            par = [1864.3, 19.395, 0.13899, 2.4502e-3, 0.2966, -1.7558]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 4:
            par = [246.18, 27.764, 0.39876, 1.9381e-3, 0.23304, -1.7165]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 5:
            par = [73.056, 37.946, 1.4528, 2.4601e-3, 0.15855, -1.4775]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m > 5:
            ratio = {6: 0.461, 7: 0.2475, 8: 0.1465, 9: 0.0920, 10: 0.0605}[m]
            par = [73.056, 37.946, 1.4528, 2.4601e-3, 0.15855, -1.4775, ratio]
            eq = '19'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 4:
            par = [4990, 22.638, 1.3118, 0.014239, 0.260596, -1.2722]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 5:
            par = [762.36, 22.192, 1.4549, 0.014996, 0.27088, -1.2894]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 6:
            par = [253.89, 23.014, 2.1708, 0.01596, 0.2673, -1.2881]
            eq = '18'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m > 6:
            ratio = {7: 0.467, 8: 0.2545, 9: 0.154, 10: 0.1}[m]
            par = [253.89, 23.014, 2.1708, 0.01596, 0.2673, -1.2881, ratio]
            eq = '19'
            return self.__CROSS_EQ[eq](e, par)

        if n > 3:
            eps = e/50
            s = m-n
            D = np.exp(-1/(n*m*eps**2))
            A = 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s)
            L = np.log((1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps))
            F = (1-0.3*s*D/(n*m))**(1+2*s)
            G = 0.5 * (eps*n**2/(m-1/m))**3
            zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
            zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
            y = 1/(1-D*np.log(18*s)/(4*s))
            par = [1.76e-16, n, A, D, L, F, G, self.__C(zm, y)-self.__C(zp, y)]
            eq = '110'
            return self.__CROSS_EQ[eq](eps, par)

    def __ex_Be(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        m = int(trans['to_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1 and m == 2:
            par = [961.8, 70.386, 1.4e-2, 1.208e-6, 0.31849, -3.3516, 7.209e-3, 30.194, 2.478e-8, 8.4206]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 3:
            par = [95.728, 90.975, 0.2198, 9.6493e-7, 0.36229, -4.1912, 3.587e-2, 28.681, 5.1187e-9, 9.1415]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 4:
            par = [32.190, 86.307, 0.27496, 1.1909e-6, 0.38748, -4.3014, 3.2598e-2, 26.395, 1.5743e-8, 8.6415]
            eq = '15'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 5:
            par = [19.918, 129.16, 5.2941e-2, 2.7053e-6, 7.7658e-2, -2.4161, 5.9445, -6.3726]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 6:
            par = [10.203, 148.7, 9.8162e-2, 3.419e-6, 6.4392e-2, -2.295, 4.9522, -5.5986]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m > 6:
            par = [10.203, 148.7, 9.8162e-2, 3.419e-6, 6.4392e-2, -2.295, 4.9522, -5.5986, m]
            eq = '17'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__ex_Z(trans, energy_grid)

    def __ex_Z(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        m = int(trans['to_level'])
        q = trans['target_charge']
        e = energy_grid/1e3/trans['target_mass']/q
        chi = self.__get_chi(q)

        if n == 1 and m == 2:
            par = [38.738, 37.033, 0.39862, 7.7582e-5, 0.25402, -2.7418, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 3:
            par = [4.3619, 57.451, 21.001, 2.3292e-4, 0.083130, -2.2364, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m == 4:
            par = [1.3730, 60.710, 31.797, 2.0207e-4, 0.082513, -2.3055, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 1 and m > 4:
            par = [0.56565, 67.333, 55.290, 2.1595e-4, 0.081624, -2.1971, chi, q, m]
            eq = '116'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 3:
            par = [358.03, 25.283, 1.4726, 0.014398, 0.12207, -0.86210, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 4:
            par = [50.744, 19.416, 4.0262, 0.014398, 0.31584, -1.4799, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m == 5:
            par = [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2 and m > 5 and m < 11:
            scaling = {6: 0.4610, 7: 0.2475, 8: 0.1465, 9: 0.0920, 10: 0.0605}
            par = [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, chi, q, scaling[m]]
            eq = '117'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 4:
            par = [1247.5, 11.319, 2.6235, 0.068781, 0.521176, -1.2722, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 5:
            par = [190.59, 11.096, 2.9098, 0.073307, 0.54177, -1.2894, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m == 6:
            par = [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, chi, q]
            eq = '115'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3 and m > 6 and m < 11:
            scaling = {7: 0.4670, 8: 0.2545, 9: 0.1540, 10: 0.1}
            par = [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, chi, q, scaling[m]]
            eq = '117'
            return self.__CROSS_EQ[eq](e, par)

        if n > 3:
            eps = e/25
            s = m-n
            A = 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s)
            D = np.exp(-1/(n*m*eps**2))
            L = np.log((1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps))
            F = (1-0.3*s*D/(n*m))**(1+2*s)
            G = 0.5*(eps*n**2/(m-1/m))**3
            zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
            zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
            y = 1/(1-D*np.log(18*s)/(4*s))
            chi = 2**(0.322*(1-(2/q)**0.5))
            par = [n, A, D, L, F, G, self.__C(zm, y)-self.__C(zp, y), chi, q]
            eq = '118'
            return self.__CROSS_EQ[eq](eps, par)

    def __ion_e(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])

        if n == 1:
            par = [0.18450, -0.032226, -0.034539, 1.4003, -2.8115, 2.2986, 13.6]
            eq = '13'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 2:
            par = [0.14784, 0.0080871, -0.062270, 1.9414, -2.1980, 0.95894, 3.4]
            eq = '13'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n == 3:
            par = [0.058463, -0.051272, 0.85310, -0.57014, 0.76684, 0.0, 1.511]
            eq = '13'
            return self.__CROSS_EQ[eq](energy_grid, par)

        if n > 3:  # Corrected according to [3]
            b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
            A = self.__get_A(n)
            eq = '14'
            par = [n, 1.94*n**(-1.57), A, 2/3*n**2*(5+b), 13.6/n**2]
            return self.__CROSS_EQ[eq](energy_grid, par)

    def __ion_p(self, trans, energy_grid):
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:   # [1]
            par = [12.899, 61.897, 9.2731e+3, 4.9749e-4, 3.9890e-2, -1.590, 3.1834, -3.7154]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2:  # [2]
            par = [3.933e-3, 1.8188, 1.887e-2, 6.7489e-3, 1.3768, 6.8852e2, 9.6435e1, 5.6515e23, n]
            e_red = e*n**2
            eq = '132'
            return self.__CROSS_EQ[eq](e_red, par)

        if n == 3:  # [2]
            par = [1.1076e-2, 1.6197, 6.7154e-3, 5.1188e-3, 1.8549, 2.3696e+2, 7.8286e1, 1.0926e23, n]
            e_red = e*n**2
            eq = '132'
            return self.__CROSS_EQ[eq](e_red, par)

        if n == 4:  # [2]
            par = [1.1033e-2, 1.6281, 5.5955e-3, 7.2023e-3, 1.7358, 2.2755e2, 8.6339e1, 3.9151e29, n]
            e_red = e*n**2
            eq = '132'
            return self.__CROSS_EQ[eq](e_red, par)

        if n >= 5:  # [2]
            par = [1.1297e-2, 1.8685, 1.5038e-2, 1.1195e-1, 1.0538, 8.6096e2, 8.9939e1, 1.9249e4, n]
            e_red = e*n**2
            eq = '132'
            return self.__CROSS_EQ[eq](e_red, par)

    def __ion_He(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [40.498, 112.61, 1.5496e6, 1.4285e-5, 4.1163e-2, -2.6347, 4.0589, -5.9204]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2:
            par = [109.01, 26.473, 1.0224e+6, 5.7286e-3, 0.040151, -2.4092, 0.014897, -0.23786]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3:
            par = [250.1, 7.9018, 2.1448e6, 0.33041, 0.093012, -0.49446, 0.63357, -2.7261]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n > 3:
            e_red = (3/n)**2*e
            eq = '111'
            par = [250.1, 7.9018, 2.1448e+6, 0.33041, 0.093012, -0.49446, 0.63357, -2.7621, n]
            return self.__CROSS_EQ[eq](e_red, par)

    def __ion_Be(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [306.63, 178.22, 62.033, 3.1376e-5, 1.3455e-2, -1.6452, 57.117, -3.53383]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__ion_Z(trans, energy_grid)

    def __ion_B(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [351.52, 233.63, 3.2952e3, 5.3787e-6, 1.8834e-2, -2.2064, 7.2074, -3.78664]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__ion_Z(trans, energy_grid)

    def __ion_C(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [438.36, 327.1, 1.4444e5, 3.5212e-3, 8.3031e-3, -0.63731, 1.9116e4, -3.1003]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__ion_Z(trans, energy_grid)

    def __ion_O(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [1244.44, 249.36, 30.892, 9.0159e-4, 7.7885e-3, -0.71309, 3.2918e3, -2.7541]
            eq = '16'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__ion_Z(trans, energy_grid)

    def __ion_Z(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        q = trans['target_charge']
        e = energy_grid/1e3/trans['target_mass']/25
        par = [n, 0.76, 0.283, 4.04, 0.662, 137, q]
        eq = '119'
        return self.__CROSS_EQ[eq](n**2*e, par)

    def __cx_p(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [3.2345, 235.88, 0.038371, 3.8068e-6, 1.1832e-10, 2.3713]
            eq = '112'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2:
            e_red = e*n**2
            par = [0.92750, 6.5040e+3, 1.3405e-2, 20.699, n]
            eq = '113'
            return self.__CROSS_EQ[eq](e_red, par)

        if n == 3:
            e_red = e*n**2
            par = [0.37271, 2.7645e+6, 1.5720e-3, 1.4857e+3, n]
            eq = '113'
            return self.__CROSS_EQ[eq](e_red, par)

        if n >= 4:
            e_red = e*n**2
            par = [0.21336, 1.0e+10, 1.8184e-3, 1.3426e+6, n]
            eq = '113'
            return self.__CROSS_EQ[eq](e_red, par)

    def __cx_He(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [17.438, 2.1263, 2.1401e-3, 1.6498, 2.6259e-6, 2.4226e-11, 15.665, 7.9193, -4.4053]
            eq = '129'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2:
            par = [88.508, 0.78429, 3.2903e-2, 1.7635, 7.3265e-5, 1.4418e-8, 0.80478, 0.22349, -0.68604]
            eq = '129'
            return self.__CROSS_EQ[eq](e, par)

        if n > 2:
            par = [2.0032e2, 1.4591, 2.0384e-4, 2e-9, n]
            eq = '130'
            return self.__CROSS_EQ[eq](e, par)

    def __cx_Be(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [19.952, 0.20036, 1.7295e-4, 3.6844e-11, 5.0411, 2.4689e-8, 4.0761, 0.88093, 0.94361, 0.14205, -0.42973]
            eq = '131'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__cx_Z(trans, energy_grid)

    def __cx_B(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [31.226, 1.1442, 4.8372e-8, 3.0961e-10, 4.7205, 6.2844e-7, 3.1297, 0.12556, 0.30098, 5.9607e-2, -0.57923]
            eq = '131'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__cx_Z(trans, energy_grid)

    def __cx_C(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [418.18, 2.1585, 3.4808e-4, 5.3333e-9, 4.6556, 0.33755, 0.81736, 0.27874, 1.8003e-6, 7.1033e-2, 0.53261]
            eq = '131'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__cx_Z(trans, energy_grid)

    def __cx_O(self, trans, energy_grid):   # [1]
        n = int(trans['from_level'])
        e = energy_grid/1e3/trans['target_mass']

        if n == 1:
            par = [54.535, 0.27486, 1.0104e-7, 2.0745e-9, 4.4416, 7.6555e-3, 1.1134, 1.1621, 0.15826, 3.6613e-2, 3.9741e-2]
            eq = '131'
            return self.__CROSS_EQ[eq](e, par)

        if n > 1:
            return self.__cx_Z(trans, energy_grid)

    def __cx_Z(self, trans, energy_grid):   # Modified based on [4,5]
        n = int(trans['from_level'])
        q = trans['target_charge']
        e = energy_grid/1e3/trans['target_mass']*(1/q**(0.5))*n**2

        if n == 1:
            par = [3.2345, 2.3588e+2, 2.3713, 3.8371e-2, 3.8068e-6, 1.1832e-10, q, n]
            eq = '122'
            return self.__CROSS_EQ[eq](e, par)

        if n == 2:
            par = [9.2750e-1, 6.5040e3, 2.0699e1, 1.3405e-2, 3.0842e-6, 1.1832e-10, q, n]
            eq = '122'
            return self.__CROSS_EQ[eq](e, par)

        if n == 3:
            par = [3.7271e-1, 2.7645e6, 1.4857e3, 1.5720e-3, 3.0842e-6, 1.1832e-10, q, n]
            eq = '122'
            return self.__CROSS_EQ[eq](e, par)

        if n > 3:
            par = [2.1336e-1, 1e10, 1.3426e6, 1.8184e-3, 3.0842e-6, 1.1832e-10, q, n]
            eq = '122'
            return self.__CROSS_EQ[eq](e, par)

    def __eloss_Z(self, trans, energy_grid):
        if trans['target_charge'] in self.__trans_type['ion']:
            sigma_ion = self.__trans_type['ion'][trans['target_charge']](trans, energy_grid)
            sigma_cx = self.__trans_type['cx'][trans['target_charge']](trans, energy_grid)
        else:
            sigma_ion = self.__trans_type['ion']['Z'](trans, energy_grid)
            sigma_cx = self.__trans_type['cx']['Z'](trans, energy_grid)
        return sigma_ion+sigma_cx
