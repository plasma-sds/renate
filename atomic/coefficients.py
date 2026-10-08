import numpy as np
from atomic import tools
from atomic import cross_section
import scipy.constants as sc

"""
Li references:
[1] Wutte et.al. ADNDT 65, 155 (1997) https://www.sciencedirect.com/science/article/abs/pii/S0092640X97907361
[2] Schweinzer et.al. ADNDT 72, 239 (1999)  https://www.sciencedirect.com/science/article/abs/pii/S0092640X9990815X
[3] Schweinzer et.al. JPB 27, 137 (1994) https://iopscience.iop.org/article/10.1088/0953-4075/27/1/017

H references:
[4] Janev et.al. IAEA-APID-4 (1993) https://inis.iaea.org/search/search.aspx?orig_q=RN:25024275
"""

Li_Schweinzer = {'e': {'2s-eloss': {'param': [5.39, 0.085, -0.004, 0.757, -0.178], 'eq': '0'},  # [1] Table p165
                       '2s-2p': {'param': [1.847, -29.466, 10.106, 128.32, -86.415, 58.696, 0.77758], 'eq': '2'},
                       '2s-3s': {'param': [3.372, 4.50, -11.393, 9.2699, 1.8822, 0, 0.32060], 'eq': '2'},
                       '2s-3p': {'param': [3.833, 2.2138, -4.3282, 4.4758, 0, 0, 0.25068], 'eq': '2'},
                       '2s-3d': {'param': [3.877, 8.20, -8.2458, -6.4716, 12.953, 0, 0.61774], 'eq': '2'},
                       '2s-4s': {'param': [4.34, 0.926, -1.1803, -1.2032, 2.0279, 0, 0.053105], 'eq': '2'},
                       '2s-4p': {'param': [4.52, 0.05455, 1.0983, -1, 0.1222, 0.15, 0.12917], 'eq': '2'},
                       '2s-4d': {'param': [4.539, 1.8676, -1.8193, 0.2679, 0, 0, 0.25478], 'eq': '2'},
                       '2s-4f': {'param': [4.54, 0.25, 3.3341, -6.5182, 4.9176, 0, 0.90017], 'eq': '2'},
                       '2p-eloss': {'param': [3.543, 0.075347, 0.1882, -0.28343, 1.5382, -1.855, 1.3411], 'eq': '1'},
                       '2p-3s': {'param': [1.525, -12.287, 7.9788, 21.593, -10.454, 10.136, 0.35903], 'eq': '2'},
                       '2p-3p': {'param': [1.986, 10.353, -22.590, 21.347, 0, 0, 0.33146], 'eq': '2'},
                       '2p-3d': {'param': [2.03, -20.091, -78.831, 306.52, -146.03, 38.767, 1.7130], 'eq': '2'},
                       '2p-4s': {'param': [2.493, -0.45472, 6.2003, -23.437, 25.556, 0.70915, 1.1250], 'eq': '2'},
                       '2p-4p': {'param': [2.673, 2.4654, -3.9177, 4.7847, 0, 0, 0.78931], 'eq': '2'},
                       '2p-4d': {'param': [2.692, 4.8531, -40.566, 83.492, -46.007, 4.5560, 0.57257], 'eq': '2'},
                       '2p-4f': {'param': [2.693, 2.1425, 24.042, -77.016, 62.941, 0.24507, 0.79098], 'eq': '2'},
                       '3s-eloss': {'param': [2.018, 5.55e-15, 0.34812, -0.29262, 0.34264, 0.99613, -0.82395],
                                    'eq': '1'},
                       '3s-3p': {'param': [0.461, -678.58, 805.29, 0, 0, 439.5, 0.78903], 'eq': '2'},
                       '3s-3d': {'param': [0.505, 96.957, -309.15, 1089.2, 0, 0, 2.4374], 'eq': '2'},
                       '3s-4s': {'param': [0.968, 18.409, -72.121, 34.575, 109.59, 0.94483, 1.2021], 'eq': '2'},
                       '3s-4p': {'param': [1.148, 8.4345, -10.284, 0, 25.887, 1.9035, 0.62176], 'eq': '2'},
                       '3s-4d': {'param': [1.167, 10.233, 407.74, -2173.8, 3286.7, 0, 3.7253], 'eq': '2'},
                       '3s-4f': {'param': [1.168, 28.327, 100.47, -421.76, 419.5, 0, 1.00], 'eq': '2'},
                       '3p-eloss': {'param': [1.557, 0.11218, 0.24966, -1.0074, 1.8549, -1.1973, 0.48164], 'eq': '1'},
                       '3p-3d': {'param': [0.044, 186.96, -1032.5, 2307.1, 0, 230.05, 2.68], 'eq': '2'},
                       '3p-4s': {'param': [0.507, -117.31, 196.06, -78.658, 0, 64.033, 0.062851], 'eq': '2'},
                       '3p-4p': {'param': [0.687, 49.0, -218.58, 392.42, -201.84, 0, 0.61809], 'eq': '2'},
                       '3p-4d': {'param': [0.706, -202.18, 335.8, -132.6, 0, 113.64, 0.28384], 'eq': '2'},
                       '3p-4f': {'param': [0.707, 115.93, -99.313, -319.37, 758.96, 0, 2.3647], 'eq': '2'},
                       '3d-eloss': {'param': [1.513, 0.02339, 0.37229, 0.011639, -3.1118, 7.7385, -4.5292], 'eq': '1'},
                       '3d-4s': {'param': [0.463, 5.80, -13.713, 54.729, -44.469, 0, 0.69859], 'eq': '2'},
                       '3d-4p': {'param': [0.643, 2.2291, -10.245, 42.434, -28.528, 2.2158, 0.21465], 'eq': '2'},
                       '3d-4d': {'param': [0.662, 43.0, -142.81, 226.64, -110.26, 0, 0.6948], 'eq': '2'},
                       '3d-4f': {'param': [0.663, -221.53, -371.82, 3313.2, -1906.1, 221.29, 3.7250], 'eq': '2'},
                       '4s-eloss': {'param': [1.05], 'eq': '3'},
                       '4s-4p': {'param': [0.18, 0.072489, 816.84, 0, 0, 992.70, 1.1539], 'eq': '2'},
                       '4s-4d': {'param': [0.199, 384.65, -489.77, 382.85, 0, 0, 0.51756], 'eq': '2'},
                       '4s-4f': {'param': [0.2, 492.35, -626.91, 490.05, 0, 0, 0.51756], 'eq': '2'},
                       '4p-eloss': {'param': [0.87], 'eq': '3'},
                       '4p-4d': {'param': [0.019, 0.11050, 126.57, 382.62, 0, 778.86, 0.92249], 'eq': '2'},
                       '4p-4f': {'param': [0.02, 538.51, -685.678, 535.99, 0, 0, 0.51756], 'eq': '2'},
                       '4d-eloss': {'param': [0.851], 'eq': '3'},
                       '4d-4f': {'param': [0.001, 0.1547, 177.198, 535.668, 0, 1090.4, 0.92249], 'eq': '2'},
                       '4f-eloss': {'param': [0.85], 'eq': '3'}},
                 '1H1+': {'2s-eloss': {},
                          '2s-2p': {},
                          '2s-3s': {},
                          '2s-3p': {},
                          '2s-3d': {},
                          '2s-4s': {},
                          '2s-4p': {},
                          '2s-4d': {},
                          '2s-4f': {},
                          '2p-eloss': {},
                          '2p-3s': {},
                          '2p-3p': {},
                          '2p-3d': {},
                          '2p-4s': {},
                          '2p-4p': {},
                          '2p-4d': {},
                          '2p-4f': {},
                          '3s-eloss': {},
                          '3s-3p': {},
                          '3s-3d': {},
                          '3s-4s': {},
                          '3s-4p': {},
                          '3s-4d': {},
                          '3s-4f': {},
                          '3p-eloss': {},
                          '3p-3d': {},
                          '3p-4s': {},
                          '3p-4p': {},
                          '3p-4d': {},
                          '3p-4f': {},
                          '3d-eloss': {},
                          '3d-4s': {},
                          '3d-4p': {},
                          '3d-4d': {},
                          '3d-4f': {},
                          '4s-eloss': {},
                          '4s-4p': {},
                          '4s-4d': {},
                          '4s-4f': {},
                          '4p-eloss': {},
                          '4p-4d': {},
                          '4p-4f': {},
                          '4d-eloss': {},
                          '4d-4f': {},
                          '4f-eloss': {}}}

Li_Wutte = {'e': {'2s-2p': {'param': [1.847, 4.1818, -27.335, 89.34, 0, 52.788, 1.1971], 'eq': '2'},
                  '2s-3s': {'param': [3.372, 4.7015, -4.6972, -3.0504, 11.332, 0, 0.51449], 'eq': '2'},
                  '2s-3p': {'param': [3.833, 1.4705, 1.5511, 0.94339, 0, 0.12721, 0.53112], 'eq': '2'},
                  '2s-3d': {'param': [3.877, 8.2190, 18.868, -51.214, 55.412, 0, 1.4119], 'eq': '2'},
                  '2p-eloss': {'param': [3.541], 'eq': '3'},
                  '2p-3s': {'param': [1.525, 1.0693e-2, -2.4093, 5.7497, 0, 8.0951, 0.55336], 'eq': '2'},
                  '2p-3p': {'param': [1.986, 10.799, -7.9161, 5.6946, 0, 0, 0.60959], 'eq': '2'},
                  '2p-3d': {'param': [2.030, 3.8735e-3, 37.646, -5.2622, 0, 36.321, 1.0935], 'eq': '2'},
                  '3s-eloss': {'param': [2.018], 'eq': '3'},
                  '3s-3p': {'param': [0.461, 8.4471e-2, 225.8, 0, 0, 322.51, 1.0632], 'eq': '2'},
                  '3s-3d': {'param': [0.505, 91.515, -48.338, 127.34, 0, 0, 1.0738], 'eq': '2'},
                  '3p-eloss': {'param': [1.557], 'eq': '3'},
                  '3p-3d': {'param': [0.044, 3.06044e-2, 25.686, 156.57, 0, 260.22, 1.034], 'eq': '2'},
                  '3d-eloss': {'param': [1.513], 'eq': '3'}},
            '1H1+': {'2s-eloss': {},
                     '2s-2p': {},
                     '2s-3s': {},
                     '2s-3p': {},
                     '2s-3d': {},
                     '2p-eloss': {},
                     '2p-3s': {},
                     '2p-3p': {},
                     '2p-3d': {},
                     '3s-eloss': {},
                     '3s-3p': {},
                     '3s-3d': {},
                     '3p-eloss': {},
                     '3p-3d': {},
                     '3d-eloss': {}}}


def get_Johnson_osc_coeff(n):  # [g0, g1, g2]
    if n == 1:
        return [1.1330, -0.4059, 0.0714]
    if n == 2:
        return [1.0785, -0.2319, 0.02947]
    if n >= 3:
        return [0.9935+0.2328/n-0.1296/n**2, -1/n*(0.6282-0.5598/n+0.5299/n**2), -1/n**2*(0.3887-1.181/n+1.470/n**2)]


def get_Johnson_osc_str(n, m):
    x = 1-(n/m)**2
    g0, g1, g2 = get_Johnson_osc_coeff(n)
    g = g0+g1/x+g2/x**2
    return 32/(3*3**0.5*np.pi)*n/m**3/x**3*g


def get_A(n):
    g0, g1, g2 = get_Johnson_osc_coeff(n)
    g = g0/3+g1/4+g2/5
    return 32*n*g/(3*3**0.5*np.pi)


def get_chi(q):
    return 2**(0.5238*(1-(2/q)**0.5))


def get_Janev_params(cross):
    transition = cross.transition
    e = cross.impact_energy
    if transition.name == 'ex':
        if str(transition.target) == 'e':
            if transition.from_level == '1':  # From RENATE, Johnson version.
                n = int(transition.to_level)
                y = 1-(1/n)**2
                f = get_Johnson_osc_str(1, n)
                b = -0.603
                B = 4/(n**3*y**2)*(1+4/(3*y)+b/y**2)
                return {'param': [1, n, y, 0.45, 2*f/y, B, 13.6*(1-1/n**2)], 'eq': '12'}
            if int(transition.from_level) > 1:
                n = int(transition.from_level)
                m = int(transition.to_level)
                y = 1-(n/m)**2
                f = get_Johnson_osc_str(n, m)
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                return {'param': [n, m, y, 1.94*n**(-1.57), 2*n**2*f/y, 4*n**4/(m**3*y**2)*(1+4/(3*y)+b/y**2),
                                  13.6*(1/n**2-1/m**2)], 'eq': '12'}
        if str(transition.target) == '1H1+':
            if transition.from_level == '1':
                n = int(transition.to_level)
                return {'param': [0.63771, 37.174, 0.39265, 3.2949e-4, 0.25757, -2.2950,
                                  0.050796, -5.5986, n], 'eq': '17'}
            if transition.from_level == '2' and int(transition.to_level) < 11:
                n = transition.to_level
                ratio = {'6': 0.4610, '7': 0.2475,
                         '8': 0.1465, '9': 0.0920, '10': 0.0605}
                return {'param': [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, ratio[n]], 'eq': '19'}
            if transition.from_level == '3' and int(transition.to_level) < 11:
                n = transition.to_level
                ratio = {'7': 0.4670, '8': 0.2545, '9': 0.1540, '10': 0.10}
                return {'param': [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, ratio[n]], 'eq': '19'}
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                m = int(transition.to_level)
                eps = e/25e3
                s = m-n
                D = np.exp(-1/(n*m*eps**2))
                zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
                zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
                y = 1/(1-D*np.log(18*s)/(4*s))
                def C(z, z1): return z**2*np.log(1+2*z/3)/(2*z1+3*z/2)
                return {'param': [n, eps, 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s),
                                  D, np.log(
                                      (1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps)),
                                  (1-0.3*s*D/(n*m))**(1+2*s), 0.5
                                  * (eps*n**2/(m-1/m))**3,
                                  C(zm, y)-C(zp, y)], 'eq': '110'}
        if str(transition.target) == 'He':
            if transition.from_level == '1' and transition.to_level == '2':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [177.69, 64.506, 0.10807, 2.1398e-4, 0.73358, -2.9773, 7.5603e-2,
                                  18.997, 2.4352e-3, 3.4085, e_red], 'eq': '123'}
            if transition.from_level == '1' and transition.to_level == '3':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [18.775, 73.938, 3.2231, 1.2879e-4, 0.75301, -4.1638, 2.366e-1,
                                  20.927, 1.6636e-3, 3.6319, e_red], 'eq': '123'}
            if transition.from_level == '1' and transition.to_level == '4':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [5.5094, 68.504, 12.621, 7.7669e-5, 0.53813, -4.1788, 4.0349e-2,
                                  16.213, 5.4493e-9, 9.5011, e_red], 'eq': '123'}
            if transition.from_level == '1' and transition.to_level == '5':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [4.9796, 64.582, 0.10588, 2.8878e-5, 0.15531, -2.4161,
                                  1.6389e-3, -6.3726, e_red], 'eq': '124'}
            if transition.from_level == '1' and transition.to_level == '6':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [2.55080, 74.348, 0.19625, 3.357e-5, 0.12878, -2.295,
                                  5.144e-3, -5.5986, e_red], 'eq': '124'}
            if transition.from_level == '1' and int(transition.to_level) > 6:
                e_red = e/1e3/transition.target.mass_number
                n = int(transition.to_level)
                return {'param': [2.55080, 74.348, 0.19625, 3.357e-5, 0.12878, -2.295,
                                  5.144e-3, -5.5986, n, e_red], 'eq': '125'}
            if transition.from_level == '2' and transition.to_level == '3':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [1864.3, 19.395, 0.13899, 2.4502e-3, 0.2966, -1.7558, e_red], 'eq': '126'}
            if transition.from_level == '2' and transition.to_level == '4':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [246.18, 27.764, 0.39876, 1.9381e-3, 0.23304, -1.7165, e_red], 'eq': '126'}
            if transition.from_level == '2' and transition.to_level == '5':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [73.056, 37.946, 1.4528, 2.4601e-3, 0.15855, -1.4775, e_red], 'eq': '126'}
            if transition.from_level == '2' and int(transition.to_level) > 5:
                e_red = e/1e3/transition.target.mass_number
                ratio = {'6': 0.461, '7': 0.2475, '8': 0.1465, '9': 0.0920, '10': 0.0605}[transition.to_level]
                return {'param': [73.056, 37.946, 1.4528, 2.4601e-3, 0.15855, -1.4775, e_red, ratio], 'eq': '127'}
            if transition.from_level == '3' and transition.to_level == '4':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [4990, 22.638, 1.3118, 0.014239, 0.260596, -1.2722, e_red], 'eq': '126'}
            if transition.from_level == '3' and transition.to_level == '5':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [762.36, 22.192, 1.4549, 0.014996, 0.27088, -1.2894, e_red], 'eq': '126'}
            if transition.from_level == '3' and transition.to_level == '6':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [253.89, 23.014, 2.1708, 0.01596, 0.2673, -1.2881, e_red], 'eq': '126'}
            if transition.from_level == '3' and int(transition.to_level) > 6:
                e_red = e/1e3/transition.target.mass_number
                ratio = {'7': 0.467, '8': 0.2545, '9': 0.154, '10': 0.1}[transition.to_level]
                return {'param': [253.89, 23.014, 2.1708, 0.01596, 0.2673, -1.2881, e_red, ratio], 'eq': '127'}
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                m = int(transition.to_level)
                eps = e/1e3/transition.target.mass_number/50
                s = m-n
                D = np.exp(-1/(n*m*eps**2))
                A = 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s)
                L = np.log((1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps))
                F = (1-0.3*s*D/(n*m))**(1+2*s)
                G = 0.5 * (eps*n**2/(m-1/m))**3
                zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
                zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
                y = 1/(1-D*np.log(18*s)/(4*s))
                def C(z, z1): return z**2*np.log(1+2*z/3)/(2*z1+3*z/2)
                return {'param': [n, eps, A, D, L, F, G, C(zm, y)-C(zp, y)], 'eq': '128'}
        if str(transition.target) == 'Be':
            if transition.from_level == '1' and transition.to_level == '2':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [961.8, 70.386, 1.4e-2, 1.208e-6, 0.31849, -3.3516, 7.209e-3,
                                  30.194, 2.478e-8, 8.4206, e_red], 'eq': '123'}
            if transition.from_level == '1' and transition.to_level == '3':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [95.728, 90.975, 0.2198, 9.6493e-7, 0.36229, -4.1912,
                                  3.587e-2, 28.681, 5.1187e-9, 9.1415, e_red], 'eq': '123'}
            if transition.from_level == '1' and transition.to_level == '4':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [32.190, 86.307, 0.27496, 1.1909e-6, 0.38748, -4.3014,
                                  3.2598e-2, 26.395, 1.5743e-8, 8.6415, e_red], 'eq': '123'}
            if transition.from_level == '1' and transition.to_level == '5':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [19.918, 129.16, 5.2941e-2, 2.7053e-6, 7.7658e-2,
                                  -2.4161, 5.9445, -6.3726, e_red], 'eq': '124'}
            if transition.from_level == '1' and transition.to_level == '6':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [10.203, 148.7, 9.8162e-2, 3.419e-6, 6.4392e-2,
                                  -2.295, 4.9522, -5.5986, e_red], 'eq': '124'}
            if transition.from_level == '1' and int(transition.to_level) > 6:
                e_red = e/1e3/transition.target.mass_number
                n = int(transition.to_level)
                return {'param': [10.203, 148.7, 9.8162e-2, 3.419e-6, 6.4392e-2,
                                  -2.295, 4.9522, -5.5986, n, e_red], 'eq': '125'}
            if int(transition.from_level) > 1:
                Z = tools.Ion(label='Z', mass_number=transition.target.mass_number,
                              atomic_number=transition.target.atomic_number,
                              charge=transition.target.charge)
                scaled_trans = tools.Transition(projectile=transition.projectile, target=Z,
                                                from_level=transition.from_level,
                                                to_level=transition.to_level,
                                                trans='ex')
                scaled_cross = cross_section.CrossSection(transition=scaled_trans,
                                                          impact_energy=e,
                                                          atomic_dict=cross.atomic_dict)
                return get_Janev_params(scaled_cross)
        if str(transition.target) in ['B', 'C', 'O']:
            Z = tools.Ion(label='Z', mass_number=transition.target.mass_number,
                          atomic_number=transition.target.atomic_number,
                          charge=transition.target.charge)
            scaled_trans = tools.Transition(projectile=transition.projectile, target=Z,
                                            from_level=transition.from_level,
                                            to_level=transition.to_level,
                                            trans='ex')
            scaled_cross = cross_section.CrossSection(transition=scaled_trans,
                                                      impact_energy=e,
                                                      atomic_dict=cross.atomic_dict)
            return get_Janev_params(scaled_cross)
        if str(transition.target) == 'Z':
            if transition.from_level == '1' and transition.to_level == '2':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [38.738, 37.033, 0.39862, 7.7582e-5, 0.25402, -2.7418, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '1' and transition.to_level == '3':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [4.3619, 57.451, 21.001, 2.3292e-4, 0.083130, -2.2364, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '1' and transition.to_level == '4':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [1.3730, 60.710, 31.797, 2.0207e-4, 0.082513, -2.3055, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '1' and int(transition.to_level) > 4:
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [0.56565, 67.333, 55.290, 2.1595e-4, 0.081624, -2.1971, chi, e_red, q,
                                  int(transition.to_level)], 'eq': '116'}
            if transition.from_level == '2' and transition.to_level == '3':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [358.03, 25.283, 1.4726, 0.014398, 0.12207, -0.86210, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '2' and transition.to_level == '4':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [50.744, 19.416, 4.0262, 0.014398, 0.31584, -1.4799, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '2' and transition.to_level == '5':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '2' and int(transition.to_level) > 5 and int(transition.to_level) < 11:
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                scaling = {'6': 0.4610, '7': 0.2475, '8': 0.1465, '9': 0.0920, '10': 0.0605}
                return {'param': [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775, chi, e_red, q,
                                  scaling[transition.to_level]], 'eq': '117'}
            if transition.from_level == '3' and transition.to_level == '4':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [1247.5, 11.319, 2.6235, 0.068781, 0.521176, -1.2722, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '3' and transition.to_level == '5':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [190.59, 11.096, 2.9098, 0.073307, 0.54177, -1.2894, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '3' and transition.to_level == '6':
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                return {'param': [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, chi, e_red, q], 'eq': '115'}
            if transition.from_level == '3' and int(transition.to_level) > 6 and int(transition.to_level) < 11:
                mass = transition.target.mass_number
                q = transition.target.charge
                e_red = e/1000/mass/q
                chi = get_chi(q)
                scaling = {'7': 0.4670, '8': 0.2545, '9': 0.1540, '10': 0.1}
                return {'param': [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881, chi, e_red, q,
                                  scaling[transition.to_level]], 'eq': '117'}
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                m = int(transition.to_level)
                mass = transition.target.mass_number
                q = transition.target.charge
                eps = e/25e3/q/mass
                s = m-n
                A = 8/(3*s)*(m/(s*n))**3*(0.184-0.04/s**(2/3))*(1-0.2*s/(n*m))**(1+2*s)
                D = np.exp(-1/(n*m*eps**2))
                L = np.log((1+0.53*eps**2*n*(m-2/m))/(1+0.4*eps))
                F = (1-0.3*s*D/(n*m))**(1+2*s)
                G = 0.5*(eps*n**2/(m-1/m))**3
                zp = 2/(eps*n**2*((2-n**2/m**2)**0.5+1))
                zm = 2/(eps*n**2*((2-n**2/m**2)**0.5-1))
                y = 1/(1-D*np.log(18*s)/(4*s))
                def C(z, z1): return z**2*np.log(1+2*z/3)/(2*z1+3*z/2)
                chi = 2**(0.322*(1-(2/q)**0.5))
                return {'param': [n, eps, A, D, L, F, G, C(zm, y)-C(zp, y), chi, q], 'eq': '118'}
    if transition.name == 'eloss':
        if str(transition.target) == 'e':
            if int(transition.from_level) > 3:  # From RENATE, Johnson version.
                n = int(transition.from_level)
                y = 1-(1/n)**2
                f = get_Johnson_osc_str(1, n)
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                A = get_A(n)
                return {'param': [n, y, 1.94*n**(-1.57), A, 2/3*n**2*(5+b), 13.6/n**2], 'eq': '14'}
        if str(transition.target) in ['1H1+', 'Z', 'He', 'Be', 'C', 'B', 'O']:
            trans_ion = tools.Transition(projectile=transition.projectile,
                                         target=transition.target,
                                         from_level=transition.from_level,
                                         to_level='ion', trans='ion')
            cross_ion = cross_section.CrossSection(transition=trans_ion,
                                                   impact_energy=e, atomic_dict=cross.atomic_dict)
            trans_cx = tools.Transition(projectile=transition.projectile,
                                        target=transition.target,
                                        from_level=transition.from_level,
                                        to_level='cx', trans='cx')
            cross_cx = cross_section.CrossSection(transition=trans_cx,
                                                  impact_energy=e, atomic_dict=cross.atomic_dict)
            return {'param': cross_ion.function+cross_cx.function, 'eq': '114'}
    if transition.name == 'elossRENMAR':  # From RENATE, Marchuk version.
        if str(transition.target) == 'e':
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                de = 13.6/n**2
                x = e/de
                xx = 1-1/x
                summa = 1/3*(0.9935+0.2328/n-0.1296/n**2)
                -1/(n*4)*(0.06282-0.5598/n+0.5299/n**2)
                +1/(n**2*5)*(0.3887-1.1810/n+1.47/n**2)
                ca = 32/(3*3**0.5*np.pi)
                An = ca*n*summa
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                Bn = 2/3*n**2*(5+b)
                r = 1.94*n**(-1.57)
                sigma = (An*np.log(x)+(Bn-An*np.log(2*n**2))*xx**2)*2*n**2/x *\
                        (1-np.exp(-r*x))*8.7973484e-17
                return {'param': sigma, 'eq': '114'}
    if transition.name == 'elossJANEV':  # INCORRECT! The formulas in Janev are wrong, and the labeling of the plots is in reverse order.
        if str(transition.target) == 'e':
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                y = 1-(1/n)**2
                f = get_Johnson_osc_str(1, n)
                b = 1/n*(4.0-18.63/n+36.24/n**2-28.09/n**3)
                A = 2*n**2*f/y
                return {'param': [n, y, 1.94*n**(-1.57), A, 2/3*n**2*(5+b), 13.6/n**2], 'eq': '14J'}
    if transition.name == 'ion':
        if str(transition.target) == '1H1+':
            if int(transition.from_level) == 2:
                n = int(transition.from_level)
                par = [3.933e-3, 1.8188, 1.887e-2, 6.7489e-3, 1.3768,
                       6.8852e2, 9.6435e1, 5.6515e23]
                e_red = e*n**2/1e3
                sigma = n**4*1e-16*par[0]*(e_red**par[1]*np.exp(-par[2]*e_red)/(1+par[3]*e_red**par[4])
                                           + par[5]*np.exp(-par[6]/e_red)*np.log(1+par[7]*e_red)/e_red)
                return {'param': sigma, 'eq': '114'}
            if int(transition.from_level) == 3:
                n = int(transition.from_level)
                par = [1.1076e-2, 1.6197, 6.7154e-3, 5.1188e-3, 1.8549, 2.3696e+2,
                       7.8286e1, 1.0926e23]
                e_red = e*n**2/1e3
                sigma = n**4*1e-16*par[0]*(e_red**par[1]*np.exp(-par[2]*e_red)/(1+par[3]*e_red**par[4])
                                           + par[5]*np.exp(-par[6]/e_red)*np.log(1+par[7]*e_red)/e_red)
                return {'param': sigma, 'eq': '114'}
            if int(transition.from_level) == 4:
                n = int(transition.from_level)
                par = [1.1033e-2, 1.6281, 5.5955e-3, 7.2023e-3, 1.7358, 2.2755e2,
                       8.6339e1, 3.9151e29]
                e_red = e*n**2/1e3
                sigma = n**4*1e-16*par[0]*(e_red**par[1]*np.exp(-par[2]*e_red)/(1+par[3]*e_red**par[4])
                                           + par[5]*np.exp(-par[6]/e_red)*np.log(1+par[7]*e_red)/e_red)
                return {'param': sigma, 'eq': '114'}
            if int(transition.from_level) >= 5:  # From ADAS
                n = int(transition.from_level)
                e_red = e*n**2/1e3
                par = [1.1297e-2, 1.8685, 1.5038e-2, 1.1195e-1, 1.0538, 8.6096e2,
                       8.9939e1, 1.9249e4]
                sigma = n**4*1e-16*par[0]*(e_red**par[1]*np.exp(-par[2]*e_red)/(1+par[3]*e_red**par[4])
                                           + par[5]*np.exp(-par[6]/e_red)*np.log(1+par[7]*e_red)/e_red)
                return {'param': sigma, 'eq': '114'}

                # n = int(transition.from_level) Janev printed
                # e_red = (3/n)**2*e/1e3
                # return {'param': [336.26, 13.608, 4.9910e+3, 3.0560e-1, 6.4364e-2, -0.14924,
                #                   3.1525, -1.6314, n, e_red], 'eq': '111'}
        if str(transition.target) == 'He':
            if transition.from_level == '1':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [40.498, 112.61, 1.5496e6, 1.4285e-5, 4.1163e-2, -2.6347,
                                  4.0589, -5.9204, e_red], 'eq': '124'}
            if transition.from_level == '2':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [109.01, 26.473, 1.0224e+6, 5.7286e-3, 0.040151, -2.4092,
                                  0.014897, -0.23786, e_red], 'eq': '124'}
            if transition.from_level == '3':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [250.1, 7.9018, 2.1448e6, 0.33041, 0.093012, -0.49446,
                                  0.63357, -2.7261, e_red], 'eq': '124'}
            if int(transition.from_level) > 3:
                n = int(transition.from_level)
                e_red = (3/n)**2*e/1e3/transition.target.mass_number
                return {'param': [250.1, 7.9018, 2.1448e+6, 0.33041, 0.093012,
                                  -0.49446, 0.63357, -2.7621, n, e_red], 'eq': '111'}
        if str(transition.target) == 'Be' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [306.63, 178.22, 62.033, 3.1376e-5, 1.3455e-2, -1.6452, 57.117,
                              -3.53383, e_red], 'eq': '124'}
        if str(transition.target) == 'B' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [351.52, 233.63, 3.2952e3, 5.3787e-6, 1.8834e-2,
                              -2.2064, 7.2074, -3.78664, e_red], 'eq': '124'}
        if str(transition.target) == 'C' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [438.36, 327.1, 1.4444e5, 3.5212e-3, 8.3031e-3, -0.63731,
                              1.9116e4, -3.1003, e_red], 'eq': '124'}
        if str(transition.target) == 'O' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [1244.44, 249.36, 30.892, 9.0159e-4, 7.7885e-3, -0.71309,
                              3.2918e3, -2.7541, e_red], 'eq': '124'}
        if str(transition.target) in ['Be', 'B', 'C', 'O'] and transition.from_level != '1':
            Z = tools.Ion(label='Z', mass_number=transition.target.mass_number,
                          atomic_number=transition.target.atomic_number,
                          charge=transition.target.charge)
            scaled_trans = tools.Transition(projectile=transition.projectile, target=Z,
                                            from_level=transition.from_level,
                                            to_level=transition.to_level,
                                            trans='ion')
            scaled_cross = cross_section.CrossSection(transition=scaled_trans,
                                                      impact_energy=e,
                                                      atomic_dict=cross.atomic_dict)
            return get_Janev_params(scaled_cross)
        if str(transition.target) == 'Z':
            if transition.from_level == '1':
                mass = transition.target.mass_number
                q = transition.target.charge
                return {'param': [1, 0.76, 0.283, 4.04, 0.662, 137, e/1000/mass/25, q], 'eq': '119'}
            if int(transition.from_level) > 1:
                mass = transition.target.mass_number
                q = transition.target.charge
                n = int(transition.from_level)
                return {'param': [n, 0.76, 0.283, 4.04, 0.662, 137, n**2*e/1000/mass/25, q], 'eq': '119'}
    if transition.name == 'cx':
        if str(transition.target) == '1H1+':
            n = int(transition.from_level)
            if n == 2:
                e_red = e*n**2/1e3
                return {'param': [0.92750, 6.5040e+3, 1.3405e-2, 20.699, n, e_red], 'eq': '113'}
            if n == 3:
                e_red = e*n**2/1e3
                return {'param': [0.37271, 2.7645e+6, 1.5720e-3, 1.4857e+3, n, e_red], 'eq': '113'}
            if n >= 4:
                e_red = e*n**2/1e3
                return {'param': [0.21336, 1.0e+10, 1.8184e-3, 1.3426e+6, n, e_red], 'eq': '113'}
        if str(transition.target) == 'He':
            if transition.from_level == '1':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [17.438, 2.1263, 2.1401e-3, 1.6498, 2.6259e-6,
                                  2.4226e-11, 15.665, 7.9193, -4.4053, e_red], 'eq': '129'}
            if transition.from_level == '2':
                e_red = e/1e3/transition.target.mass_number
                return {'param': [88.508, 0.78429, 3.2903e-2, 1.7635, 7.3265e-5,
                                  1.4418e-8, 0.80478, 0.22349, -0.68604, e_red], 'eq': '129'}
            if int(transition.from_level) > 2:
                n = int(transition.from_level)
                e_red = n**2*e/1e3/transition.target.mass_number
                return {'param': [2.0032e2, 1.4591, 2.0384e-4, 2e-9, n, e_red], 'eq': '130'}
        if str(transition.target) == 'Be' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [19.952, 0.20036, 1.7295e-4, 3.6844e-11, 5.0411, 2.4689e-8,
                              4.0761, 0.88093, 0.94361, 0.14205, -0.42973, e_red], 'eq': '131'}
        if str(transition.target) == 'B' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [31.226, 1.1442, 4.8372e-8, 3.0961e-10, 4.7205, 6.2844e-7,
                              3.1297, 0.12556, 0.30098, 5.9607e-2, -0.57923, e_red], 'eq': '131'}
        if str(transition.target) == 'C' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [418.18, 2.1585, 3.4808e-4, 5.3333e-9, 4.6556, 0.33755,
                              0.81736, 0.27874, 1.8003e-6, 7.1033e-2, 0.53261, e_red], 'eq': '131'}
        if str(transition.target) == 'O' and transition.from_level == '1':
            e_red = e/1e3/transition.target.mass_number
            return {'param': [54.535, 0.27486, 1.0104e-7, 2.0745e-9, 4.4416, 7.6555e-3,
                              1.1134, 1.1621, 0.15826, 3.6613e-2, 3.9741e-2, e_red], 'eq': '131'}
        if str(transition.target) in ['Be', 'B', 'C', 'O'] and transition.from_level != '1':
            Z = tools.Ion(label='Z', mass_number=transition.target.mass_number,
                          atomic_number=transition.target.atomic_number,
                          charge=transition.target.charge)
            scaled_trans = tools.Transition(projectile=transition.projectile, target=Z,
                                            from_level=transition.from_level,
                                            to_level=transition.to_level,
                                            trans='cx')
            scaled_cross = cross_section.CrossSection(transition=scaled_trans,
                                                      impact_energy=e,
                                                      atomic_dict=cross.atomic_dict)
            return get_Janev_params(scaled_cross)
        if str(transition.target) == 'Z':
            if transition.from_level == '1':
                mass = transition.target.mass_number
                q = transition.target.charge
                # e_red = e/1000/mass*(1/q**(3/7))# Janev printed
                # return {'param': [0.73362, 2.9391e4, 41.8648, 7.1023e-3, 3.4749e-6, 1.1832e-10, e_red, q], 'eq': '120'}
                e_red = e/1000/mass*(1/q**(0.5))  # RENATE Marschuk
                return {'param': [3.2345, 2.3588e+2, 2.3713, 3.8371e-2, 3.8068e-6, 1.1832e-10, e_red, q, 1], 'eq': '122'}
            if transition.from_level == '2':  # RENATE Marschuk
                mass = transition.target.mass_number
                q = transition.target.charge
                n = int(transition.from_level)
                e_red = e/1000/mass*(1/q**(0.5))*n**2
                return {'param': [9.2750e-1, 6.5040e3, 2.0699e1, 1.3405e-2, 3.0842e-6, 1.1832e-10, e_red, q, n], 'eq': '122'}
            if transition.from_level == '3':  # RENATE Marschuk
                mass = transition.target.mass_number
                q = transition.target.charge
                n = int(transition.from_level)
                e_red = e/1000/mass*(1/q**(0.5))*n**2
                return {'param': [3.7271e-1, 2.7645e6, 1.4857e3, 1.5720e-3, 3.0842e-6, 1.1832e-10, e_red, q, n], 'eq': '122'}
            if int(transition.from_level) > 3:  # RENATE Marschuk
                mass = transition.target.mass_number
                q = transition.target.charge
                n = int(transition.from_level)
                e_red = e/1000/mass*(1/q**(0.5))*n**2
                return {'param': [2.1336e-1, 1e10, 1.3426e6, 1.8184e-3, 3.0842e-6, 1.1832e-10, e_red, q, n], 'eq': '122'}
            # if int(transition.from_level) > 1: Janev printed
            #     mass = transition.target.mass_number
            #     q = transition.target.charge
            #     n = int(transition.from_level)
            #     e_red = e/1000/mass*n**2/q**0.5
            #     return {'param': [1.507e5, 1.974e-5, n, q, e_red], 'eq': '121'}


def H_deex_modifier(rate):
    g1 = int(rate.transition.to_level)**2
    g2 = int(rate.transition.from_level)**2
    if str(rate.transition.target) == 'e':
        deltaE = 13.605693122994*(1/g1-1/g2)
        return rate.rate*g1/g2*np.exp(deltaE/rate.temperature)
    if str(rate.transition.target) in ['1H1+', 'Z', 'He', 'Be', 'C', 'B', 'O']:
        return rate.rate*g1/g2


H_ALADDIN = {'e': {'1-2': {'param': [1.4182, -20.877, 49.735, -46.249, 17.442, 4.4979], 'eq': '10'},
                   '1-3': {'param': [0.42956, -0.58288, 1.0693, 0.0, 0.75448, 0.38277, 12.09], 'eq': '11'},
                   '1-4': {'param': [0.24846, 0.19701, 0.0, 0.0, 0.243, 0.41844, 12.75], 'eq': '11'},
                   '1-5': {'param': [0.13092, 0.23581, 0.0, 0.0, 0.11508, 0.45929, 13.06], 'eq': '11'},
                   '2-3': {'param': [5.2373, 119.25, -595.39, 816.71, 38.906, 1.3196, 1.889], 'eq': '11'},
                   '1-eloss': {'param': [0.18450, -0.032226, -0.034539, 1.4003, -2.8115, 2.2986, 13.6], 'eq': '13'},
                   '2-eloss': {'param': [0.14784, 0.0080871, -0.062270, 1.9414, -2.1980, 0.95894, 3.4], 'eq': '13'},
                   '3-eloss': {'param': [0.058463, -0.051272, 0.85310, -0.57014, 0.76684, 0.0, 1.511], 'eq': '13'}
                   },
             '1H1+': {'1-2': {'param': [34.433, 44.507, 0.56870, 8.5476, 7.8501, -9.2217,
                                        1.8020e-2, 1.6931, 1.9422e-3, 2.9068], 'eq': '15'},
                      '1-3': {'param': [6.1950, 35.773, 0.54818, 5.5162e-3, 0.291114,
                                        -4.5264, 6.0311, -2.0679], 'eq': '16'},
                      '1-4': {'param': [2.0661, 34.975, 0.91213, 5.133e-4, 0.28953,
                                        -2.2849, 0.11528, -4.8970], 'eq': '16'},
                      '1-5': {'param': [1.2449, 32.291, 0.21176, 3.0826e-4, 0.31063,
                                        -2.4161, 0.024664, -6.3726], 'eq': '16'},
                      '1-6': {'param': [0.63771, 37.174, 0.39265, 3.2949e-4, 0.25757,
                                        -2.2950, 0.050796, -5.5986], 'eq': '16'},
                      '2-3': {'param': [394.51, 21.606, 0.62426, 0.013597, 0.16565, -0.8949], 'eq': '18'},
                      '2-4': {'param': [50.744, 19.416, 4.0262, 0.014398, 0.31584, -1.4799], 'eq': '18'},
                      '2-5': {'param': [18.264, 18.973, 2.9056, 0.013701, 0.31711, -1.4775], 'eq': '18'},
                      '3-4': {'param': [1247.5, 11.319, 2.6235, 0.068781, 0.521176, -1.2722], 'eq': '18'},
                      '3-5': {'param': [190.59, 11.096, 2.9098, 0.073307, 0.54177, -1.2894], 'eq': '18'},
                      '3-6': {'param': [63.494, 11.507, 4.3417, 0.077953, 0.53461, -1.2881], 'eq': '18'},
                      '1-ion': {'param': [12.899, 61.897, 9.2731e+3, 4.9749e-4, 3.9890e-2,
                                          -1.590, 3.1834, -3.7154], 'eq': '16'},
                      # '2-ion': {'param': [107.63, 29.860, 1.0176e+6, 6.9713e-3, 2.8448e-2, Janev printed
                      #                     -1.80, 4.7852e-2, -0.20923], 'eq': '16'},
                      # '3-ion': {'param': [336.26, 13.608, 4.9910e+3, 3.0560e-1, 6.4364e-2,
                      #                     -0.14924, 3.1525, -1.6314], 'eq': '16'},
                      '1-cx': {'param': [3.2345, 235.88, 0.038371, 3.8068e-6, 1.1832e-10, 2.3713], 'eq': '112'},
                      },
             'Z': {},
             'He': {},
             'Be': {},
             'B': {},
             'C': {},
             'O': {},
             'generalized': get_Janev_params,
             'de-ex': H_deex_modifier}

ATOMIC_SOURCES = {'Li_Schweinzer': Li_Schweinzer,
                  'Li_Wutte': Li_Wutte,
                  'H_ALADDIN': H_ALADDIN}
