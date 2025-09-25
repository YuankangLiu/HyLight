import numpy as np
import pandas
import string
from copy import deepcopy
import time
import pickle
import scipy.interpolate as interpolate
import unyt
import os
import importlib
import scipy.special as sc

class HIAtom:
    '''
    Compute level population for HI using the cascade matrix formalism.
    See Osterbrock & Ferland 2006, section 4.2
    '''
    def __init__(self, nmax=60, recom=True, coll=True,
                 cache_path = './cache/',
                 caseB = True, verbose=False):
        """Initialize the hydrogen  model. 
        
        :param nmax: Number of levels included in the modeln. Default is ``60``.
        :param recom: Whether to include radiative recombination. Default is ``True``.
        :param coll: Whether to include collisional excitation from the ground state. Default is ``True``.
        :param cache_path: Path to store the cache files. Default is the current working directory ``'./cache/'``.
        :param caseB: Whether to execute in Case B. Default is ``True``.
        :param verbose: Whether to include diagnostic information. Default is ``False``.
        """
        print(" ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░ ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓████████▓▒░░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒▒▓███▓▒░▒▓████████▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        
        # set maximum number of principle quantum number to be used - max allowed is 150
        self.nmax = nmax
        assert self.nmax >= 5, "At least five levels are required. "
        self.nmaxcoll = nmax
        if self.nmax > 150:
            self.nmax = 150
        self.verbose = verbose
        self.caseB  = caseB
        self.recom = recom
        self.coll  = coll
        assert self.recom == True or self.coll == True, "Recombination or collisional processes has to be turned on."

        R_H = 1 / (1 + unyt.electron_mass / unyt.proton_mass) * unyt.R_inf
        self.Eion = (unyt.planck_constant * unyt.c * R_H).in_units('eV')
        self.min_val = 1e-64

        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
            print(f"Folder '{self.cache_path}' created.")
        else:
            print(f"Folder '{self.cache_path}' already exists.")
        
        # Read Einstein A coefficients
        self.TabulatedEinsteinAs = importlib.resources.files('hylightpy.data').joinpath('Einstein_As_150.txt')  # name of the file
        self.A                   = self.read_tabulated_Einstein_coefficients(self.TabulatedEinsteinAs)

    
        # Read level-resolved recombination rates
        self.TabulatedRecombinationRates = importlib.resources.files('hylightpy.data').joinpath('h_iso_recomb_HI_150.dat')   # name of the file
        self.Recom_table = self.read_recombination_rates(self.TabulatedRecombinationRates) # tabulated rates
        self.Alpha_nl = self.fit_recombination_rates()                                     # fitting function to tabulated rates
        if verbose:
            print("Recombination rates read and fitted")
            
        # Read level-resolved collsional exicitation rates
        #self.TabulatedCollisonalExRates = TabulatedCollisionalExRates
        #self.CollEx_table = self.ReadCollisionalExRates(self.TabulatedCollisonalExRates)
        self.q_nl = self.fit_collisional_ex_rates()
        if verbose:
            print("Collisional Excitaion Rates read and fitted")

        # Compute cascade matrix
        self.C    = self.compute_cascade_matrix()
        if verbose:
            print("Cascade matrix class initialized ")

        self.allconfigs = self.get_all_configs()
 

    def get_all_configs(self):
        """Get all the configurations of all the atomic states (n, l).
        
        :return: All the configurations of the atomic states. 
        """
        configs = []
        for nu in np.arange(1, self.nmax+1): # list of indices, 
            for lu in np.arange(nu):
                conf_i = self.config(n=nu, l=lu)
                configs.append(conf_i)
        return configs
        
    ##################################################################    
    #                Recombination rate method                       #
    ##################################################################  
    def alpha_A(self, LogT=4.0):
        """Fit to Case A recombination coefficient at log temperature.
        
        :param LogT: Log10 of temperature. Default is ``4.0``.
        :type LogT: float
        :return: Fitted recombination rate coefficient. 
        :rtype: float
        """
        T      = 10.**LogT
        lamb   = 315614 / T
        alphaA = 1.269e-13 * lamb**(1.503)*(1.0+(lamb/0.522)**(0.470))**(-1.923)
        return alphaA

    def alpha_B(self, LogT=4.0):
        """
        Fit to Case B recombination coefficient at log temperature.
 
        :param LogT: Log10 of temperature. Default is ``4.0``.
        :type LogT: float
        :return: Fitted recombination rate coefficient.
        :rtype: float
        """
        T      = 10.**LogT
        lamb   = 315614 / T
        alphaB = 2.753e-14 * lamb**(1.5)*(1.0+(lamb/2.740)**(0.470))**(-2.2324)
        return alphaB
        
    def read_recombination_rates(self, fname):
        """
        Read level-resolved recombination rates from ascii file in the data folder (h_iso_recomb_HI_150.dat).
        
        :param fname: Name of the tabulated recombination coefficient file. 
        :type fname: str
        :returns: Dictionary of the recombination coefficients.
        :rtype: dict
        """
        
        # contents of the cloudy data file containing l-resolved recombination rates
        # the first line is a comment
        # the second line is the total recombination rate (case-A value)
        # the next lines give the l-resolved recombination rates
        # line = 3: n=1, l=0
        # line = 4: n=2, l=0
        # line = 5: n=2, l=1
        # etc
        # the pandas dataframe recom_data ignores the first line, so that line 1 is case A, line 2 is nl=(1,0), etc
        verbose = False   # set to true to get more information
        

        temp_index = np.arange(41)
        temp_index = [str(x) for x in temp_index]
        # level n has n-1 l values, so number total nyumber of resolved levels is nmax*(nmax+1)/2
        # first row is a magic number, and we start from 0 - hence an offset of 2
        nrows      = int(self.nmax * (self.nmax+1) / 2) + 2
        rows       = np.arange(1, nrows)
        colnames   = ['Z', 'levels'] + temp_index
        try:
            recom_data = pandas.read_csv(fname, delimiter='\t', names=colnames, skiprows=lambda x: x not in rows)
            if verbose:
                print("Successfully read {} l-resolved levels ".format(self.nmax))
        except:
            print("Error reading recombination rates ")
            return -1
        LogTs      = np.linspace(0, 10, 41, endpoint=True)
        return {'LogTs':LogTs, 'recom_data':recom_data}
    
    def read_effective_collisional_strength(self, verbose=False):
        """
        Read level-resolved collisional strength up to n = 5 level from ascii file in the data folder (h_coll_str.dat). 
        
        :param verbose: Output diagnostic information.
        :type verbose: boolean
        :returns: Collisional stength up to n = 5 tabulated by Anderson et al. 2000. 
        :rtype: dict
        """
        fpath = importlib.resources.files('hylightpy.data').joinpath('h_coll_str.dat')

        # columns in Anderson et al. 2000, 2002
        column_type = {'nu': int, 'lu': int, 'nl': int, 'll': int, 
                       '0.5eV': float, '1.0eV': float, '3.0eV': float, '5.0eV': float, 
                       '10.0eV': float, '15.0eV': float, '20.0eV': float, '25.0eV': float}
        self.energy_Anderson = np.array([0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0]) * unyt.eV
        self.temps_Anderson = self.energy_Anderson / unyt.boltzmann_constant_cgs
        self.temps_Anderson = self.temps_Anderson.in_units('K')

        # Cloudy temperature array
        self.log_temps_Cloudy = np.linspace(0, 10, 41, endpoint=True)
        self.temps_Cloudy = 10**(self.log_temps_Cloudy) * unyt.K

        # for levels <= 5, read the tabulated values in Anderson et al. 2000
        try:
            coll_csv = pandas.read_csv(fpath, skiprows=13, delimiter='\t',
                                       names=['nu', 'lu', 'nl', 'll', '0.5eV', '1.0eV', '3.0eV', '5.0eV', '10.0eV', '15.0eV', '20.0eV', '25.0eV'],
                                       converters = column_type)
            
            # if verbose:
            #     print("Successfully read collsional data from levels ".format(self.nmax))
        except:
            print("Error reading collsional excitation rates ")
            return -1
        
        return {'Ts':self.temps_Anderson, 'ups_data':coll_csv}
        

    def fit_recombination_rates(self):
        """
        Provide fitting function for recombination rate as a function of log10 T.
        The fitting funcition is of the form:
        Recombination_rate(n, l, T) = 10**Alpha_nk(10**LogT)
        
        :returns: Fitting function. 
        """
        
        def get_l_level_index(n=1, l=0):
            assert type(n) == np.int64 and type(l) == np.int64, 'n and l must be intergers.'
            assert n >= 1, 'Principle quantum number can not be smaller than 1.'
            assert l < n and l >= 0, 'Angular momentum must be positive and smalled than principle quantum number.'

            # index is numbered from 1 to number of levels up to (nl)
            # offset by 1, since first line is a comment line
            return int((n-1)*n/2) + l # + 1
            
        def fit_rate(recom_data, LogTs, n=1, l=0):
            index   = get_l_level_index(n=n, l=l)
            rate    = interpolate.interp1d(LogTs, recom_data.iloc[index, 2:43].values,fill_value="extrapolate", bounds_error=False)   
            return rate

        rates      = self.read_recombination_rates(self.TabulatedRecombinationRates)
        LogTs      = rates['LogTs']
        recom_data = rates['recom_data']
        
        nmax     = self.nmax
        Alpha_nl = {}
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                conf_i             = self.config(n=n, l=l)
                Alpha_nl[conf_i]   = fit_rate(recom_data, LogTs, n=n, l=l) # try 10**()
        return Alpha_nl

    def collisional_excitation_rate_Lebedev_Beigman(self, nu=6, nl=1, Te=1e4):
        """
        Collisional de-excitation rate calculated based on Lebedev & Beigman 1998 for Rydberg atoms (n >= 5). 
        
        :param nu: Upper level. Default is ``6``.
        :type nu: int
        :param nl: Lower level. Default is ``1``.
        :type nl: int
        :param Te: Temperature. Default is ``10000.0``.
        :type Te: float
        :returns: Collisional de-excitation rate cefficient q_ul.
        :rtype: float
        """
        Te = Te * unyt.K
        gnu = 2 * nu**2
        gnl = 2 * nl**2
        deltaE = self.Eion * (1 / nl**2 - 1 / nu**2)
        qul = gnu / gnl * np.exp(- deltaE / (unyt.boltzmann_constant_cgs * Te)) * self.collisional_deexcitation_rate_Lebedev_Beigman(nu=nu, nl=nl, Te=Te.value)
        return qul

    def collisional_deexcitation_rate_Lebedev_Beigman(self, nu=6, nl=1, Te=1e4):
        """
        Collisional excitation rate calculated based on Lebedev & Beigman 1998 for Rydberg atoms (n >= 5).

        :param nu: Upper level. Default is ``6``.
        :type nu: int
        :param nl: Lower level. Default is ``1``.
        :type nl: int
        :param Te: Temperature. Default is ``10000.0``.
        :type Te: float
        :returns: Collisional excitation rate cefficient q_lu.
        :rtype: float
        """
        Te = Te * unyt.K
        gnu = 2 * nu**2 # 2n**2
        gnl = 2 * nl**2
        alpha = 1. / 137.
        a0 = unyt.reduced_planck_constant_cgs / (unyt.electron_mass_cgs * unyt.c_cgs * alpha)
        Z = 1
        qlu = gnl / gnu * 2. * np.pi * a0**2 * alpha * unyt.c_cgs * nl * (nu / (Z * (nu - nl)))**3 * self.ftheta(nu=nu, nl=nl, Te=Te, Z=Z) * self.psi(nu=nu, nl=nl, Te=Te) / np.sqrt(self.get_theta(Te, Z))
        return qlu
        
    def get_theta(self, Te, Z):
        theta = unyt.boltzmann_constant_cgs * Te / (Z**2 * self.Eion)
        return theta

    def En_potential(self, n):
        En = self.Eion / n**2
        return En

    def psi(self, nu, nl, Te):
        '''
        E1 is the first order exponential integral. I have used sc.exp1 to calculate that. 
        The energy difference is expressed in absolute values, see Lebedev & Beigman 1998 page 225 eq. 8.30.
        '''
        
        numexp = np.float32(self.En_potential(n=nl) / (unyt.boltzmann_constant_cgs * Te * unyt.K))
        
        # values inside sc.exp1 must be dimensionless
        psi_value = 2 * nu**2 * nl**2 / ((nu + nl)**4 * (nu - nl)**2) * (4 * (nu - nl) - 1) * \
                    np.exp(self.En_potential(n=nl) / (unyt.boltzmann_constant_cgs * Te)) * sc.exp1(numexp.value) + \
                    8 * nl**3 / ((nu + nl)**2 * (nu - nl) * nu**2 * nl**2) * (nu - nl - 0.6) * (4 / 3 + nl**2 * (nu - nl)) * \
                    (1 - self.En_potential(nl) / (unyt.boltzmann_constant_cgs * Te) * np.exp(self.En_potential(n=nl) / (unyt.boltzmann_constant_cgs * Te)) * sc.exp1(numexp.value))
        return psi_value

    def ftheta(self, nu, nl, Te, Z=1):
        fval1 = np.log(1 + (nl * self.get_theta(Te=Te, Z=Z)) / (Z * (nu - nl) * np.sqrt(self.get_theta(Te=Te, Z=Z)) + 2.5) )
        fval2 = np.log(1 + (nl * np.sqrt(self.get_theta(Te=Te, Z=Z))) / (Z * (nu-nl)))
        fval = fval1 / fval2
        return fval
        
    def fit_collisional_ex_rates(self):
        """
        Provide fitting function for collisional excitation rate from the ground state.
        The fitting funcition is of the form:
        Collisional_rate(n, l, T) = 10**q_lu(10**LogT)
        
        :returns: Fitting function of the collisioanl excitation rate coefficient.
        """
        
        def fit_coll_ex_rate(ups_data, Ts, n=1, l=0):

            if n <= 5:
                #index   = get_l_level_index(n=n, l=l)
                # collisional excitation rate from the 1s state
                mask = (ups_data['nu'] == n) & (ups_data['lu'] == l) & (ups_data['nl'] == 1) & (ups_data['ll'] == 0)
                upsilon_Anderson = ups_data[mask].iloc[:, 4:].values[0]
                
                delta_E = - self.Eion * (1 - 1 / n**2)
                g_l = 2 # statistical weight of 1s state
                
                q_lu = 8.629e-6 * upsilon_Anderson * np.exp(delta_E / (unyt.boltzmann_constant_cgs * Ts)) / (g_l * np.sqrt(Ts))
                q_lu = q_lu.value
                # filter out NaN values and small values
                q_lu[np.isnan(q_lu)] = self.min_val
                q_lu[q_lu < self.min_val] = self.min_val
                ##qfit = interpolate.interp1d(np.log10(Ts), np.log10(q_lu), fill_value='extrapolate') # simply extrapolate here, will improve later
                qfit = np.polyfit(np.log10(Ts), np.log10(q_lu), deg=5) # return the fitting coefficients
                #q3_model_poly = 10**np.polyval(poly1, 4)
                
            else:
                # if n > 5 (Rydberg atom), use Lebdev and Beigman 1998 method
                if l == 1:

                    q_lu = self.collisional_excitation_rate_Lebedev_Beigman(nu=n, nl=l, Te=Ts.value)
                    q_lu = q_lu.value
                    q_lu[np.isnan(q_lu)] = self.min_val
                    q_lu[q_lu < self.min_val] = self.min_val
                    ##qfit = interpolate.interp1d(np.log10(Ts), np.log10(q_lu), fill_value='extrapolate') # simply extrapolate here, will improve later
                    qfit = np.polyfit(np.log10(Ts), np.log10(q_lu), deg=5)
                    
                else:
                    q_lu = self.min_val + np.zeros_like(self.temps_Anderson)
                    ##qfit = interpolate.interp1d(np.log10(Ts), np.log10(q_lu), fill_value='extrapolate') # simply extrapolate here, will improve later
                    qfit = np.polyfit(np.log10(Ts), np.log10(q_lu), deg=5) 
                    
            return qfit

        ups     = self.read_effective_collisional_strength(verbose=True)
        Ts      = ups['Ts']
        upsdata = ups['ups_data']
        nmax     = self.nmaxcoll
        q_nl = {}
        for n in np.arange(2, nmax+1): # mininum level is 2
            for l in np.arange(n):
                conf_i             = self.config(n=n, l=l)
                coeffs             = fit_coll_ex_rate(upsdata, Ts, n=n, l=l)
                q_nl[conf_i]       = np.poly1d(coeffs) # add 10**
                
        return q_nl
    
    
    
    ##################################################################    
    #                Cascade matrix methods                          #
    ##################################################################
    def compute_level_pop(self, nHII = [1e2] * unyt.unyt_array(1, 'cm**(-3)'), 
                                ne = [1e2] * unyt.unyt_array(1, 'cm**(-3)'), 
                                nHI = [1e-5] * unyt.unyt_array(1, 'cm**(-3)'), 
                                temp = [1e4] * unyt.unyt_array(1, 'K'), 
                                n=3, l=0, verbose=False):
        """
        Compute level population for a given level at a given density and temperature.
        
        :param nHII: Proton number density [cm^{-3}]. Default is [1e2] * unyt.unyt_array(1, 'cm**(-3)').
        :type nHII: float
        :param ne: Electron number density [cm^{-3}]. Default is [1e2] * unyt.unyt_array(1, 'cm**(-3)').
        :type ne: float
        :param nHI: Neutral hydrogen density [cm^{03}]. Default is [1e-5] * unyt.unyt_array(1, 'cm**(-3)'). If collsional excitation is not enabled, any number input here will not go into the calculation. 
        :type nHI: float
        :param temp: Temperature. Default is [1e4] * unyt.unyt_array(1, 'K').
        :type temp: float
        :param n: Principle quantum number of the desired level. Default is ``2``.
        :type n: int
        :param l: Angular momentum quantum number of the desired level. Default is ``0``.
        :type l: int
        :param verbose: Output diagnostic information. Default is ``False``.
        :type verbose: boolean
        :returns: Level population density of the desired level in units of cm^{-3}.
        :rtype: float
        """
        assert type(ne) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(nHI) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(nHII) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(temp) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert (type(n) == int) and (n >= 3), "Invalid input for principle quantum number n."
        assert (type(l) == int) and (l < n), "Invalid input for angular momentum quantum number l."
        
        try:
            ne = ne.in_units('cm**(-3)').value
            nHI = nHI.in_units('cm**(-3)').value
            nHII = nHII.in_units('cm**(-3)').value
            temp = temp.in_units('K').value
        except TypeError:
            pass
        
        # check if all the quantities have the same dimension
        # convert LogT to float
        LogT = np.log10(temp)
        
        nmax     = self.nmax
        A        = self.A
        C        = self.C
        Alpha_nl = self.Alpha_nl
        q_nl     = self.q_nl
        Config   = self.config

        # test for consistency
        if (n < 1) or (n > self.nmax):
            print("Error: n needs to be in range 2 - {}".format(self.nmax))
        if (l<0) or (l >= n):
            print("Error: l needs to be in the range 0 -- {}".format(n-1))
        
        lhs    = np.zeros_like(LogT)
        lhs_rr    = np.zeros_like(LogT)
        lhs_ce    = np.zeros_like(LogT)
        
        conf_k = Config(n=n, l=l)
        config_ind = 0
        
        for ind, _ in enumerate(self.allconfigs):
            if self.allconfigs[ind] == (np.int64(n), np.int64(0)):
                config_ind = ind
                break
        
        configs_subset = self.allconfigs[config_ind:]

        for conf_i in configs_subset:
            if self.recom:
                # radiative contribution
                lhs_rr += 10**Alpha_nl[conf_i](LogT) * C[conf_i][conf_k]
            if self.coll:
                # collisional excitation from the ground state
                lhs_ce += 10**q_nl[conf_i](LogT) * C[conf_i][conf_k]
        
        
        lhs_rr *= nHII * ne
        lhs_ce *= nHI * ne
        lhs    = lhs_rr + lhs_ce

        # 
        rhs    = np.zeros_like(LogT)
        conf_i = Config(n=n, l=l)
        for nd in np.arange(1, n):
            for ld in [l-1, l+1]:
                if (ld >=0) & (ld < nd):
                    conf_k = Config(n=nd, l=ld)
                    rhs += A[conf_i][conf_k]
            if (nd == 1) & (n == 2) & (l == 0):
                ld     = 0
                conf_k = Config(n=nd, l=ld)
                rhs    += A[conf_i][conf_k]

        N       = np.zeros_like(LogT) * unyt.unyt_array(1, 'cm**(-3)')
        mask    = rhs > 0
        N[mask] = lhs[mask]/rhs[mask]
        if verbose:
            print("Computed level pop for level = {0:s}, log N = {1:2.4f}".format(conf_i, np.log10(N)))
        return N

    def compute_all_level_pops(self, nHII = [1e2] * unyt.unyt_array(1, 'cm**(-3)'), 
                                     ne = [1e2] * unyt.unyt_array(1, 'cm**(-3)'), 
                                     nHI = [1e-5] * unyt.unyt_array(1, 'cm**(-3)'), 
                                     temp = [1e4] * unyt.unyt_array(1, 'K'), verbose=False):
        """
        Compute level population for all levels.
        
        :param nHII: Proton number density [cm^{-3}]. Default is [1e2] * unyt.unyt_array(1, 'cm**(-3)').
        :type nHII: float
        :param ne: Electron number density [cm^{-3}]. Default is [1e2] * unyt.unyt_array(1, 'cm**(-3)').
        :type ne: float
        :param nHI: Neutral hydrogen number density [cm^{-3}]. Default is [1e-5] * unyt.unyt_array(1, 'cm**(-3)'). If collsional excitation is not enabled, any number input here will not go into the calculation. 
        :type nHI: float
        :param temp: Temperature. Default is [1e4] * unyt.unyt_array(1, 'K').
        :type temp: float
        :returns: Level population for all levels in units of cm^{-3}. 
        :rtype: float
        """
        assert type(ne) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(nHI) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(nHII) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(temp) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        
        
        try:
            ne = ne.in_units('cm**(-3)').value
            nHI = nHI.in_units('cm**(-3)').value
            nHII = nHII.in_units('cm**(-3)').value
            temp = temp.in_units('K').value
        except TypeError:
            pass
        
        # check if all the quantities have the same dimension
        # convert LogT to float
        LogT = np.log10(temp)
        
        nmax     = self.nmax
        A        = self.A
        C        = self.C
        Alpha_nl = self.Alpha_nl
        q_nl     = self.q_nl
        Config   = self.config
        

        #
        N        = {}
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                lhs       = 0.0
                lhs_rr    = 0.0
                lhs_ce    = 0.0
                conf_k = Config(n=n, l=l)
                for nu in np.arange(n, nmax+1):
                    for lu in np.arange(nu):
                        conf_i = Config(n=nu, l=lu)
                        if self.recom:
                            # radiative contribution
                            lhs_rr += 10**Alpha_nl[conf_i](LogT) * C[conf_i][conf_k]
                        if self.coll:
                            # collisional excitation from the gorund state
                            lhs_ce += 10**np.polyval(q_nl[conf_i], LogT) * C[conf_i][conf_k]
                        
                lhs_rr *= nHII * ne
                lhs_ce *= nHI * ne
                lhs = lhs_rr + lhs_ce
                
                # 
                rhs    = 0.0
                conf_i = Config(n=n, l=l)
                for nd in np.arange(1, n):
                    for ld in [l-1, l+1]:
                        if (ld >=0) & (ld < nd):
                            conf_k = Config(n=nd, l=ld)
                            rhs += A[conf_i][conf_k]
                    if (nd == 1) & (n == 2) & (l == 0):
                        ld     = 0
                        conf_k = Config(n=nd, l=ld)
                        rhs    += A[conf_i][conf_k]

                N[conf_i] = 0.0
                if rhs>0:
                    N[conf_i] = lhs/rhs * unyt.unyt_array(1, 'cm**(-3)')
        return N
        
    def compute_cascade_matrix(self):
        """
        Compute cascade matrix from Einstein coefficients
        
        :returns: Cascade matrix.
        :rtype: dict
        """
        
        nmax     = self.nmax          # max upper level
        A        = self.A             # Einstein coefficient
        verbose  = self.verbose
        Config   = self.config
        topickle = True
        
        # if pickle file exists, read it
        if topickle:
        
            if self.caseB:
                pname   = 'CascadeC_' + str(self.nmax) + '_' + 'B.pickle'
            else:
                pname   = 'CascadeC_' + str(self.nmax) + '_' + 'A.pickle'
            try:
                with open(os.path.join(self.cache_path, pname), 'rb') as file:
                    data = pickle.load(file)

                # check if nmax is correct
                success = (self.nmax == data['nmax'])
                if success:
                    C = data['C']
                    P = data['P']
                    self.P = P
                    if self.verbose:
                        print("Cascade matrix coefficients unpickled")
                    return C
                else:
                    if self.verbose:
                        print("Computing cascade matrix coefficients")
            except:
                pass
        else:
            if self.verbose:
                 print("Computing cascade matrix coefficients")
        
        
        # compute probability matrix (eq. 4.8)
        t0 = time.time()
        P  = deepcopy(A)
        if self.verbose:
            print(" ... Cascade matrix: P copied in time {0:1.2f} s".format(time.time()-t0))
        #
        t0 = time.time()
        for nu in np.arange(2, nmax+1):
            for lu in np.arange(nu):
                conf_i    = Config(n=nu, l=lu)
                #
                denom  = 0.0
                if conf_i == (2,0):
                    denom += A[(2,0)][(1,0)]
                for nprime in np.arange(nu):
                    for lprime in [lu-1, lu+1]:
                        if (lprime >= 0) & (lprime < nprime):
                            conf_prime = Config(n=nprime, l=lprime)
                            denom     += A[conf_i][conf_prime]
                for nd in np.arange(1, nu):
                    # add 2s->1s forbidden transition
                    if (nd == 1) & (nu == 2) & (lu == 0):
                        ld     = 0
                        conf_k = Config(n=nd, l=ld)
                        P[conf_i][conf_k] = 1.0
                        
                    # other transitions
                    if denom > 0:
                        for ld in [lu-1, lu+1]:
                            if (ld >= 0) & (ld < nd):
                                conf_k = Config(n=nd, l=ld)
                                P[conf_i][conf_k] = A[conf_i][conf_k] / denom
        if self.verbose:
            print(" ... Cascade matrix: probability matrix computed in time {0:1.2f}".format(time.time()-t0))
        self.P = P
        
        # Compute the transpose of P
        t1 = time.time()
        Pt = {}
        for nd in np.arange(1, nmax+1):
            for ld in np.arange(nd):
                conf_k = Config(n=nd, l=ld)
                Pt[conf_k] = {}
                for nu in np.arange(nd+1, nmax+1):
                    for lu in np.arange(nu):
                        conf_i = Config(n=nu, l=lu)
                        Pt[conf_k][conf_i] = P[conf_i][conf_k]
        if self.verbose:
            print(" ... Cascade matrix: transpose of probability matrix computed in time {0:1.2f}".format(time.time()-t1))
                        
                
        # Compute cascade matrix (eq. 4.10)
        t1 = time.time()
        C  = {}
        for nu in np.arange(1, nmax+1):
            for lu in np.arange(nu):
                conf_i    = Config(n=nu, l=lu)
                C[conf_i] = {}
                for nd in np.arange(1, nu+1):
                    for ld in np.arange(nd):
                        conf_k = Config(n=nd, l=ld)
                        C[conf_i][conf_k] = 0.0
                        if (nd==nu) & (ld==lu):
                            C[conf_i][conf_k] = 1.0

        # Initialize recurrence (below 4.8)
        nu   = nmax
        nd   = nu - 1
        for lu in np.arange(nu):
            conf_i    = Config(n=nu, l=lu)
            for ld in [lu-1, lu+1]:
                if (ld >= 0) & (ld < nd):
                    conf_k = Config(n=nd, l=ld)
                    C[conf_i][conf_k] = P[conf_i][conf_k]
                    
        if verbose:
            print(" ... Cascade matrix: matrix initialized (eq. 4.10) in time {0:1.2f}".format(time.time()-t1))

                    
        # add 2s->1s forbidden transition
        conf_i            = Config(n=2, l=0)
        conf_k            = Config(n=1, l=0)
        C[conf_i][conf_k] = P[conf_i][conf_k]
        
        # Recur (complete Equation 4.10)
        tp2 = time.time()
        
        #
        for nu in np.arange(nmax, 0, -1):
            tsplit = time.time()
            for lu in np.arange(nu):
                conf_i    = Config(n=nu, l=lu)
                #
                for nd in np.arange(nu, 0, -1):
                    for ld in np.arange(nd):
                        conf_k = Config(nd, ld)
                       # create list, conf_prime, of all intermediate levels that contribute
                        conf_prime = []
                        C_prime    = []
                        P_prime    = []
                        for lprime in [ld-1, ld+1]:
                            if (lprime >=0) & (lprime < nd+1):
                                for nprime in range(nd+1, nu+1):
                                    conf = Config(nprime, lprime)
                                    C_prime.append(C[conf_i][conf])
                                    P_prime.append(Pt[conf_k][conf])
                        res = np.sum(np.array(C_prime) * np.array(P_prime))

                        # update cascade matrix
                        C[conf_i][conf_k] += res
            tsplit = time.time() - tsplit
            print(" ...    Computed level = {0:d} in time {1:1.2f}, len = {2:d}".format(nu, tsplit, len(C_prime)))
        tp2  = time.time() - tp2

        if verbose:
            print(" ... Cascade matrix: calculation finished in time {0:1.2f}s".format(tp2))
            
        # save as a pickle file
        if topickle:
            data = {'nmax':self.nmax, 'C':C, 'P':P}
            with open(os.path.join(self.cache_path, pname), 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            if self.verbose:
                print("Cascade matric elements pickled to file ", pname)


        return C

                    
    def read_tabulated_Einstein_coefficients(self, fname):
        """
        Read tabulated Einstein coefficients.
        
        :param fname: Filename.
        :type fname: str
        :returns: Dictionary of Einstein A values. 
        :rtype: dict
        """
        
        verbose = False  # set True to get timing info
        
        # check if Einstein A pickle file exists, read it if it exits
        pname   = 'Einstein_As.pickle'
        try:
            with open(os.path.join(self.cache_path, pname), 'rb') as file:
                data = pickle.load(file)

            # check if nmax is correct
            success = (self.nmax == data['nmax'])
            if success:
                A = data['A']
                if self.verbose:
                    print("Einstein coefficients unpickled from existing file")

                # impose Case B
                if self.caseB:
                    if self.verbose:
                        print(" ... Imposing caseB (no Lyman-transitions) ")
                        
                    conf_k = self.config(n=1, l=0) # ground state
                    for nu in np.arange(2, self.nmax+1):
                        conf_i = self.config(n=nu, l=1) # all p-state
                        A[conf_i][conf_k] = 0.0
                return A
            else:
                if self.verbose:
                    print("Reading Einstein coefficients from file {}".format(fname))
        except:
            pass

        # Nist value of forbidden 2s-1s transition. This value is not in the data file read here
        A_2s_1s = 8.224 #2.496e-06 # 
        
        
        # columns are n_low, l_low, n_up, l_up, A [1/s]
        tinit    = time.time()
        dtype    = {'names': ('nd', 'ld', 'nu', 'lu', 'A'),
                  'formats': (np.int32, np.int32, np.int32, np.int32, np.float64)}
        data    = np.loadtxt(fname, delimiter=",", dtype=dtype, comments='#', ).T
        nmax    = self.nmax
        tinit   = time.time() - tinit
        if self.verbose:
            print(" ... Read numerical data in time {0:1.2f}".format(tinit))

        # create Einstein coefficients dictionary
        t0      = time.time()
        A       = {}
        # loop over upper level
        for nu in np.arange(2, nmax+1):
            for lu in np.arange(nu):
                conf_i = self.config(n=nu, l=lu)
                A[conf_i] = {}
                # loop over lower level
                for nd in np.arange(nu):
                    for ld in np.arange(nd):
                        conf_k = self.config(n=nd, l=ld)
                        A[conf_i][conf_k] = 0
        t0 = time.time() - t0
        if verbose:
            print(" ... Created dictionary of Einstein coefficients in a time {0:1.2f}".format(t0))
                        
        # insert the values from the file
        t1       = time.time()
        nups     = data['nu'][:]
        lups     = data['lu'][:]
        nds      = data['nd'][:]
        lds      = data['ld'][:]
        Avals    = data['A'][:]
        for nup, lup, nd, ld, Aval in zip(nups, lups, nds, lds, Avals):
            conf_i = self.config(n=nup, l=lup)
            conf_k = self.config(n=nd, l=ld)
            if nup <= nmax:
                A[conf_i][conf_k] = Aval
            else:
                continue
        t1 = time.time() - t1
        if verbose:
            print(" ... Inserted numerical values in Einstein dictionary in a time {0:1.2}".format(t1))
        
        # insert A_2s-1s
        nu = 2
        lu = 0
        nd = 1
        ld = 0
        conf_i = self.config(n=nu, l=lu)
        conf_k = self.config(n=nd, l=ld)
        A[conf_i][conf_k] = A_2s_1s

        # Save the file first, before imposing Case B, otherwise the As are saved with wrong case
        # Original A values are saved, then check if in Case B limit.
        data = {'nmax':self.nmax, 'A':A}
        with open(os.path.join(self.cache_path, pname), 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            print(" ... Einstein dictionary pickled to file {}".format(pname))
        
        # imposing Case B limit
        if self.caseB:
            if self.verbose:
                print(" ... Imposing caseB (no Lyman-transitions) ")
            conf_k = self.config(n=1, l=0) # ground state
            for nu in np.arange(2, nmax+1):
                conf_i = self.config(n=nu, l=1) # p-state
                A[conf_i][conf_k] = 0.0
        
        return A

        
    def config(self, n=1, l=1):
        """
        Configuration states are tuples of the form (n,l), where        
        n = principle quantum number
        l = angular momentum quantum number
        
        :param n: Principle quantum number. Default is ``1``.
        :type n: int
        :param l: Angular momentum quantum number. Default is ``1``.
        :type l: int
        :returns: Configuration of a specific state in tuple. 
        :rtype: tuple
        """
        return (n,l)
        
    def de_config(self, config=(1,0)):
        """
        Extract n and l value for a given configuration.
        
        :param config: Configuration of a state.
        :type config: tuple
        :returns: Principle quantum number and angular momentum quantum number.
        :rtype: int, int
        """
        return config[0], config[1]

    def alpha_effective(self, n=2, l=0, LogT=4.0):
        """
        Effective recombination rate coefficient to a speciic level nl. 
        
        :param n: Principle quantum number. Default is ``2``.
        :type n: int
        :param l: Angular momentum quantum number. Default is ``0``.
        :type l: int
        :param LogT: Temperature in log10. Default is ``4.0``.
        :type LogT: float
        :returns: Effective recombination coefficient. 
        :rtype: float
        """
        assert n >= 2, "Onlt n >= 2 makes sense."
        alpha_nl = 0 # based on Eq. 30 in Pengelly 1964 
        conf_k = self.config(n=n, l=l) # loop over all l-states within this upper level
        # alpha * recom coeff
        for n in np.arange(n, self.nmax+1):
            for l in np.arange(n):
                conf_i = self.config(n=n, l=l)
                if self.recom:
                    alpha_nl += 10**self.Alpha_nl[conf_i](LogT) * self.C[conf_i][conf_k]
        return alpha_nl

    def get_A_coeffs(self, nupper=3, nlower=2):
        '''
        Summing Einstein coefficients for all substates in the upper level.

        :param nupper: Principle quantum number for the upper level. Default is ``3``.
        :type nupper: int
        :param nlower: Principle quantum number for the lower level. Default is ``2``.
        :type nlower: int
        
        :return: Summed Einstein coefficient.
        :rtype: dict
        '''
        assert (type(nupper) == int) and (nupper >= 2) and (nupper > nlower), "Invalid input for nup."
        assert (type(nlower) == int) and (nlower >= 1), "Invalid input for ndown."
        
        As = {}
        for lup in np.arange(nupper):
            conf_up  = self.config(n=nupper, l=lup)
            A        = 0.0
            for ldown in np.arange(nlower):
                conf_down = self.config(n=nlower, l=ldown)
                try:
                    A += self.A[conf_up][conf_down]
                except:
                    pass
            As[conf_up] = A
        return As

    def get_emissivity(self, ne=[1e2] * unyt.unyt_array(1, 'cm**(-3)'), 
                             nHI=[1e-5] * unyt.unyt_array(1, 'cm**(-3)'), 
                             nHII=[1e2] * unyt.unyt_array(1, 'cm**(-3)'), 
                             temp=[1e4] * unyt.unyt_array(1, 'K'), 
                             nupper=3, nlower=2):
        '''Compute line emissivity for given density and temperature. 

        :param ne: Electron density. Default is [1e2] * unyt.unyt_array(1, 'cm**(-3)'). 
        :type ne: float
        :param nHI: Neutral hydrogen density. Default is [1e-5] * unyt.unyt_array(1, 'cm**(-3)'). 
        :type nHI: float
        :param nHII: Proton density. Default is [1e2] * unyt.unyt_array(1, 'cm**(-3)'). 
        :type nHII: float
        :param temp: Temperature. Default is [1e4] * unyt.unyt_array(1, 'K'). 
        :type: float
        :param nupper: Principle quantum number for the upper level. Default is ``3``.
        :type nupper: int
        :param nlower: Principle quantum number for the lower level. Default is ``2``.
        :type nlower: int 
        :return: Line emissisity in erg s**(-1) cm**(-3) 
        :rtype: float
        '''
        assert type(ne) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(nHI) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(nHII) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert type(temp) == unyt.array.unyt_array, 'Please input quantities with units using unyt_array.'
        assert (type(nupper) == int) and (nupper >= 2) and (nupper > nlower), "Invalid input for principle quantum number nupper."
        assert (type(nlower) == int) and (nlower >= 1), "Invalid input for principle quantum number nlower."
        
        try:
            ne = ne.in_units('cm**(-3)').value
            nHI = nHI.in_units('cm**(-3)').value
            nHII = nHII.in_units('cm**(-3)').value
            temp = temp.in_units('K').value
        except TypeError:
            pass
        
        population = np.zeros((len(ne), nupper), dtype=float) * unyt.unyt_array(1, 'cm**(-3)')
        for l in np.arange(nupper):
            population[:, l] = self.compute_level_pop(nHII=nHII * unyt.array.unyt_array(1, 'cm**(-3)'), 
                                                   ne=ne * unyt.array.unyt_array(1, 'cm**(-3)'), 
                                                   nHI=nHI * unyt.array.unyt_array(1, 'cm**(-3)'), 
                                                   temp=temp * unyt.array.unyt_array(1, 'K'),  
                                                   n=nupper, l=int(l), verbose=False)
            
        As = self.get_A_coeffs(nupper=nupper, nlower=nlower)
    
        emis = {}
        emis_tot = 0
        for lup in np.arange(nupper):
            # get population level
            conf        = self.config(n=nupper, l=lup)
            emis[conf] = {}
            emis[conf] = As[conf] * unyt.s**(-1) * self.Eion * (1./nlower**2 - 1./nupper**2) * population[:, lup]
            emis_tot += emis[conf]
            
        return emis_tot.in_units('erg * cm**(-3) * s**(-1)')


    def branching_ratio(self, nupper=3, nlower=2, LogT=4.0, caseB=True):
        '''
        Calculate the fraction of recombination that results in the emission of a specific line, taking into account only radiative processes. 
        The definition can be found in Liu et al. 2025. 

        :param nupper: Principle quantum number for the upper level. Default is ``3``.
        :type nupper: int
        :param nlower: Principle quantum number for the lower level. Default is ``2``.
        :type nlower: int
        :return: The fraction of recombination that results in the emission of a specific line. 
        :rtype: float
        '''
        if caseB == True:
            alpha_tot = self.alpha_B(LogT=LogT)
        else:
            alpha_tot = self.alpha_A(LogT=LogT)
        
        lterms = np.zeros(nupper)
        
        As = {}
        for lup in np.arange(nupper):
            conf_up  = self.config(n=nupper, l=lup)
            Atemp    = 0.0
            for ldown in np.arange(nlower):
                conf_down = self.config(n=nlower, l=ldown)
                try:
                    Atemp += self.A[conf_up][conf_down]
                except:
                    pass
            As[conf_up] = Atemp

        for lup in np.arange(nupper):
            lhs_rr = 0
            conf_k = self.config(n=nupper, l=lup) # loop over all l-states within this upper level
            # alpha * recom coeff
            for n in np.arange(nupper, self.nmax+1):
                for l in np.arange(n):
                    conf_i = self.config(n=n, l=l)
                    if self.recom:
                        lhs_rr += 10**self.Alpha_nl[conf_i](LogT) * self.C[conf_i][conf_k]
            
            rhs    = 0
            conf_i = self.config(n=nupper, l=lup)
            
            for nd in np.arange(1, nupper):
                for ld in [lup-1, lup+1]:
                    if (ld >=0) & (ld < nd):
                        conf_k = self.config(n=nd, l=ld)
                        rhs += self.A[conf_i][conf_k]
                if (nd == 1) & (nupper == 2) & (lup == 0):
                    ld     = 0
                    conf_k = self.config(n=nd, l=ld)
                    rhs    += self.A[conf_i][conf_k]
            
            lterms[lup] = lhs_rr / rhs

        alpha_eff = 0
        for i, level_config in enumerate(As.keys()):
            alpha_eff += lterms[i] * As[level_config]
        R = alpha_eff / alpha_tot
        return R
