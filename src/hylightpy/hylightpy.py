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

class HILevelPopulations:
    '''
    Compute level population for HI using the cascade matrix formalism.
    See Osterbrock & Ferland 2006, section 4.2
    '''
    def __init__(self, nmax=60, recom=True, coll=True,
                 cache_path = './cache/',
                 caseB = True, verbose=False):
        print(" ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░ ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓████████▓▒░░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒▒▓███▓▒░▒▓████████▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        print(" ░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░  ░▒▓█▓▒░     ", flush=True)
        
        # set maximum number of principle quantum number to be used - max allowed is 40
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
        
        # Read Einstein coefficients
        self.TabulatedEinsteinAs = importlib.resources.files('hylightpy.data').joinpath('Einstein_As_150.txt')  # name of the file
        self.A                   = self.ReadTabulatedEinsteinCoefficients(self.TabulatedEinsteinAs)

    
        # Read level-resolved recombination rates
        self.TabulatedRecombinationRates = importlib.resources.files('hylightpy.data').joinpath('h_iso_recomb_HI_150.dat')   # name of the file
        self.Recom_table = self.ReadRecombinationRates(self.TabulatedRecombinationRates) # tabulated rates
        self.Alpha_nl = self.FitRecombinationRates()                                     # fitting function to tabulated rates
        if verbose:
            print("Recombination rates read and fitted")
            
        # Read level-resolved collsional exicitation rates
        #self.TabulatedCollisonalExRates = TabulatedCollisionalExRates
        #self.CollEx_table = self.ReadCollisionalExRates(self.TabulatedCollisonalExRates)
        self.q_nl = self.FitCollisionalExRates()
        if verbose:
            print("Collsional Excitaion Rates read and fitted")

        # Compute cascade matrix
        self.C    = self.ComputeCascadeMatrix()
        if verbose:
            print("Cascade matrix class initialized ")

        self.allconfigs = self.GetAllConfigs()
 

    def GetAllConfigs(self):
        configs = []
        for nu in np.arange(1, self.nmax+1): # list of indices, 
            for lu in np.arange(nu):
                conf_i = self.Config(n=nu, l=lu)
                configs.append(conf_i)
        return configs
        
    ##################################################################    
    #                Recombination rate method                       #
    ##################################################################  
    def AlphaA(self, LogT=4.0):
        ''' Fit to case-A recombination coefficient at log temperature LogT'''
        T      = 10.**LogT
        lamb   = 315614 / T
        alphaA = 1.269e-13 * lamb**(1.503)*(1.0+(lamb/0.522)**(0.470))**(-1.923)
        return alphaA

    def AlphaB(self, LogT=4.0):
        ''' Fit to case-B recombination coefficient at log temperature T'''        
        T      = 10.**LogT
        lamb   = 315614 / T
        alphaB = 2.753e-14 * lamb**(1.5)*(1.0+(lamb/2.740)**(0.470))**(-2.2324)
        return alphaB
        
    def ReadRecombinationRates(self, fname):
        '''
        Read level-resolved recombination rates from ascii file fname, and return them
        '''
        
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
    
    def ReadEffectiveCollisonalStrength(self, verbose=False):
        '''
        Read level-resolved collisional excitation rates from ascii file fname, and return them
        '''
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
        

    def FitRecombinationRates(self):
        ''' 
        Provide fitting function for recombination rate as a function of Log T
        
        The fitting funcition is of the form 
        
        Recombination_rate(n, l, T) = 10**Alpha_nk(10**LogT)
        '''
        def get_l_level_index(n=1, l=0):
            assert type(n) == np.int64 and type(l) == np.int64, 'n and l must be intergers.'
            assert n >= 1, 'Principle quantum number can not be smaller than 1.'
            assert l < n and l >= 0, 'Angular momentum must be positive and smalled than principle quantum number.'

            # index is numbered from 1 to number of levels up to (nl)
            # offset by 1, since first line is a comment line
            return int((n-1)*n/2) + l # + 1
            
        def FitRate(recom_data, LogTs, n=1, l=0):
            index   = get_l_level_index(n=n, l=l)
            rate    = interpolate.interp1d(LogTs, recom_data.iloc[index, 2:43].values,fill_value="extrapolate", bounds_error=False)   
            return rate

        rates      = self.ReadRecombinationRates(self.TabulatedRecombinationRates)
        LogTs      = rates['LogTs']
        recom_data = rates['recom_data']
        
        nmax     = self.nmax
        Alpha_nl = {}
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                conf_i             = self.Config(n=n, l=l)
                Alpha_nl[conf_i]   = FitRate(recom_data, LogTs, n=n, l=l) # try 10**()
        return Alpha_nl

    def collisional_excitation_rate_Lebedev_Beigman(self, nu=6, nl=1, Te=1e4):
        Te = Te * unyt.K
        gnu = 2 * nu**2
        gnl = 2 * nl**2
        deltaE = self.Eion * (1 / nl**2 - 1 / nu**2)
        #print(deltaE, Te)
        qul = gnu / gnl * np.exp(- deltaE / (unyt.boltzmann_constant_cgs * Te)) * self.collisional_deexcitation_rate_Lebedev_Beigman(nu=nu, nl=nl, Te=Te.value)
        return qul

    def collisional_deexcitation_rate_Lebedev_Beigman(self, nu=6, nl=1, Te=1e4):
        Te = Te * unyt.K
        gnu = 2 * nu**2 # 2n**2
        gnl = 2 * nl**2
        alpha = 1. / 137.
        a0 = unyt.reduced_planck_constant_cgs / (unyt.electron_mass_cgs * unyt.c_cgs * alpha)
        Z = 1
        qlu = gnl / gnu * 2. * np.pi * a0**2 * alpha * unyt.c_cgs * nl * (nu / (Z * (nu - nl)))**3 * self.ftheta(nu=nu, nl=nl, Te=Te, Z=Z) * self.psi(nu=nu, nl=nl, Te=Te) / np.sqrt(self.get_theta(Te, Z))
        return qlu
        
    def get_theta(self, Te, Z):
        #print('temp is', Te)
        theta = unyt.boltzmann_constant_cgs * Te / (Z**2 * self.Eion)
        #print('theta is', theta)
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
        
    def FitCollisionalExRates(self):
        ''' 
        Provide fitting function for collisional excitation rate of the ground state. 
        
        The fitting funcition is of the form 
        
        Recombination_rate(n, l, T) = 10**Alpha_nk(10**LogT)
        '''
        
        def FitCollExRate(ups_data, Ts, n=1, l=0):

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

        
        ups     = self.ReadEffectiveCollisonalStrength(verbose=True)
        Ts      = ups['Ts']
        upsdata = ups['ups_data']
        nmax     = self.nmaxcoll
        q_nl = {}
        for n in np.arange(2, nmax+1): # mininum level is 2
            for l in np.arange(n):
                conf_i             = self.Config(n=n, l=l)
                
                coeffs             = FitCollExRate(upsdata, Ts, n=n, l=l)
                q_nl[conf_i]       = np.poly1d(coeffs) # add 10**
                
        return q_nl
    
    
    
    ##################################################################    
    #                Cascade matrix methods                          #
    ##################################################################
    def TestAllLevelPops(self, nH = 1.0, ne = 1.0, LogT = 4.0, N={}):
        '''
        Verify whether these pop levels satisfy the equilibrium relation - Eq. 4.1
        '''
        #
        nmax     = self.nmax
        Config   = self.Config
        Alpha_nl = self.Alpha_nl
        A        = self.A
        #
        TestConfig = []
        TestDiff   = []
        for n in np.arange(1, nmax+1):
            for l in np.arange(n):
                lhs    = 0.0
                conf   = Config(n=n, l=l)
                lhs   += nH * ne * 10**Alpha_nl[conf](LogT)
                #
                for nu in np.arange(n+1, nmax+1):
                    for lu in [l-1, l+1]:
                        if (lu>= 0) & (lu < nu):
                            conf_i = Config(n=nu, l=lu)
                            lhs += N[conf_i] * A[conf_i][conf]
                #
                rhs = 0.0
                for nd in np.arange(1, n):
                    for ld in [l-1, l+1]:
                        if (ld >= 0) & (ld < nd):
                            conf_k = Config(n=nd, l=ld)
                            rhs    += A[conf][conf_k]
                    if (nd == 1) & (n==2) & (l == 0):
                        ld      = 0
                        conf_k  = Config(n=nd, l=ld)
                        rhs    += A[conf][conf_k]
                #
                Nnl  = 0.0
                diff = 1e2
                if rhs > 0:
                    Nnl  = lhs / rhs
                    diff = (Nnl-N[conf])/N[conf] * 100.
#                     if n < 10:
#                         print("Conf = {0:s}, % diff = {1:1.4f}, N = {2:1.3e}".format(conf, diff, (N[conf])))
                TestConfig.append(conf)
                TestDiff.append(diff)
        return {'Conf':TestConfig, 'Diff':TestDiff}

    def ComputeLevelPop(self, nHII = 1.0, ne = 1.0, nHI=1.0, LogT=4.0, n=2, l=0, verbose=False):
        '''
        Compute level population for a given level  - implementents Eq. 4.10
        Input: 
           nH   = proton number density nH [cm^[-3]]
           ne   = electron number density [[cm^-3]]
           logT = logarithm of temperature
           n    = principle quantum number of desired level
           l    = angular momentum state of this level
        '''

        #
        # check if all the quantities have the same dimension
        # convert LogT to float
        LogT = float(LogT)
        
        nmax     = self.nmax
        A        = self.A
        C        = self.C
        Alpha_nl = self.Alpha_nl
        q_nl     = self.q_nl
        Config   = self.Config

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

        N       = np.zeros_like(LogT)
        mask    = rhs > 0
        N[mask] = lhs[mask]/rhs[mask]
        if verbose:
            print("Computed level pop for level = {0:s}, log N = {1:2.4f}".format(conf_i, np.log10(N)))
        return N

    def ComputeAllLevelPops(self, nHII = 1.0, ne = 1.0, nHI = 1.0, LogT=4.0):
        '''
        Compute level population for all levels  - implementents Eq. 4.10
        Input: 
           nHII(nH in the previous version) = proton number density nH [cm^[-3]]
           nHI  = neutral hydrogen number density [cm^{-3}]
           ne   = electron number density [[cm^-3]]
           logT = logarithm of temperature
        '''
        
        #
        nmax     = self.nmax
        A        = self.A
        C        = self.C
        Alpha_nl = self.Alpha_nl
        q_nl     = self.q_nl
        Config   = self.Config
        

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
                    N[conf_i] = lhs/rhs
        return N
        
    def ComputeCascadeMatrix(self):
        '''
           Compute cascade matrix from Einstein coefficients
        '''
        import time
        nmax     = self.nmax          # max upper level
        A        = self.A             # Einstein coefficient
        verbose  = self.verbose
        Config   = self.Config
        topickle = True
        
        # if pickle file exists, read it
        if topickle:
            import pickle
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
            print(" ... Cascade matrix: probability matrix computed (eq. 4.8) in time {0:1.2f}".format(time.time()-t0))
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

                    
    def ReadTabulatedEinsteinCoefficients(self, fname):
        '''
          Read tabulated Einstein coefficients
          Use these to compute the casecade
        '''
        
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
                        
                    conf_k = self.Config(n=1, l=0) # ground state
                    for nu in np.arange(2, self.nmax+1):
                        conf_i = self.Config(n=nu, l=1) # all p-state
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
                conf_i = self.Config(n=nu, l=lu)
                A[conf_i] = {}
                # loop over lower level
                for nd in np.arange(nu):
                    for ld in np.arange(nd):
                        conf_k = self.Config(n=nd, l=ld)
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
            conf_i = self.Config(n=nup, l=lup)
            conf_k = self.Config(n=nd, l=ld)
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
        conf_i = self.Config(n=nu, l=lu)
        conf_k = self.Config(n=nd, l=ld)
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
            conf_k = self.Config(n=1, l=0) # ground state
            for nu in np.arange(2, nmax+1):
                conf_i = self.Config(n=nu, l=1) # p-state
                A[conf_i][conf_k] = 0.0
        
        return A

        
    def Config(self, n=1, l=1):
        '''
              configuration states are tuples of the form (n,l), where:
          n = principle quantum number, n=1->nmax
          l = angular momentum number, l=0->n-1
        '''
        return (n,l)
        
    def DeConfig(self, config='1s'):
        '''
            extract n and l value for a given configuration state
        '''
        return config[0], config[1]

    def alpha_effective(self, n=2, l=0, LogT=4.0):
        alpha_nl = 0 # based on Eq. 30 in Pengelly 1964 
        conf_k = self.Config(n=n, l=l) # loop over all l-states within this upper level
        # alpha * recom coeff
        for n in np.arange(n, self.nmax+1):
            for l in np.arange(n):
                conf_i = self.Config(n=n, l=l)
                if self.recom:
                    alpha_nl += 10**self.Alpha_nl[conf_i](LogT) * self.C[conf_i][conf_k]
        return alpha_nl

    def branching_ratio(self, nupper=3, nlower=2, LogT=4.0, caseB=True):
        if caseB == True:
            alpha_tot = self.AlphaB(LogT=LogT)
        else:
            alpha_tot = self.AlphaA(LogT=LogT)
        
        lterms = np.zeros(nupper)
        
        As = {}
        for lup in np.arange(nupper):
            conf_up  = self.Config(n=nupper, l=lup)
            Atemp    = 0.0
            for ldown in np.arange(nlower):
                conf_down = self.Config(n=nlower, l=ldown)
                try:
                    Atemp += self.A[conf_up][conf_down]
                except:
                    pass
            As[conf_up] = Atemp

        for lup in np.arange(nupper):
            lhs_rr = 0
            conf_k = self.Config(n=nupper, l=lup) # loop over all l-states within this upper level
            # alpha * recom coeff
            for n in np.arange(nupper, self.nmax+1):
                for l in np.arange(n):
                    conf_i = self.Config(n=n, l=l)
                    if self.recom:
                        lhs_rr += 10**self.Alpha_nl[conf_i](LogT) * self.C[conf_i][conf_k]
            
            rhs    = 0
            conf_i = self.Config(n=nupper, l=lup)
            
            for nd in np.arange(1, nupper):
                for ld in [lup-1, lup+1]:
                    if (ld >=0) & (ld < nd):
                        conf_k = self.Config(n=nd, l=ld)
                        rhs += self.A[conf_i][conf_k]
                if (nd == 1) & (nupper == 2) & (lup == 0):
                    ld     = 0
                    conf_k = self.Config(n=nd, l=ld)
                    rhs    += self.A[conf_i][conf_k]
            
            lterms[lup] = lhs_rr / rhs
            
        print('lterms', lterms)

        alpha_eff = 0
        for i, level_config in enumerate(As.keys()):
            alpha_eff += lterms[i] * As[level_config]
            print('alpha_eff', alpha_eff, level_config)
            print('alpha_a', alpha_tot)
        R = alpha_eff / alpha_tot
        return alpha_eff, R
