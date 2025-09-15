import unittest
import unyt
import hylightpy
import numpy as np

class TestHylightpy(unittest.TestCase):

    def test_Einstein_A_values_case_A(self):
        test_HI = hylightpy.HIAtom(nmax=10, caseB=False, verbose=True)
        test_A = test_HI.A[(2,1)][(1,0)]
        self.assertEqual(test_A, 626159000)

    def test_Einstein_A_values_case_B(self):
        test_HI = hylightpy.HIAtom(nmax=10, caseB=True, verbose=True)
        test_A = test_HI.A[(2,1)][(1,0)]
        self.assertEqual(test_A, 0.0)

    def test_level_population_nmax_100_case_b(self):
        test_HI = hylightpy.HIAtom(nmax=100, recom=True, coll=False, 
                                   caseB=True, verbose=True)
        test_levelpop = test_HI.compute_level_pop(nHII=1e2 * unyt.cm**(-3), 
                                                  ne=1e2 * unyt.cm**(-3),
                                                  nHI=1e-5 * unyt.cm**(-3), 
                                                  temp=1e4 * unyt.K, 
                                                  n=3, l=0)
        self.assertLessEqual(np.abs(test_levelpop - 2.4209498e-17), 1e-6)

    def test_level_population_nmax_50_case_b(self):
        test_HI = hylightpy.HIAtom(nmax=50, recom=True, coll=False, 
                                   caseB=True, verbose=True)
        test_levelpop = test_HI.compute_level_pop(nHII=1e2 * unyt.cm**(-3), 
                                                  ne=1e2 * unyt.cm**(-3),
                                                  nHI=1e-5 * unyt.cm**(-3), 
                                                  temp=1e4 * unyt.K, 
                                                  n=3, l=0)
        self.assertLessEqual(np.abs(test_levelpop - 2.41359247e-17), 1e-6)

    def test_get_emissivity_nmax_50_case_b(self):
        test_HI = hylightpy.HIAtom(nmax=50, recom=True, coll=False, 
                                   caseB=True, verbose=True)
        test_emis = test_HI.get_emissivity(nHII = 1e2 * unyt.cm**(-3), 
                                           nHI  = 1e-5 * unyt.cm**(-3),
                                           ne   = 1e2 * unyt.cm**(-3), 
                                           temp = 1e4 * unyt.K,
                                           nupper=3, nlower=2)
        self.assertLessEqual(np.abs(test_emis.value - 3.45088986e-21), 1e-6)
        
    def test_get_emissivity_nmax_100_case_b(self):
        test_HI = hylightpy.HIAtom(nmax=100, recom=True, coll=False, 
                                   caseB=True, verbose=True)
        test_emis = test_HI.get_emissivity(nHII = 1e2 * unyt.cm**(-3), 
                                           nHI  = 1e-5 * unyt.cm**(-3),
                                           ne   = 1e2 * unyt.cm**(-3), 
                                           temp = 1e4 * unyt.K,
                                           nupper=3, nlower=2)
        self.assertLessEqual(np.abs(test_emis.value - 3.50935979e-21), 1e-6)

    def test_solution_to_equation(self):
        test_HI = hylightpy.HIAtom(nmax=20, recom=True, coll=False, 
                                   caseB=True, verbose=True)

        N = test_HI.compute_all_level_pops(nHII = 100 * unyt.cm**(-3), 
                                           ne = 100 * unyt.cm**(-3), 
                                           nHI = 1e-5 * unyt.cm**(-3), 
                                           temp=1e4 * unyt.K)
        
        nH = 100.
        ne = 100.
        LogT = 4.0
        
        nmax     = test_HI.nmax
        Config   = test_HI.config
        Alpha_nl = test_HI.Alpha_nl
        A        = test_HI.A
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
                TestConfig.append(conf)
                TestDiff.append(diff)

        # Starting from 3, skip 2s, 2p state
        for k in np.arange(3, len(TestDiff)):
            self.assertLessEqual(TestDiff[k], 1e-10)

if __name__ == '__main__':
    unittest.main()
