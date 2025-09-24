Usage
=====

Initialisation
--------------

To import the package, type

.. code-block::
   
       import hylightpy

Then we can initialise the hydrogen class using

.. code-block::
   
       HI = hylightpy.HIAtom(nmax = 50, verbose=False, caseB=True,
                             recom=True, coll=True, cache_path='./')

Here, we have initialised the class with 40 n-levels in Case B. We turn on the radiative recombination but not the collisional excitation from the ground state.

We also set the cache folder to the current working directory. The cache folder will store the cascade matrices and Einstein coefficients. 

Line emissivity calculation
---------------------------

We utilise ``unyt`` package to specify the gas density and temperature.

.. code-block::
   
       import unyt


For a typical nebular condition (electron density of 100 :math:`\rm{cm}^{-3}`, proton density of 100 :math:`\rm{cm}^{-3}` and neutral hydrogen density of :math:`10^{-5}\,\rm{cm}^{-3}`) and temperature (:math:`10^4\,\rm{K}`)):

.. code-block::
   
       ne=unyt.array.unyt_array([1e2], 'cm**(-3)'),
       nHI=unyt.array.unyt_array([1e-5], 'cm**(-3)'),
       nHII=unyt.array.unyt_array([1e2], 'cm**(-3)'),
       temp=unyt.array.unyt_array([1e4], 'K')


We use the function ``get_emissivity`` to calculate the H :math:`\alpha` line emissivity (:math:`n_{\rm upper}=3 \to n_{\rm lower}=2`)  at the above condition.

.. code-block::

       eps = HI.get_line_emissivity(ne=ne,
                                    nHI=nHI, 
                                    nHII=nHII, 
                                    temp=temp, 
                                    nupper=3, nlower=2)
       print(print("Emissivity in the n={0:1d} to n={1:1d} line is {2:1.3e}".format(nupper, nlower, eps[0])))

Output::

       Emissivity in the n=3 to n=2 line is 3.451e-21 erg/(cm**3*s)
  


Level population
----------------

The function ``compute_level_pop`` computes the level population density. The following line calculates the 3 :math:`p` state (principle quantum number is 3, angular momentum quantum number is 1) population density at the same condition

.. code-block::

       levelpop = HI.compute_level_pop(nHII=nHII, 
                                       ne=ne, 
                                       nHI=nHI, 
                                       temp=temp, 
                                       n=3, l=1)
       print('Level population of 3p state is {0:1.2e}'.format(levelpop[0]))

Output::
  
       Level population of 3p state is 1.64e-17 cm**(-3)
 
