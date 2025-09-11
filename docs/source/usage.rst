Usage
=====

Initialisation
--------------

To import the package, type
.. code-block::
   
       import hylightpy

Then we can initialise the hydrogen class using
.. code-block::
   
       HI = hylightpy.HIAtom(nmax = 40, verbose=True, caseB=True,
                             recom=True, coll=False, cache_path='./cache/')

Here, we have initialised the class with 40 n-levels in Case B. We turn on the radiative recombination but not the collisioanal excitation from the ground state.

We also set the cache folder to the current working directory. The cache folder will store the cascade matrices and Einstein coefficients. 

Line emissivity calculation
---------------------------

We utilise ``unyt`` package to specify the gas density and temperature.

.. code-block::
   
       import unyt

We use the function ``get_emissivity`` to calculate the line emissivity at a given density and temperature.

.. code-block::

       HI.get_line_emissivity(ne=100. * unyt.cm**(-3), 
                              nHI=1e-5 * unyt.cm**(-3), 
                              nHII=100. * unyt.cm**(-3), 
                              temp=1e4 * unyt.K, 
                              nupper=3, nlower=2)

The above line calculates the H :math:`\alpha` line emissivity at a given gas density (electron density of 100 :math:`\rm{cm}^{-3}`, proton density of 100 :math:`\rm{cm}^{-3}` and neutral hydrogen density of :math:`10^{-5}\,\rm{cm}^{-3}`) and temperature (:math:`10^4\,\rm{K}`). 

Level population
----------------

The function ``compute_level_pop`` computes the level population density. The following line calculates the 3 :math:`p` state population density at the same condition

.. code-block::

       HI.compute_level_pop(nHII=100. * unyt.cm**(-3), 
                            ne=100. * unyt.cm**(-3), 
                            nHI=1e-5 * unyt.cm**(-3), 
                            temp=1e4 * unyt.K, 
                            n=3, l=1)


Now we can calculate the level population of the excited states. For examples, we are intersted in the 3s state. 
