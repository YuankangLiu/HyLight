Usage
=====

To import the package, type
.. code-block::
   
       import hylightpy

Then we can initialise the hydrogen class using
.. code-block::
   
       HI = hylightpy.HIAtom(nmax = 40, verbose=True, caseB=True,
                             recom=True, coll=False, cache_path='./cache/')

Here, we have initialse the class with 40 n-levels in Case B. We turn on the radiative recombination but not the collisioanal excitation from the ground state.

We also set the cache folder to the current working directory. The cache folder will store the cascade matrices and Einstein coefficients. 
