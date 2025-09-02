# HyLight

HyLight (HYdrogen recombination LIne emission from ionized Gas in varying tHermal condiTions) calculates the level population of the excited states in atomic hydrogen and hence the typical recombination line emissivity. The results are accurate under typical photoionised nebular conditions. 

## Installation

The package can be installed via
```
pip install hylightpy
```
## Example usage

To import the package, type

```
import hylightpy
```

Then initialise the class using 
```
HI = hylightpy.HILevelPopulations(nmax = 10, verbose=True, caseB=True, 
                                  recom=True, coll=False, 
                                  cache_path='./cache/')
```
where the user specify the number of levels in the hydrogen atom, the Case, and whether to include radiative processes and collisional processes. The user also have the freedom to specify the cache folder path, which will be storing cascade matrix elements. 

Then call the function above to compute the level popualtion density. We use 3p state as an example (n = 3, l = 1):
```
HI.ComputeLevelPop(ne=1e2, nHI=1e-5, nHII=1e2, LogT=4., n=3, l=1)
```
Here, the user has to provide electron, HI and HII densities as well as the temperature. 

More examples can be found in the `examples/` folder. 

## Method

The method is described in Liu et al. 2025. 

## License

This package is released under the permissive MIT license. 

## How to cite?

If you use this code in your work, we kindly ask you to cite Liu et al. 2025. 