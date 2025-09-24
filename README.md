# HyLight

HyLight (HYdrogen recombination LIne emission from ionized Gas in varying tHermal condiTions) calculates the level population of the excited states in atomic hydrogen and hence the typical recombination line emissivity. The results are accurate under typical photoionised nebular conditions. 

## Installation

The package can be installed via
```
pip install hylightpy
```
## Examples

Example usage can be found on [Google Colab](https://colab.research.google.com/drive/1H6TPbzPtAu9vaII_YNPJStM4WmlDHWS-?usp=sharing), 

or see the [`examples/` folder](https://github.com/YuankangLiu/HyLight/tree/main/examples). 

## Example usage

To import the package, type

```
import hylightpy
```

### Initialisation
Then initialise the class using 
```
HI = hylightpy.HIAtom(nmax = 50, verbose=True, caseB=True, 
                      recom=True, coll=True, 
                      cache_path='./')
```
where the user specify the number of levels in the hydrogen atom, the Case, and whether to include radiative processes and collisional processes. The user also have the freedom to specify the cache folder path, which will be storing cascade matrix elements. By default, the code will use the currect working directory as to store the cache. 

### Line emissivity calculation

We utilise `unyt` package to specify the gas density and temperature. 

```
import unyt
```

In the following code block, we specify typical nebular conditions:

```
ne=unyt.array.unyt_array([1e2], 'cm**(-3)')
nHI=unyt.array.unyt_array([1e-5], 'cm**(-3)')
nHII=unyt.array.unyt_array([1e2], 'cm**(-3)')
temp=unyt.array.unyt_array([1e4], 'K')
```

Then we use the function `get_emissivity` to calculate the line emissivity at a given density and temperature. 

```
eps = HI.get_line_emissivity(ne=ne, 
                             nHI=nHI, 
                             nHII=nHII, 
                             temp=temp, 
                             nupper=3, nlower=2)
```
The above line calculates the H $\alpha$ line emissivity at a given gas density (electron density of 100 $\rm{cm}^{-3}$, proton density of 100 $\rm{cm}^{-3}$ and neutral hydrogen density of 1e-5 $\rm{cm}^{-3}$) and temperature (1e4 K). 

### Level population

The function `compute_level_pop` computes the level popualtion density. The following line calculates the 3 $p$ state population density at the same condition:
```
HI.compute_level_pop(nHII=nHII, 
                     ne=ne, 
                     nHI=nHI, 
                     temp=temp, 
                     n=3, l=1)
```

More examples can be found in the `examples/` folder. 

## Documentation

The documentation is available [here](https://yuankangliu.github.io/HyLight/).

## Method

The method is described in Liu et al. 2025. 

## License

This package is released under the permissive MIT license. 

## How to cite?

If you use this code in your work, we kindly ask you to cite Liu et al. 2025. 