class_ds = \
"""Reprocess Fuel Cycle Component Class.  Daughter of FCComp class.

Parameters
----------
sepeff : dict or map or None, optional 
    A dictionary containing the separation efficiencies (float) to initialize
    the instance with.  The keys of this dictionary may be strings or ints::

        # ssed = string dictionary of separation efficiencies.  
        # Of form {zz: 0.99}, eg 
        ssed = {92: 0.999, "94": 0.99} 
        # of form {LL: 0.99}, eg 
        ssed = {"U": 0.999, "PU": 0.99} 
        # or of form {mixed: 0.99}, eg 
        ssed = {"U235": 0.9, 922350: 0.999, "94239": 0.99}

name : str, optional
    The name of the reprocessing fuel cycle component instance.

"""

sepeff_ds = \
"""This is a dictionary or map representing the separation 
efficiencies of each isotope in bright.bright_conf.track_nucs. Therefore 
it has zzaaam-integer keys and float (double) values. During initialization, 
other SE dictionaries are converted to this standard form::

    sepeff = {922350: 0.999, 942390: 0.99}

"""


init_ds = """initialize(sepeff)
The initialize() function calculates the sepeff from an integer-keyed 
dictionary of separation efficiencies.  The difference is that sepdict 
may contain either elemental or isotopic keys and need not contain every 
isotope tracked.  On the other hand, sepeff must have only zzaaam keys 
that match exactly the isotopes in bright.track_nucs.

Parameters
----------
sepeff : dict or other mappping
    Integer valued dictionary of SE to be converted to sepeff.
        
"""

calc_params_ds = """calc_params()
Here the parameters for Reprocess are set.  For reprocessing, this amounts 
to just a "Mass" parameter::

    self.params_prior_calc["Mass"] = self.mat_feed.mass
    self.params_after_calc["Mass"] = self.mat_prod.mass

"""


calc_ds = """calc(input=None)
This method performs the relatively simply task of multiplying the current 
input stream by the SE to form a new output stream::

    incomp  = self.mat_feed.mult_by_mass()
    outcomp = {}
    for iso in incomp.keys():
        outcomp[iso] = incomp[iso] * sepeff[iso]
    self.mat_prod = Material(outcomp)
    return self.mat_prod

Parameters
----------
input : dict or Material or None, optional 
    If input is present, it set as the component's mat_feed.  If input is a 
    isotopic dictionary (zzaaam keys, float values), this dictionary is first 
    converted into a Material before being set as mat_feed.

Returns
-------
output : Material
    mat_prod

"""


desc = {
    'docstrings': {
        'class': class_ds,
        'attrs': {
            'sepeff_ds': sepeff_ds,
            },
        'methods': {
            'initialize': init_ds, 
            'calc_params': calc_params_ds,
            'calc': calc_ds,
            },
        },
    'attrs': {
        'sepeff': 'sepeff_t',
        },
    'methods': {
        ('Reprocess', ('sepeff', 'sepeff_t'), ('name', 'str', '""')): None,
        },
    }

mod = {'Reprocess': desc,
       'func': {
            'docstring': "I am a weird function."
            },
       'docstring': "Python wrapper for Reprocess.",}

from xdress.typesystem import TypeSystem

ts = TypeSystem()
ts.register_refinement('sepeff_t', ('map', 'int32', 'float64'),
    cython_cyimport='xdtest.typeconverters', cython_pyimport='xdtest.typeconverters',
    cython_py2c='xdtest.typeconverters.sepeff_py2c({var})',)

