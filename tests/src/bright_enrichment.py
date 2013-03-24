mod_ds = """Python wrapper for enrichment."""

class_ds = \
"""Enrichment Fuel Cycle Component Class.  Daughter of FCComp.

Parameters
----------
ep : EnrichmentParameters, optional 
    This specifies how the enrichment cascade should be set up.  It is a 
    EnrichmentParameters instance.  If enrich_params is not specified, then 
    the cascade is initialized with values from uranium_enrichment_defaults().
n : str, optional
    The name of the enrichment fuel cycle component instance.

"""

desc = {
    'docstrings': {
        'module': mod_ds,
        'class': class_ds,
        'attrs': {},
        'methods': {},
        },
    'attrs': {
        'j': 'nucid',
        'k': 'nucid',
        },
    }


desc['docstrings']['attrs']['alpha_0'] = \
r"""The :math:`\\alpha_0` attribute specifies the overall stage separation factor
for the cascade.  This should be set on initialization.  Values should be
greater than one.  Values less than one represent de-enrichment.
""" 

desc['docstrings']['attrs']['Mstar_0'] = \
"""The :math:`M^*_0` represents a first guess at what the `Mstar` should be.
The value of Mstar_0 on initialization should be in the ballpark
of the optimized result of the Mstar attribute.  However, :math:`M^*_0` must
always have a value between the weights of the j and k key components.
"""

desc['docstrings']['attrs']['j'] = \
"""This is an integer in zzaaam-form that represents the jth key component.
This nuclide is preferentially enriched in the product stream.
For standard uranium cascades j is 922350 (ie U-235).
"""
desc['docstrings']['attrs']['k'] = \
"""This is an integer in zzaaam-form that represents the kth key component.
This nuclide is preferentially enriched in the waste stream.
For standard uranium cascades k is 922380 (ie U-238).
"""

desc['docstrings']['attrs']['N0'] = \
"""This is the number of enriching stages initially guessed by the user."""

desc['docstrings']['attrs']['M0'] = \
"""This is the number of stripping stages initially guessed by the user."""

desc['docstrings']['attrs']['xP_j'] = \
"""This is the target enrichment of the jth isotope in the
product stream mat_prod.  The :math:`x^P_j` value is set by 
the user at initialization or run-time.  For typical uranium 
vectors, this value is about U-235 = 0.05.
"""

desc['docstrings']['attrs']['xW_j'] = \
"""This is the target enrichment of the jth isotope in the
waste stream ms_tail.  The :math:`x^W_j` value is set by the 
user at initialization or runtime.  For typical uranium vectors,
this value is about U-235 = 0.0025.
"""

desc['docstrings']['attrs']['Mstar'] = \
r"""The :math:`M^*` attribute represents the mass for which the adjusted
stage separation factor, :math:`\\alpha^*_i`, is equal to one.  It is this
value that is varied to achieve an optimized enrichment cascade.
"""

desc['docstrings']['attrs']['mat_tail'] = \
"""In addition to the mat_feed and mat_prod materials, Enrichment
also has a tails or waste stream that is represented by this attribute.
The mass of this material and the ms_prod product material should always 
add up to the mass of the mat_feed feed stock.
"""

desc['docstrings']['attrs']['TotalPerFeed'] = \
"""This represents the total flow rate of the cascade divided by the 
feed flow rate.  As such, it shows the mass of material needed in the
cascade to enrich an additional kilogram of feed.  Symbolically,
the total flow rate is given as :math:`L` while the feed rate is
:math:`F`.  Therefore, this quantity is sometimes seen as 'L-over-F'
or as 'L/F'.  TotalPerFeed is the value that is minimized to form an 
optimized cascade.
"""

desc['docstrings']['attrs']['SWUperFeed'] = \
"""This value denotes the number of separative work units (SWU) required
per kg of feed for the specified cascade.
"""

desc['docstrings']['attrs']['SWUperProduct'] = \
"""This value is the number of separative work units (SWU) required
to produce 1 [kg] of product in the specified cascade.
"""

desc['docstrings']['methods']['initialize'] = \
"""The initialize method takes an enrichment parameter object and sets
the corresponding Enrichment attributes to the same value.

Parameters
----------
enrich_params : EnrichmentParameters
    A class containing the values to (re-)initialize an Enrichment cascade with.

"""

desc['docstrings']['methods']['calc_params'] = \
"""This sets the Enrichment parameters to the following 
values::

    self.params_prior_calc["MassFeed"] = self.mat_feed.mass
    self.params_after_calc["MassFeed"] = 0.0

    self.params_prior_calc["MassProduct"] = 0.0
    self.params_after_calc["MassProduct"] = self.mat_prod.mass

    self.params_prior_calc["MassTails"] = 0.0
    self.params_after_calc["MassTails"] = self.mat_tail.mass

    self.params_prior_calc["N"] = self.N
    self.params_after_calc["N"] = self.N

    self.params_prior_calc["M"] = self.M
    self.params_after_calc["M"] = self.M

    self.params_prior_calc["Mstar"] = self.Mstar
    self.params_after_calc["Mstar"] = self.Mstar

    self.params_prior_calc["TotalPerFeed"] = self.TotalPerFeed
    self.params_after_calc["TotalPerFeed"] = self.TotalPerFeed

    self.params_prior_calc["SWUperFeed"] = self.SWUperFeed
    self.params_after_calc["SWUperFeed"] = 0.0

    self.params_prior_calc["SWUperProduct"] = 0.0
    self.params_after_calc["SWUperProduct"] = self.SWUperProduct

"""

desc['docstrings']['methods']['calc'] = \
"""This method performs an optimization calculation on M* and solves for 
appropriate values for all Enrichment attributes.  This includes the 
product and waste streams flowing out of the the cascade as well.

Parameters
----------
input : dict or Material or None, optional
    If input is present, it is set as the component's mat_feed.  If input is 
    a nuclide mapping (zzaaam keys, float values), it is first converted into a 
    Material before being set as mat_feed.

Returns
-------
output : Material
    mat_prod

"""

desc['docstrings']['methods']['PoverF'] = \
r"""Solves for the product over feed enrichment ratio.

.. math::

    \\frac{p}{f} = \\frac{(x_F - x_W)}{(x_P - x_W)}

Parameters
----------
x_F : float
    Feed enrichment.
x_P : float
    Product enrichment.
x_W : float
    Waste enrichment.

Returns
-------
pfratio : float
    As calculated above.

"""

desc['docstrings']['methods']['WoverF'] = \
r"""Solves for the waste over feed enrichment ratio.

.. math::

    \\frac{p}{f} = \\frac{(x_F - x_P)}{(x_W - x_P)}

Parameters
----------
x_F : float
    Feed enrichment.
x_P : float
    Product enrichment.
x_W : float
    Waste enrichment.

Returns
-------
wfratio : float
    As calculated above.

"""
