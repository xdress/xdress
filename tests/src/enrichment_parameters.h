// enrichment_parameters.h
// Header for general Fuel Cycle Component Objects

#if !defined(_BRIGHT_ENRICHMENT_PARAMETERS_)
#define _BRIGHT_ENRICHMENT_PARAMETERS_

#include "bright.h"

/************************************************/
/*** Enrichment Component Class and Functions ***/
/************************************************/

namespace bright {

  class EnrichmentParameters 
  {
    /** Set of physical parameters used to specify an enrichment cascade. **/

  public:
    // Constructors
    EnrichmentParameters();
    ~EnrichmentParameters();

    // Attributes
    double alpha_0; //Initial stage separation factor
    double Mstar_0; //Initial guess for mass separation factor

    int j; //Component to enrich (U-235), zzaaam form
    int k; //Component to de-enrich, or strip (U-238), zzaaam form

    double N0; //Initial guess for the number of enriching stages
    double M0; //Initial guess for the number of stripping stages

    double xP_j; //Enrichment of the jth isotope in the product stream
    double xW_j; //Enrichment of the jth isotope in the waste (tails) stream
  };

  EnrichmentParameters fillUraniumEnrichmentDefaults();
  extern EnrichmentParameters UraniumEnrichmentDefaults;

// end bright
};

#endif


