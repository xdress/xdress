// Enrichment Parameters Class

#include "enrichment_parameters.h"



/*********************************/
/*** Enrichment Helper Classes ***/
/*********************************/
bright::EnrichmentParameters::EnrichmentParameters()
{
  alpha_0 = 0.0;
  Mstar_0 = 0.0;

  j = 0;
  k = 0;

  N0 = 0.0;
  M0 = 0.0;

  xP_j = 0.0;
  xW_j = 0.0;
};


bright::EnrichmentParameters::~EnrichmentParameters()
{
};


bright::EnrichmentParameters bright::fillUraniumEnrichmentDefaults()
{
  // Default enrichment paramters for uranium-based enrichment
  EnrichmentParameters ued; 

  ued.alpha_0 = 1.05;
  ued.Mstar_0 = 236.5;

  ued.j = 922350;
  ued.k = 922380;

  ued.N0 = 30.0;
  ued.M0 = 10.0;

  ued.xP_j = 0.05;
  ued.xW_j = 0.0025;

  return ued;
};
bright::EnrichmentParameters bright::UraniumEnrichmentDefaults(bright::fillUraniumEnrichmentDefaults());

