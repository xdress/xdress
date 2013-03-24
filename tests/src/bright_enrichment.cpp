// Enrichment Component Class

#include "bright_enrichment.h"



std::string bright::enr_p2t [] = {"MassFeed", "MassProduct", "MassTails", "N", "M", \
                                  "Mstar", "TotalPerFeed", "SWUperFeed", "SWUperProduct"};
std::set<std::string> bright::enr_p2track (enr_p2t, enr_p2t+9);

/************************************************/
/*** Enrichment Component Class and Functions ***/
/************************************************/

/***************************/
/*** Protected Functions ***/
/***************************/

void bright::Enrichment::initialize(EnrichmentParameters ep)
{
  // Initializes the enrichment component. 
  alpha_0 = ep.alpha_0;
  Mstar_0 = ep.Mstar_0;
  j       = ep.j;
  k       = ep.k;
  N0      = ep.N0;
  M0      = ep.M0;
  xP_j    = ep.xP_j;
  xW_j    = ep.xW_j;
};


/*******************************/
/*** Enrichment Constructors ***/
/*******************************/

bright::Enrichment::Enrichment(std::string n) : bright::FCComp(enr_p2track, n)
{
  // Enrichmenting Fuel Cycle Component.  Applies Separation Efficiencies.
  initialize(bright::UraniumEnrichmentDefaults);
}


bright::Enrichment::Enrichment(bright::EnrichmentParameters ep, std::string n) : bright::FCComp(enr_p2track, n)
{
  // Enrichmenting Fuel Cycle Component.  Applies Separation Efficiencies.
  initialize(ep);
};

bright::Enrichment::~Enrichment()
{
}


/************************/
/*** Public Functions ***/
/************************/

void bright::Enrichment::calc_params()
{
  params_prior_calc["MassFeed"]  = 1.0;
  params_after_calc["MassFeed"] = 0.0;	

  params_prior_calc["MassProduct"]  = 0.0;
  params_after_calc["MassProduct"] = 1.0;	

  params_prior_calc["MassTails"]  = 0.0;
  params_after_calc["MassTails"] = 1.0;	

  params_prior_calc["N"]  = N;
  params_after_calc["N"] = N;	

  params_prior_calc["M"]  = M;
  params_after_calc["M"] = M;	

  params_prior_calc["Mstar"]  = Mstar;
  params_after_calc["Mstar"] = Mstar;	

  params_prior_calc["TotalPerFeed"]  = TotalPerFeed;
  params_after_calc["TotalPerFeed"] = TotalPerFeed;	

  params_prior_calc["SWUperFeed"]  = SWUperFeed;
  params_after_calc["SWUperFeed"] = 0.0;	

  params_prior_calc["SWUperProduct"]  = 0.0;
  params_after_calc["SWUperProduct"] = SWUperProduct;	
};


int bright::Enrichment::calc()
{
  return 2;
};


double bright::Enrichment::PoverF(double x_F, double x_P, double x_W)
{
  // Product over Feed Enrichment Ratio
  return ((x_F - x_W)/(x_P - x_W));
}

double bright::Enrichment::WoverF(double x_F, double x_P, double x_W)
{
  // Waste over Feed Enrichment Ratio
  return ((x_F - x_P)/(x_W - x_P));
}


double bright::Enrichment::get_alphastar_i (double M_i)
{
  // M_i is the mass of the ith isotope    
  return pow(alpha_0, (Mstar - M_i));
}

double bright::Enrichment::get_Ei (double M_i)
{
  double alphastar_i = get_alphastar_i(M_i);
  return ((alphastar_i - 1.0) / (1.0 - pow(alphastar_i, -N) ));
};


double bright::Enrichment::get_Si (double M_i)
{
  double alphastar_i = get_alphastar_i(M_i);
  return ((alphastar_i - 1.0)/(pow(alphastar_i, M+1) - 1.0));
};

void bright::Enrichment::FindNM()
{
};
  

double bright::Enrichment::xP_i(int i)
{
  return 65.0;
};


double bright::Enrichment::xW_i(int i)
{
  return 42.0;
};


void bright::Enrichment::SolveNM()
{
};


void bright::Enrichment::Comp2UnitySecant()
{
};


// I have serious doubts that this works...
void bright::Enrichment::Comp2UnityOther()
{
};


double bright::Enrichment::deltaU_i_OverG(int i)
{
  return 1.0;
};


void bright::Enrichment::LoverF()
{
};


void bright::Enrichment::MstarOptimize()
{
};
