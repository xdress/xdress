// Enrichment.h
// Header for general Fuel Cycle Component Objects

#if !defined(_BRIGHT_ENRICHMENT_)
#define _BRIGHT_ENRICHMENT_

#include "fccomp.h"
#include "enrichment_parameters.h"

/************************************************/
/*** Enrichment Component Class and Functions ***/
/************************************************/

namespace bright {

  extern std::string enr_p2t [];
  extern std::set<std::string> enr_p2track;

  class Enrichment : public FCComp
  {
  // Reprocessing class
  public:
    // Reprocessing Constructors
    Enrichment(std::string n="");
    Enrichment(EnrichmentParameters ep, std::string n="");
    ~Enrichment ();

    // Public data
    double alpha_0;           // specify on init.
    double Mstar_0;           // specify on init.
    double Mstar;             // Current Mstar
    std::map<int, double> mat_tail;  // Waste Stream

    // key isotopic info
    int j;          // The jth isotope is the key, in zzaaam form, must be in mat_feed.
    int k;          // The kth isotope is the other key to separate j away from.
    double xP_j;    // Product enrichment of jth isotope
    double xW_j;    // Waste/Tails enrichment of the jth isotope

    // Stage info
    double N;       // N Enriching Stages
    double M;       // M Stripping Stages
    double N0;      // initial guess of N-stages
    double M0;      // initial guess of M-stages

    // Flow Rates
    double TotalPerFeed;    // Total flow rate per feed rate.
    double SWUperFeed;      // This is the SWU for 1 kg of Feed material.
    double SWUperProduct;   // This is the SWU for 1 kg of Product material.


    // Public access functions
    void initialize(EnrichmentParameters ep);		// Initializes the constructors.
    void calc_params();
    int calc();

    double PoverF(double x_F, double x_P, double x_W);
    double WoverF(double x_F, double x_P, double x_W);

    double get_alphastar_i(double M_i);

    double get_Ei(double M_i);
    double get_Si(double M_i);
    void FindNM();

    double xP_i(int i);
    double xW_i(int i);
    void SolveNM();
    void Comp2UnitySecant();
    void Comp2UnityOther();
    double deltaU_i_OverG(int i);
    void LoverF();
    void MstarOptimize();
  };


  /******************/
  /*** Exceptions ***/
  /******************/
  class EnrichmentInfiniteLoopError: public std::exception
  {
    virtual const char* what() const throw()
    {
      return "Inifinite loop found while calculating enrichment cascade!  Breaking...";
    };
  };


  class EnrichmentIterationLimit: public std::exception
  {
    virtual const char* what() const throw()
    {
      return "Iteration limit hit durring enrichment calculation!  Breaking...";
    };
  };


  class EnrichmentIterationNaN: public std::exception
  {
    virtual const char* what() const throw()
    {
      return "Iteration has hit a point where some values are not-a-number!  Breaking...";
    };
  };

// end bright
};

#endif


