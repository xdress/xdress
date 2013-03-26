// Reprocess.h

#if !defined(_BRIGHT_REPROCESS_)
#define _BRIGHT_REPROCESS_

#include "fccomp.h"

/**************************************************/
/*** Reprocessing Component Class and Functions ***/
/**************************************************/

namespace bright {

  typedef std::map<int, double> sep_eff_dict;
  typedef sep_eff_dict::iterator sep_eff_iter;

  static std::string rep_p2t [] = {"Mass"};
  static std::set<std::string> rep_p2track (rep_p2t, rep_p2t+1);

  class Reprocess : public FCComp
  {
  // Reprocessing class
  public:
    // Reprocessing Constructors
    Reprocess();
    Reprocess(sep_eff_dict sed, std::string n="");
    Reprocess(std::map<std::string, double> ssed, std::string n="");
    ~Reprocess();
    
    // Public data
    sep_eff_dict sepeff;			// separation efficiency dictionary

    // Public access functions
    void initialize(sep_eff_dict sed);		// Initializes the constructors.
    void calc_params();
    int calc();
  };


  int func();
  int func(double x, int y=10);
  int func(std::string x, double y=10.0);

// end namespace
};

#endif
