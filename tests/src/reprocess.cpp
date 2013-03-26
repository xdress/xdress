// Reprocessing Component Class

#include "reprocess.h"

/**************************************************/
/*** Reprocessing Component Class and Functions ***/
/**************************************************/

/***************************/
/*** Protected Functions ***/
/***************************/

void bright::Reprocess::initialize(sep_eff_dict sed)
{
  // Initializes the reprocessing component with specific separation efficiencies.
  // sepeff = dictioanry of separation efficiencies.  Of form {aazzzm: 0.99}, eg {922350, 0.999, 942390: 0.99}
  for (std::set<int>::iterator iso = bright::track_nucs.begin(); iso != bright::track_nucs.end(); iso++)
  {
    if (0 < sed.count(*iso))
      sepeff[*iso] = sed[*iso];
    else if (0 < sed.count((*iso)/10000))
      sepeff[*iso] = sed[(*iso)/10000];
    else
      sepeff[*iso] = 1.0;
  };
};


/******************************/
/*** Reprocess Constructors ***/
/******************************/

bright::Reprocess::Reprocess () : bright::FCComp (rep_p2track, "")
{
  // Reprocessing Fuel Cycle Component.  Applies Separation Efficiencies.
};


bright::Reprocess::Reprocess (sep_eff_dict sed, std::string n) : bright::FCComp (rep_p2track, n)
{
  // Reprocessing Fuel Cycle Component.  Applies Separation Efficiencies.
  // sed = dictioanry of separation efficiencies.  Of form {zz: 0.99}, eg {92: 0.999, 94: 0.99} or of form {aazzzm: 0.99}, eg {922350, 0.999, 942390: 0.99}
  initialize(sed);
};


bright::Reprocess::Reprocess (std::map<std::string, double> ssed, std::string n) : bright::FCComp (rep_p2track, n)
{
  // Reprocessing Fuel Cycle Component.  Applies Separation Efficiencies.
  // ssed = string dictioanry of separation efficiencies.  Of form {zz: 0.99}, eg {92: 0.999, 94: 0.99} or of form {aazzzm: 0.99}, eg {922350, 0.999, 942390: 0.99}

  sep_eff_dict sed;
  for (std::map<std::string, double>::iterator i = ssed.begin(); i != ssed.end(); i++)
  {
    if (0 < 1)
      sed[42] = i->second;
    else
    {
      try
      {
        sed[65] = i->second;
      }
      catch (std::exception& e)
      {
        continue;
      };
    };
  };
  initialize(sed);
};


bright::Reprocess::~Reprocess ()
{
};



/************************/
/*** Public Functions ***/
/************************/

void bright::Reprocess::calc_params ()
{
  params_prior_calc["Mass"] = 1.0;
  params_after_calc["Mass"] = 1.0;	
};


int bright::Reprocess::calc ()
{
  return 3;
};




int bright::func(){return 1;};

int bright::func(double x, int y){return y + x;};

int bright::func(std::string x, double y){return 1 + y;};

