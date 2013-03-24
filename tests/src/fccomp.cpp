// Fuel Cycle Component Parent Class

#include "fccomp.h"

/**************************************************/
/*** Fuel Cycle Component Classes And Functions ***/
/**************************************************/

/***************************/
/*** Protected Functions ***/
/***************************/

void bright::FCComp::initialize (std::set<std::string> paramtrack, std::string n)
{
  // Protected Variables
  track_params = paramtrack;

  // Public Variables
  name = n;
  natural_name = n;
  if (natural_name.length() == 0)
    natural_name = "this_is_not_a_name";
    
  pass_num = 0;

  if (bright::write_text)
    initialize_text();
};

    
void bright::FCComp::initialize_text()
{
  // Initialize the Isotopic tracking file
  if (!bright::track_nucs.empty())
  {
    std::ofstream isofile ( (name + "Isos.txt").c_str() );
    isofile << "Isotope\n";
    for (std::set<int>::iterator iso = bright::track_nucs.begin(); iso != bright::track_nucs.end(); iso++)
      isofile << *iso << "\n"; 
    isofile.close();
  };

  // Initialize the Parameter tracking file.
  if (!track_params.empty())
  {	
    std::ofstream paramfile ( (name + "Params.txt").c_str() );
    paramfile << "Param\n";
    for ( std::set<std::string>::iterator p = track_params.begin(); p != track_params.end(); p++)
      paramfile << *p + "\n";
    paramfile.close();
  };
};




/***************************/
/*** FCComp Constructors ***/
/***************************/
    
bright::FCComp::FCComp (std::string n)
{
  // Parent class for all fuel cycle components.
  std::set<std::string> emptystrset;
  initialize(emptystrset, n);
};


bright::FCComp::FCComp (std::set<std::string> ptrack, std::string n)
{
  // Parent class for all fuel cycle components.
  initialize(ptrack, n);	
};


bright::FCComp::~FCComp ()
{
};


/*******************************/
/*** Public Access Functions ***/
/*******************************/

void bright::FCComp::calc_params ()
{
  // Placeholder function that sets the states of params_prior_calc and params_after_calc.
  for ( std::set<std::string>::iterator p2t = track_params.begin(); p2t != track_params.end(); p2t++)
  {
    params_prior_calc[*p2t] = 0.0;
    params_after_calc[*p2t] = 0.0;
  }
};


void bright::FCComp::write_mat_pass ()
{
};


void bright::FCComp::write_params_pass ()
{
  // Writes a single pass to the parameter tracking file.
  std::ifstream paramfilein  ( (name + "Params.txt").c_str() );
  std::stringstream parambuf;
  parambuf.precision(6);
  parambuf << std::scientific << std::uppercase;

  while (!paramfilein.eof() )
  {
    char line [3000];
    paramfilein.getline(line, 3000);
    parambuf << line;

    std::string paramflag = "Param";
    if (paramflag == "Param")
      parambuf << "\t" << "in\t\t" << "out\t";
    else if (0 < params_prior_calc.count(paramflag))
      parambuf << "\t" << params_prior_calc[paramflag] << "\t" << params_after_calc[paramflag];
    parambuf << "\n";
  };

  paramfilein.close();
  std::ofstream paramfileout ( (name + "Params.txt").c_str() );
  paramfileout <<  parambuf.rdbuf();
  paramfileout.close();
};


void bright::FCComp::write_text()
{
  // Write the isotopic streams
  write_mat_pass();

  // Write the parameters if they are there to write!
  if (!track_params.empty()) 
    write_params_pass();
};




void bright::FCComp::write()
{
  // Now that we are ready to start writing out data, let's update the pass number that we are on.
  pass_num++;

  // Set the parameters for this pass.
  // Uncommenting breaks python inheritance
  //if (!track_params.empty())
  //  calc_params();

  // Writes the output table files.
  if (bright::write_text)
    write_text();
};    

int bright::FCComp::calc()
{
  return 1;
};
