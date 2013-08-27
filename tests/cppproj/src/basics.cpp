#include "basics.h"

namespace cppproj {

std::string GET_LUCKY = "She's up all night til the sun.";
void voided() {};
int verbosity  = 0;

std::vector<double> func0(double x, std::vector<double> vec)
{
  std::vector<double> d (vec.size(), 0.0);
  return d;
};


bool func1(std::pair<int, double> i, std::pair<int, double> j)
{
  return true;
};


std::vector< std::vector<int> > func2(std::vector<int> a, std::vector<int> b)
{
  std::vector< std::vector<int> > c (a.size(), std::vector<int>(b.size(), 0.0)); 
  return c;
};


int func3(char * val, char ** arr, int arr_len)
{
  return -1;
};

int func4(PersonID id)
{
  return id;
};

} // namespace cppproj
