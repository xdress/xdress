#include "basics.h"

namespace cppproj {

std::string GET_LUCKY = "She's up all night til the sun.";
void voided() {};
int verbosity  = 0;

std::vector<double> func0(double x, std::vector<double> vec) {
  std::vector<double> d (vec.size(), 0.0);
  return d;
};


bool func1(std::map<int, double> i, std::map<int, double> j) {
  return i.size() < j.size();
};


std::vector< std::vector<int> > func2(std::vector<int> a, std::vector<int> b) {
  std::vector< std::vector<int> > c (a.size(), std::vector<int>(b.size(), 0.0)); 
  return c;
};


int func3(char * val, char ** arr, int arr_len) {
  return -1;
};

void call_with_void_fp_struct(VoidFPStruct x) {
  x.op(10);
};

// FIXME #96
//int func4(PersonID id)
int func4(int id) {
  return id;
};

} // namespace cppproj

double call_threenums_op_from_c(ThreeNums x) {
  return x.op(x.a, x.b, x.c);
}

bool operator<(ThreeNums x, ThreeNums y) {
  return (x.a < y.a) && (x.b < y.b) && (x.c < y.c);
};
