#include "basics.h"

namespace cppproj {

  B::B() {
    for(int i = 0; i < 3; ++i){
      std::vector<int> tmp1;
      for(int j = 0; j < 5; ++j){
        tmp1.push_back(j*i);
      }
      clist.push_back(tmp1);
      tmp1.clear();
    }
  }

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

std::set<int> setfunc(int a, int b, int c) {
  std::set<int> ret;
  ret.insert(a);
  ret.insert(b);
  ret.insert(c);
  return ret;
}

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
