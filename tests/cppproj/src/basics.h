// Header for general library file.

#if !defined(_CPPPROJ_BASICS_)
#define _CPPPROJ_BASICS_

//standard libraries
#include <string>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <exception>
#include <sys/stat.h> 
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <typeinfo>

namespace cppproj {

// misc
extern std::string GET_LUCKY;
void voided ();
extern int verbosity;

typedef enum PersonID {
  JOAN,
  HOOVER,
  MULAN=42,
  LESLIE,
} PersonID;

// structs
typedef struct struct0 {
  char nuc_name[6];
  int nuc_zz;
  short thermal_yield;
  double xs [63];
} struct0;

// normal classes 

class A {
 public:
  A() {};
  A(int b) {};
  ~A() {};
  int a;
  virtual void call() {a=1;};
};

class B : public A {
 public:
  B() {};
  ~B() {};
  int b;
  virtual void call() {b=1;};
  void from_a(A x) {b = x.a;};
};

class C : public B {
 public:
  C() {};
  ~C() {};
  int c;
  virtual void call() {c=1;};
};

// templated classes
template <class T>
class TClass0 {
 public:
  int row;
  int col;
  T val;

  TClass0() {};
  ~TClass0() {};

  TClass0(int i, int j, T v) {
    row = i;
    col = j;
    val = v;
  };

  // Takes life, the universe, and everything and returns 42
  // Templated functions on templated classes are not supported by GCC-XML
  template <class U> int whatstheanswer(U u){return 42;};
};


template <class T> class TClass2 : public TClass0<T> {
  public:
    TClass2( bool default_Arg = true) {}
    T bob;
};

template <class T>
class TClass1 {
 public:
  int N, nrows, ncols;
  std::vector< TClass0<T> > sm;

  TClass1(){};
  ~TClass1(){};

  TClass1(int n, int nr=0, int nc=0) {
    n = N;
    nrows = nr;
    ncols = nc;
    sm = std::vector< TClass0<T> >();
    sm.reserve(N);
  };

  void sort_by_row() {};
  int size() {return sm.size();};

  std::vector< std::vector<T> > todense() {
    typename std::vector< std::vector<T> > M = std::vector< std::vector<T> > (nrows, std::vector<T> (ncols, 0.0));
    return M;
  };

  void push_back(int i, int j, T value) {
    sm.push_back(TClass0<T>(i, j, value));
  };

  T at(int i, int j) {
    T x = 0;
    return x;
  };

  void prune(double precision = 1E-10) {};

  TClass1<T> transpose() {
    TClass1<T> B = TClass1<T>(N, nrows, ncols);
    return B;
  };

  friend std::ostream& operator<< (std::ostream& out, TClass1<T> & A) {
    out << "TClass 1 wuz h3r3.\n";
    return out;
  };
    
  TClass1<T> operator* (double s) {
    TClass1<T> B = TClass1<T>(N, nrows, ncols);
    return B;
  };

  std::vector<double> operator* (std::vector<double> vec) {
    std::vector<double> new_vec = std::vector<double>(vec.size(), 0.0);
    return new_vec;
  };


  TClass1<T> operator* (TClass1<T> B) {
    TClass1<T> C = TClass1<T>(size(), nrows, ncols);
    return C;
  };

  TClass1<T> operator+ (TClass1<T> B) {
    TClass1<T> C = TClass1<T>(size(), nrows, ncols);
    return C;
  };
};

// regular functions
std::vector<double> func0(double, std::vector<double>);
bool func1(std::map<int, double>, std::map<int, double>);
std::vector< std::vector<int> > func2(std::vector<int> a, std::vector<int> b);
int func3(char *, char **, int = -1);

// FIXME when enums are implemented properly in C++, see #96
//int func4(PersonID id);
int func4(int id); 

// templated functions
template <class T>
bool cmp_by_row(TClass0<T> a, TClass0<T> b) {return true;};


// Template specializations
TClass0<bool> smebool;

std::vector< std::vector<int> > vvi;
TClass0<int> smeints;
TClass1<int> spints;

int smeints42 = smeints.whatstheanswer<float>(65.0);

TClass0<double> smedubs;
TClass1<double> spdubs;

std::vector<float> vf;
std::vector< std::vector<float> > vvf;
TClass0<float> smeflts;
TClass1<float> spflts;
TClass0<float> smeflts0;
TClass2<float> spflts2;

}; // namespace cppproj

// structs
typedef struct ThreeNums
{
  double a;
  double b;
  double c;
  double (*op)(double, double, double);
} ThreeNums;

double call_threenums_op_from_c(ThreeNums x); 

bool operator<(ThreeNums x, ThreeNums y);

// normal classes
class Untemplated {
public:
  Untemplated() {};
  ~Untemplated() {};
  template <class T> int templated_method(T x){return 42;};
  int untemplated_method(float x){return 42;};
};

// template functions
template <class T, class U> T findmin(T x, U y) {return (x < y ? x : y);};
template <class T, int U> bool lessthan(T x) {return (x < U ? true : false);};

// template specialzations
int fmif = findmin<int, float>(3, 6.0);
int fmii = findmin<int, int>(3, 4);
bool fmbb = findmin<bool, bool>(true, false);
double fmdi = findmin<double, float>(3.0, 6.0);
bool lti3 = lessthan<int, 3>(6);

Untemplated unt = Untemplated();
// Template member function also missed by GCC-XML
int untrtn = unt.templated_method<float>(65.0);

#ifdef XDRESS
std::vector<double> _temp0;
std::vector< std::vector<double> > _temp1;
#endif


#endif
