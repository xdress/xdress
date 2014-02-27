// Teast header!
// Toaster.h

#if !defined(_XDRESS_TOASTER_)
#define _XDRESS_TOASTER_

#include <string>
#include <vector>

/*********************************************/
/*** Toaster Component Class and Functions ***/
/*********************************************/

namespace xdress {

template<class T, int i=0> struct Base {
  // Fields in template classes
  int field;

  // Functions in template classes
  void base(int a=1);
};

// Enums
enum Choices { CA, CB = 18-1 };

// Boolean and enum template params
template<bool B=false, Choices C=CA> class Point {};
template<> class Point<true,CA> {};

// Toaster class
class Toaster : Base<int,6+1> {
public:
  // Toaster Constructors
  Toaster(int slices=7, bool flag=false);
  // Constructor with string literal default arg
  Toaster(double d, std::string arg = "\n");
  ~Toaster();

  // Public fields
  std::string toastiness;
  unsigned int nslices;
  float rate;
  char array[10];
  int (*fp)(float);
  std::vector<char> vec;

  // Default enum values
  void make_choice(Choices a=CA, Choices b=CB);

  // Public access functions
  int make_toast(std::string when, unsigned int nslices=1, double dub=3e-8);
  Base<float> templates(Base<int,3> strange);

  // Test more types
  const int const_(const int c) const;
  int* pointers(int* a, const int* b, int* const c, const int* const d);
  int& reference(int& a, const int& b);

private:
  // Should not be described
  int hidden();
};

// Free functions
int simple(float s);
template<int i,class A,class B> int lasso(A a, const B& b);
extern template int lasso<17>(int,const float&);
extern template int lasso<18>(int,const float&);
void twice(int); // declaration without argument name
void twice(int x) {} // definition with argument name
void conflict(int good); // first declaration with correct argument name
void conflict(int bad); // second declaration with wrong argument name

// Classes should have implicit default constructors only if their parents do
struct Default : Base<int,7> {};
struct NoDefaultBase { NoDefaultBase(int i); };
struct NoDefault : NoDefaultBase { NoDefault(int i); };

} // end namespace xdress

#endif
