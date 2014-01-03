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

// Toaster class
class Toaster : Base<int,6+1> {
public:
  // Toaster Constructors
  Toaster(int slices=7, bool flag=false);
  ~Toaster();

  // Public fields
  std::string toastiness;
  unsigned int nslices;
  float rate;
  int (*fp)(float);
  std::vector<char> vec;

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

// Enums
enum Choices { CA, CB = 18-1 };

} // end namespace xdress

#endif
