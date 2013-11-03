// Teast header!
// Toaster.h

#if !defined(_XDRESS_TOASTER_)
#define _XDRESS_TOASTER_

#include <string>

/*********************************************/
/*** Toaster Component Class and Functions ***/
/*********************************************/

namespace xdress {

  template<class T, int i=0> struct Base {};

  // Toaster class
  class Toaster {
  public:
    // Toaster Constructors
    Toaster(int slices = 7);
    ~Toaster();
    
    // Public data
    std::string toastiness;
    unsigned int nslices;
    float rate;

    // Public access functions
    int make_toast(std::string when, unsigned int nslices=1);
    Base<float> templates(Base<int,3> strange);

  private:
    // Should not be described
    int hidden();
  };

// end namespace
};

#endif
