// Teast header!
// Toaster.h

#if !defined(_XDRESS_TOASTER_)
#define _XDRESS_TOASTER_

#include <string>

/*********************************************/
/*** Toaster Component Class and Functions ***/
/*********************************************/

namespace xdress {

  // Toaster class
  class Toaster {
  public:
    // Toaster Constructors
    Toaster();
    ~Toaster();
    
    // Public data
    std::string toastiness;
    unsigned int nslices;
    float rate;

    // Public access functions
    int make_toast(std::string when, unsigned int nslices=1);
  };

// end namespace
};

#endif
