// Teast header!
// Toaster.h

#if !defined(_XDRESS_TOASTER_)
#define _XDRESS_TOASTER_

/*********************************************/
/*** Toaster Component Class and Functions ***/
/*********************************************/

namespace xdress {

  class Toaster :
  {
  // Toaster class
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
