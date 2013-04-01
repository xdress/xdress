#if !defined(_XDRESS_NEW_INPLACE_)
#define _XDRESS_NEW_INPLACE_

#include <string.h>

template <class T>
class NewInplace
{
  public:
    NewInplace(){};
    ~NewInplace(){};
    T * reinit(void * ptr){
      return new (ptr) T();
    };

    void delnull(T * ptr){
      T * val = new T();
      //*val = *ptr;
      memcpy((void *) val, (void *) ptr, sizeof(T));
      delete(val);
      new (ptr) T();
      //for(int i=0; i < sizeof(T); i++)
      //  ((void *) ptr)[i] = NULL;
      //ptr = (T *) NULL;
    };
};
#endif
