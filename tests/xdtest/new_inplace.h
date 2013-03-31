#if !defined(_XDRESS_NEW_INPLACE_)
#define _XDRESS_NEW_INPLACE_

template <class T>
class NewInplace
{
  public:
    NewInplace(){};
    ~NewInplace(){};
    T * reinit(void * ptr){return new (ptr) T();};
};
#endif
