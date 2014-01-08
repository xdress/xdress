// Pretend that we're a Python module to allow easy import from xdress
#ifdef XDRESS

#include <Python.h>

void XDRESS_FATAL(const char* message) {
  fprintf(stderr, "XDRESS LIBCLANG FATAL ERROR: %s\n", message);
  ::abort();
}

PyMODINIT_FUNC initlibclang(void) {
  Py_InitModule("libclang", 0);
}
#endif
