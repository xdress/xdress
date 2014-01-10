// Pretend that we're a Python module to allow easy import from xdress
#ifdef XDRESS

#include <Python.h>

void XDRESS_FATAL(const char* message) {
  fprintf(stderr, "XDRESS LIBCLANG FATAL ERROR: %s\n", message);
  ::abort();
}

#if PY_MAJOR_VERSION >= 3

PyMODINIT_FUNC PyInit_libclang(void) {
  static struct PyModuleDef def = {
    PyModuleDef_HEAD_INIT,
    "libclang"
  };
  return PyModule_Create(&def);
}

#else // Python 2.x

PyMODINIT_FUNC initlibclang(void) {
  Py_InitModule("libclang", 0);
}

#endif
#endif
