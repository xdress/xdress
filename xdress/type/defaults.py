from xdress.utils import Arg, indent
from .matching import TypeMatcher, MatchAny


CYTHON_PY2C_CONV_VECTOR_REF = ((
        '# {var} is a {t.type}\n'
        'cdef int i{var}\n'
        'cdef int {var}_size\n'
        'cdef {t.cython_npctypes_nopred[0]} * {var}_data\n'
        '{var}_size = len({var})\n'
        'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {t.cython_nptype}:\n'
        '    {var}_data = <{t.cython_npctypes_nopred[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
        '    {proxy_name} = {t.cython_ctype_nopred}(<size_t> {var}_size)\n'
        '    for i{var} in range({var}_size):\n'
        '        {proxy_name}[i{var}] = {var}_data[i{var}]\n'
        'else:\n'
        '    {proxy_name} = {t.cython_ctype_nopred}(<size_t> {var}_size)\n'
        '    for i{var} in range({var}_size):\n'
        '        {proxy_name}[i{var}] = <{t.cython_npctypes_nopred[0]}> {var}[i{var}]\n'),
        '{proxy_name}')     # FIXME There might be improvements here...


def get_defaults():
    """Returns a dictionary containing the default values for a TypeSystem
    instance.
    """

    return {
        'base_types': _get_base_types(),
        'template_types': _get_template_types(),
        'refined_types': _get_refined_types(),
        'humannames': _get_humannames(),
        'argument_kinds': _get_argument_kinds(),
        'variable_namespace': {},
        'type_aliases': _get_type_aliases(),
        'cpp_types': _get_cpp_types(),
        'numpy_types': _get_numpy_types(),
        'from_pytypes': _get_from_pytypes(),
        'cython_ctypes': _get_cython_ctypes(),
        'cython_cytypes': _get_cython_cytypes(),
        'cython_pytypes': _get_cython_pytypes(),
        'cython_cimports': _get_cython_cimports(),
        'cython_cyimports': _get_cython_cyimports(),
        'cython_pyimports': _get_cython_pyimports(),
        'cython_functionnames': _get_cython_functionnames(),
        'cython_classnames': _get_cython_classnames(),
        'cython_c2py_conv': _get_cython_c2py_conv(),
        'cython_py2c_conv_vector_ref': CYTHON_PY2C_CONV_VECTOR_REF,
        'cython_py2c_conv': _get_cython_py2c_conv(),
    }


def _get_base_types():
    return set(
        ['char', 'uchar', 'str', 'int16', 'int32', 'int64', 'int128', 'uint16',
         'uint32', 'uint64', 'uint128', 'float32', 'float64', 'float128',
         'complex128', 'void', 'bool', 'type', 'file', 'exception']
    )


def _get_template_types():
    return {
        'map': ('key_type', 'value_type'),
        'dict': ('key_type', 'value_type'),
        'pair': ('key_type', 'value_type'),
        'set': ('value_type',),
        'list': ('value_type',),
        'tuple': ('value_type',),
        'vector': ('value_type',),
        'enum': ('name', 'aliases'),
        'function': ('arguments', 'returns'),
        'function_pointer': ('arguments', 'returns'),
    }


def _get_refined_types():
    return {
        'nucid': 'int32',
        'nucname': 'str',
        ('enum', ('name', 'str'),
                 ('aliases', ('dict', 'str', 'int32', 0))): 'int32',
        ('function', ('arguments', ('list', ('pair', 'str', 'type'))),
                     ('returns', 'type')): 'void',
        ('function_pointer', ('arguments', ('list', ('pair', 'str', 'type'))),
                             ('returns', 'type')): ('void', '*'),
    }


def _get_humannames():
    return {
        'char': 'character',
        'uchar': 'unsigned character',
        'str': 'string',
        'bool': 'boolean',
        'int16': 'short integer',
        'int32': 'integer',
        'int64': 'long integer',
        'int128': 'very long integer',
        'uint16': 'unsigned short integer',
        'uint32': 'unsigned integer',
        'uint64': 'unsigned long integer',
        'uint128': 'unsigned very long integer',
        'float32': 'float',
        'float64': 'double',
        'float128': 'long double',
        'complex128': 'complex',
        'file': 'file',
        'exception': 'exception',
        'dict': 'dict of ({key_type}, {value_type}) items',
        'map': 'map of ({key_type}, {value_type}) items',
        'pair': '({key_type}, {value_type}) pair',
        'set': 'set of {value_type}',
        'vector': 'vector [ndarray] of {value_type}',
        }


def _get_argument_kinds():
    return {
        ('vector', 'bool', 0): (Arg.TYPE,),
        ('vector', 'char', 0): (Arg.TYPE,),
    }


def _get_type_aliases():
    return {
        'i': 'int32',
        'i2': 'int16',
        'i4': 'int32',
        'i8': 'int64',
        'i16': 'int128',
        'int': 'int32',
        'ui': 'uint32',
        'ui2': 'uint16',
        'ui4': 'uint32',
        'ui8': 'uint64',
        'ui16': 'uint128',
        'uint': 'uint32',
        'f': 'float64',
        'f4': 'float32',
        'f8': 'float64',
        'f16': 'float128',
        'float': 'float64',
        'double': 'float64',
        'complex': 'complex128',
        'b': 'bool',
        'v': 'void',
        's': 'str',
        'string': 'str',
        'FILE': 'file',
        '_IO_FILE': 'file',
        # 'c' has char / complex ambiguity, not included
        'NPY_BYTE': 'char',
        'NPY_UBYTE': 'uchar',
        'NPY_STRING': 'str',
        'NPY_INT16': 'int16',
        'NPY_INT32': 'int32',
        'NPY_INT64': 'int64',
        'NPY_UINT16': 'uint16',
        'NPY_UINT32': 'uint32',
        'NPY_UINT64': 'uint64',
        'NPY_FLOAT32': 'float32',
        'NPY_FLOAT64': 'float64',
        'NPY_COMPLEX128': 'complex128',
        'NPY_BOOL': 'bool',
        'NPY_VOID': 'void',
        'NPY_OBJECT': 'void',
        'np.NPY_BYTE': 'char',
        'np.NPY_UBYTE': 'uchar',
        'np.NPY_STRING': 'str',
        'np.NPY_INT16': 'int16',
        'np.NPY_INT32': 'int32',
        'np.NPY_INT64': 'int64',
        'np.NPY_UINT16': 'uint16',
        'np.NPY_UINT32': 'uint32',
        'np.NPY_UINT64': 'uint64',
        'np.NPY_FLOAT32': 'float32',
        'np.NPY_FLOAT64': 'float64',
        'np.NPY_COMPLEX128': 'complex128',
        'np.NPY_BOOL': 'bool',
        'np.NPY_VOID': 'void',
        'np.NPY_OBJECT': 'void',
    }


def _get_cpp_types():
    def cpp_types_function(t, ts):
        rtnct = ts.cpp_type(t[2][2])
        argcts = [ts.cpp_type(argt) for n, argt in t[1][2]]
        if argcts == ['void']:
            argcts = []
        return rtnct + " {type_name}(" + ", ".join(argcts) + ")"

    def cpp_types_function_pointer(t, ts):
        rtnct = ts.cpp_type(t[2][2])
        argcts = [ts.cpp_type(argt) for n, argt in t[1][2]]
        if argcts == ['void']:
            argcts = []
        return rtnct + " (*{type_name})(" + ", ".join(argcts) + ")"

    return {
        'char': 'char',
        'uchar': 'unsigned char',
        'str': 'std::string',
        'int16': 'short',
        'int32': 'int',
        'int64': 'long long',
        'uint16': 'unsigned short',
        'uint32': 'unsigned long',
        'uint64': 'unsigned long long',
        'float32': 'float',
        'float64': 'double',
        'float128': 'long double',
        'complex128': '{extra_types}complex_t',
        'bool': 'bool',
        'void': 'void',
        'file': 'FILE',
        'exception': '{extra_types}exception',
        'map': 'std::map',
        'dict': 'std::map',
        'pair': 'std::pair',
        'set': 'std::set',
        'vector': 'std::vector',
        True: 'true',
        'true': 'true',
        'True': 'true',
        False: 'false',
        'false': 'false',
        'False': 'false',
        'function': cpp_types_function,
        'function_pointer': cpp_types_function_pointer,
    }


def _get_numpy_types():
    return {
        'char': 'np.NPY_BYTE',
        'uchar': 'np.NPY_UBYTE',
        #'str': 'np.NPY_STRING',
        'int16': 'np.NPY_INT16',
        'int32': 'np.NPY_INT32',
        'int64': 'np.NPY_INT64',
        'uint16': 'np.NPY_UINT16',
        'uint32': 'np.NPY_UINT32',
        'uint64': 'np.NPY_UINT64',
        'float32': 'np.NPY_FLOAT32',
        'float64': 'np.NPY_FLOAT64',
        'float128': 'np.NPY_FLOAT128',
        'complex128': 'np.NPY_COMPLEX128',
        'bool': 'np.NPY_BOOL',
        'void': 'np.NPY_VOID',
    }


def _get_from_pytypes():
    return {
        'str': ['basestring'],
        'char': ['basestring'],
        'uchar': ['basestring'],
        'int16': ['int'],
        'int32': ['int'],
        'int64': ['int'],
        'uint16': ['int'],
        'uint32': ['int'],
        'uint64': ['int'],
        'float32': ['float', 'int'],
        'float64': ['float', 'int'],
        'complex128': ['complex', 'float'],
        'file': ['file'],
        ('file', '*'): ['file'],
        'pair': ['tuple'],
        'set': ['collections.Set', 'list', 'basestring', 'tuple'],
        'map': ['collections.Mapping', 'list', 'tuple'],
        'vector': ['list', 'tuple', 'np.ndarray'],
    }


def _get_cython_ctypes():
    def cython_ctypes_function(t, ts):
        rtnct = ts.cython_ctype(t[2][2])
        argcts = [ts.cython_ctype(argt) for n, argt in t[1][2]]
        if argcts == ['void']:
            argcts = []
        return rtnct + " {type_name}(" + ", ".join(argcts) + ")"

    def cython_ctypes_function_pointer(t, ts):
        rtnct = ts.cython_ctype(t[2][2])
        argcts = [ts.cython_ctype(argt) for n, argt in t[1][2]]
        if argcts == ['void']:
            argcts = []
        return rtnct + " (*{type_name})(" + ", ".join(argcts) + ")"

    return {
        'char': 'char',
        'uchar': '{extra_types}uchar',
        'str': 'std_string',
        'int16': 'short',
        'int32': 'int',
        'int64': '{extra_types}int64',
        'uint16': '{extra_types}uint16',
        'uint32': '{extra_types}uint32',
        'uint64': '{extra_types}uint64',
        'float32': 'float',
        'float64': 'double',
        'float128': '{extra_types}float128',
        'complex128': '{extra_types}complex_t',
        'bool': 'cpp_bool',
        'void': 'void',
        'file': 'c_file',
        'exception': '{extra_types}exception',
        'map': 'cpp_map',
        'dict': 'dict',
        'pair': 'cpp_pair',
        'set': 'cpp_set',
        'vector': 'cpp_vector',
        'function': cython_ctypes_function,
        'function_pointer': cython_ctypes_function_pointer,
    }


def _get_cython_cytypes():
    return {
        'char': 'char',
        'uchar': 'unsigned char',
        'str': 'char *',
        'int16': 'short',
        'int32': 'int',
        ('int32', '*'): 'int *',
        'int64': 'long long',
        'uint16': 'unsigned short',
        'uint32': 'unsigned long',  # 'unsigned int'
        'uint64': 'unsigned long long',
        'float32': 'float',
        'float64': 'float',
        'float128': 'long double',
        'complex128': 'object',
        'bool': 'bool',
        'void': 'void',
        'file': 'c_file',
        'exception': '{extra_types}exception',
        'map': '{stlcontainers}_Map{key_type}{value_type}',
        'dict': 'dict',
        'pair': '{stlcontainers}_Pair{key_type}{value_type}',
        'set': '{stlcontainers}_Set{value_type}',
        'vector': 'np.ndarray',
        'function': 'object',
        'function_pointer': 'object',
    }


def _get_cython_pytypes():
    return {
        'char': 'str',
        'uchar': 'str',
        'str': 'str',
        'int16': 'int',
        'int32': 'int',
        'int64': 'int',
        'uint16': 'int',  # 'unsigned int'
        'uint32': 'int',  # 'unsigned int'
        'uint64': 'int',  # 'unsigned int'
        'float32': 'float',
        'float64': 'float',
        'float128': 'np.float128',
        'complex128': 'object',
        'file': 'file',
        'exception': 'Exception',
        'bool': 'bool',
        'void': 'object',
        'map': '{stlcontainers}Map{key_type}{value_type}',
        'dict': 'dict',
        'pair': '{stlcontainers}Pair{key_type}{value_type}',
        'set': '{stlcontainers}Set{value_type}',
        'vector': 'np.ndarray',
    }


def _get_cython_cimports():
    def cython_cimports_functionish(t, ts, seen):
        seen.add(('cython.operator', 'dereference', 'deref'))
        for n, argt in t[1][2]:
            ts.cython_cimport_tuples(argt, seen=seen, inc=('c',))
        ts.cython_cimport_tuples(t[2][2], seen=seen, inc=('c',))

    return {
        'char': (None,),
        'uchar':  (('{extra_types}',),),
        'str': (('libcpp.string', 'string', 'std_string'),),
        'int16': (None,),
        'int32': (None,),
        ('int32', '*'): 'int *',
        'int64':  (('{extra_types}',),),
        'uint16':  (('{extra_types}',),),
        'uint32': (('{extra_types}',),),
        'uint64':  (('{extra_types}',),),
        'float32': (None,),
        'float64': (None,),
        'float128':  (('{extra_types}',),),
        'complex128': (('{extra_types}',),),
        'bool': (('libcpp', 'bool', 'cpp_bool'),),
        'void': (None,),
        'file': (('libc.stdio', 'FILE', 'c_file'),),
        'exception': (('{extra_types}',),),
        'map': (('libcpp.map', 'map', 'cpp_map'),),
        'dict': (None,),
        'pair': (('libcpp.utility', 'pair', 'cpp_pair'),),
        'set': (('libcpp.set', 'set', 'cpp_set'),),
        'vector': (('libcpp.vector', 'vector', 'cpp_vector'),),
        'nucid': (('pyne', 'cpp_nucname'),),
        'nucname': (('pyne', 'cpp_nucname'),
                    ('libcpp.string', 'string', 'std_string')),
        'function': cython_cimports_functionish,
        'function_pointer': cython_cimports_functionish,
    }


def _get_cython_cyimports():
    def cython_cyimports_functionish(t, ts, seen):
        for n, argt in t[1][2]:
            ts.cython_cimport_tuples(argt, seen=seen, inc=('cy',))
        ts.cython_cimport_tuples(t[2][2], seen=seen, inc=('cy',))

    return {
        'char': (None,),
        'uchar': (None,),
        'str': (None,),
        'int16': (None,),
        'int32': (None,),
        'int64': (None,),
        'uint16': (None,),
        'uint32': (None,),
        'uint64': (None,),
        'float32': (None,),
        'float64': (None,),
        'float128': (None,),
        'complex128': (('{extra_types}',),),  # for py2c_complex()
        'bool': (None,),
        'void': (None,),
        'file': (('{extra_types}',),),
        'exception': (('{extra_types}',),),
        'map': (('{stlcontainers}',),),
        'dict': (None,),
        'pair': (('{stlcontainers}',),),
        'set': (('{stlcontainers}',),),
        'vector': (('numpy', 'as', 'np'), ('{dtypes}',)),
        'nucid': (('pyne', 'nucname'),),
        'nucname': (('pyne', 'nucname'),),
        'function': cython_cyimports_functionish,
        'function_pointer': cython_cyimports_functionish,
    }


def _get_cython_pyimports():
    def cython_pyimports_functionish(t, ts, seen):
        seen.add(('warnings',))
        for n, argt in t[1][2]:
            ts.cython_import_tuples(argt, seen=seen)
        ts.cython_import_tuples(t[2][2], seen=seen)

    return {
        'char': (None,),
        'uchar': (None,),
        'str': (None,),
        'int16': (None,),
        'int32': (None,),
        'int64': (None,),
        'uint16': (None,),
        'uint32': (None,),
        'uint64': (None,),
        'float32': (None,),
        'float64': (None,),
        'float128': (None,),
        'complex128': (None,),
        'bool': (None,),
        'void': (None,),
        'file': (None,),
        'exception': (None,),
        'map': (('{stlcontainers}',), ('collections',)),
        'dict': (None,),
        'pair': (('{stlcontainers}',),),
        'set': (('{stlcontainers}',), ('collections',)),
        'vector': (('numpy', 'as', 'np'),),
        'nucid': (('pyne', 'nucname'),),
        'nucname': (('pyne', 'nucname'),),
        'function': cython_pyimports_functionish,
        'function_pointer': cython_pyimports_functionish,
    }


def _get_cython_functionnames():
    return {
        # base types
        'char': 'char',
        'uchar': 'uchar',
        'str': 'str',
        'int16': 'short',
        'int32': 'int',
        'int64': 'long',
        'uint16': 'ushort',
        'uint32': 'uint',
        'uint64': 'ulong',
        'float32': 'float',
        'float64': 'double',
        'float128': 'longdouble',
        'complex128': 'complex',
        'bool': 'bool',
        'void': 'void',
        'file': 'file',
        'exception': 'exception',
        # template types
        'map': 'map_{key_type}_{value_type}',
        'dict': 'dict',
        'pair': 'pair_{key_type}_{value_type}',
        'set': 'set_{value_type}',
        'vector': 'vector_{value_type}',
        'nucid': 'nucid',
        'nucname': 'nucname',
        'function': 'function',
        'function_pointer': 'functionpointer',
    }


def _get_cython_classnames():
    return {
        # base types
        'char': 'Char',
        'uchar': 'UChar',
        'str': 'Str',
        'int32': 'Short',
        'int32': 'Int',
        'int64': 'Long',
        'uint16': 'UShort',
        'uint32': 'UInt',
        'uint64': 'ULong',
        'float32': 'Float',
        'float64': 'Double',
        'float128': 'Longdouble',
        'complex128': 'Complex',
        'bool': 'Bool',
        'void': 'Void',
        'file': 'File',
        'exception': 'Exception',
        # template types
        'map': 'Map{key_type}{value_type}',
        'dict': 'Dict',
        'pair': 'Pair{key_type}{value_type}',
        'set': 'Set{value_type}',
        'vector': 'Vector{value_type}',
        'nucid': 'Nucid',
        'nucname': 'Nucname',
    }


def _get_cython_c2py_conv():
    def cython_c2py_conv_function_pointer(t_, ts):
        """Wrap function pointers in C/C++ to Python functions."""
        t = t_[1]
        argnames = []
        argdecls = []
        argbodys = []
        argrtns = []
        for n, argt in t[1][2]:
            argnames.append(n)
            decl, body, rtn = ts.cython_py2c(n, argt, proxy_name="c_" + n)
            argdecls += decl.split('\n') if isinstance(decl,basestring) else [decl]
            argbodys += body.split('\n') if isinstance(body,basestring) else [body]
            argrtns += rtn.split('\n') if isinstance(rtn,basestring) else [rtn]
        rtnname = 'rtn'
        rtnprox = 'c_' + rtnname
        rtncall = 'c_call_' + rtnname
        while rtnname in argnames or rtnprox in argnames:
            rtnname += '_'
            rtnprox += '_'
        argdecls = indent(argdecls)
        argbodys = indent(argbodys)
        rtndecl, rtnbody, rtnrtn, _ = ts.cython_c2py(rtncall, t[2][2],
            cached=False, proxy_name=rtnprox, existing_name=rtncall)
        if rtndecl is None and rtnbody is None:
            rtnprox = rtnname
        rtndecls = [rtndecl]
        returns_void = (t[2][2] == 'void')
        if not returns_void:
            rtndecls.append("cdef {0} {1}".format(ts.cython_ctype(t[2][2]),
                                                   rtncall))
        rtndecl = indent(rtndecls)
        rtnbody = indent(rtnbody)
        s = ('def {{proxy_name}}({arglist}):\n'
             '{argdecls}\n'
             '{rtndecl}\n'
             '    if {{var}} == NULL:\n'
             '        raise RuntimeError("{{var}} is NULL and may not be '
                                         'safely called!")\n'
             '{argbodys}\n')
        s += '    {{var}}({carglist})\n' if returns_void else \
             '    {rtncall} = {{var}}({carglist})\n'
        s += '{rtnbody}\n'
        s = s.format(arglist=", ".join(argnames), argdecls=argdecls,
                     cvartypeptr=ts.cython_ctype(t_).format(type_name='cvartype'),
                     argbodys=argbodys, rtndecl=rtndecl, rtnprox=rtnprox,
                     rtncall=rtncall, carglist=", ".join(argrtns), rtnbody=rtnbody)
        caches = 'if {cache_name} is None:\n' + indent(s)
        if not returns_void:
            caches += "\n        return {rtnrtn}".format(rtnrtn=rtnrtn)
            caches += '\n    {cache_name} = {proxy_name}\n'
        return s, s, caches

    return {
        # Has tuple form of (copy, [view, [cached_view]])
        # base types
        'char': ('chr(<int> {var})',),
        ('char', '*'): ('bytes({var}).decode()',),
        'uchar': ('chr(<unsigned int> {var})',),
        ('uchar', '*'): ('bytes(<char *> {var}).decode()',),
        'str': ('bytes(<char *> {var}.c_str()).decode()',),
        ('str', '*'): ('bytes(<char *> {var}[0].c_str()).decode()',),
        'int16': ('int({var})',),
        ('int16', '*'): ('int({var}[0])',),
        'int32': ('int({var})',),
        ('int32', '*'): ('int({var}[0])',),
        'int64': ('int({var})',),
        ('int64', '*'): ('int({var}[0])',),
        'uint16': ('int({var})',),
        ('uint16', '*'): ('int({var}[0])',),
        'uint32': ('int({var})',),
        ('uint32', '*'): ('int({var}[0])',),
        'uint64': ('int({var})',),
        'float32': ('float({var})',),
        ('float32', '*'): ('float({var}[0])',),
        'float64': ('float({var})',),
        ('float64', '*'): ('float({var}[0])',),
        'float128': ('np.array({var}, dtype=np.float128)',),
        ('float128', '*'): ('np.array({var}[0], dtype=np.float128)',),
        'complex128': ('complex(float({var}.re), float({var}.im))',),
        ('complex128', '*'): ('complex(float({var}[0].re), float({var}[0].im))',),
        'bool': ('bool({var})',),
        ('bool', '*'): ('bool({var}[0])',),
        'void': ('None',),
        'file': ('{extra_types}PyFile_FromFile(&{var}, "{var}", "r+", NULL)',),
        ('file', '*'): (
            '{extra_types}PyFile_FromFile({var}, "{var}", "r+", NULL)',),
        # template types
        'map': ('{t.cython_pytype}({var})',
               ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                '{proxy_name}.map_ptr = &{var}\n'),
               ('if {cache_name} is None:\n'
                '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                '    {proxy_name}.map_ptr = &{var}\n'
                '    {cache_name} = {proxy_name}\n'
                )),
        'dict': ('dict({var})',),
        'pair': ('{t.cython_pytype}({var})',
                 ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                  '{proxy_name}.pair_ptr = <{t.cython_ctype}*> &{var}\n'),
                 ('if {cache_name} is None:\n'
                  '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                  '    {proxy_name}.pair_ptr = &{var}\n'
                  '    {cache_name} = {proxy_name}\n'
                  )),
        'set': ('{t.cython_pytype}({var})',
               ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                '{proxy_name}.set_ptr = &{var}\n'),
               ('if {cache_name} is None:\n'
                '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                '    {proxy_name}.set_ptr = &{var}\n'
                '    {cache_name} = {proxy_name}\n'
                )),
        TypeMatcher(('set', MatchAny, '*')): ('{t.cython_pytype}(deref({var}))',
               ('{proxy_name} = {t.cython_pytype}(False, False)\n'
                '{proxy_name}.set_ptr = {var}\n'),
               ('if {cache_name} is None:\n'
                '    {proxy_name} = {t.cython_pytype}(False, False)\n'
                '    {proxy_name}.set_ptr = {var}\n'
                '    {cache_name} = {proxy_name}\n'
                )),
        'vector': (
            ('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '{proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {t.cython_nptypes[0]}, &{var}[0])\n'
             '{proxy_name} = np.PyArray_Copy({proxy_name})\n'),
            ('{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '{proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {t.cython_nptypes[0]}, &{var}[0])\n'),
            ('if {cache_name} is None:\n'
             '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '    {proxy_name} = np.PyArray_SimpleNewFromData(1, {proxy_name}_shape, {t.cython_nptypes[0]}, &{var}[0])\n'
             '    {cache_name} = {proxy_name}\n'
            )),
        ('vector', 'bool', 0): (  # C++ standard is silly here
            ('cdef int i\n'
             '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '{proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {t.cython_nptypes[0]})\n'
             'for i in range({proxy_name}_shape[0]):\n'
             '    {proxy_name}[i] = {var}[i]\n'),
            ('cdef int i\n'
             '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '{proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {t.cython_nptypes[0]})\n'
             'for i in range({proxy_name}_shape[0]):\n'
             '    {proxy_name}[i] = {var}[i]\n'),
            ('cdef int i\n'
             'if {cache_name} is None:\n'
             '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '    {proxy_name} = np.PyArray_SimpleNew(1, {proxy_name}_shape, {nptype[0]})\n'
             '    for i in range({proxy_name}_shape[0]):\n'
             '        {proxy_name}[i] = {var}[i]\n'
             '    {cache_name} = {proxy_name}\n'
             )),
        # C/C++ chars are ints while Python Chars are length-1 strings
        ('vector', 'char', 0): (
            ('cdef int i\n'
             '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '{proxy_name} = np.empty({proxy_name}_shape[0], "U1")\n'
             'for i in range({proxy_name}_shape[0]):\n'
             '    {proxy_name}[i] = chr(<int> {var}[i])\n'),
            ('cdef int i\n'
             '{proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '{proxy_name} = np.empty({proxy_name}_shape[0], "U1")\n'
             'for i in range({proxy_name}_shape[0]):\n'
             '    {proxy_name}[i] = chr(<int> {var}[i])\n'),
            ('cdef int i\n'
             'if {cache_name} is None:\n'
             '    {proxy_name}_shape[0] = <np.npy_intp> {var}.size()\n'
             '    for i in range({proxy_name}_shape[0]):\n'
             '        {proxy_name}[i] = chr(<int> {var}[i])\n'
             '    {cache_name} = {proxy_name}\n'
             )),
        'nucid': ('nucname.zzaaam({var})',),
        'nucname': ('nucname.name({var})',),
        TypeMatcher((('enum', MatchAny, MatchAny), '*')): ('int({var}[0])',),
        TypeMatcher((('int32', ('enum', MatchAny, MatchAny)), '*')): \
                                                            ('int({var}[0])',),
        # Strip const when going c -> py
        TypeMatcher((MatchAny, 'const')): (
            lambda t, ts: ts.cython_c2py_getitem(t[0])),
        TypeMatcher(((MatchAny, 'const'), '&')) : (
            lambda t, ts: ts.cython_c2py_getitem((t[0][0], '&'))),
        TypeMatcher(((MatchAny, 'const'), '*')): (
            lambda t, ts: ts.cython_c2py_getitem((t[0][0], '*'))),
        'function_pointer': cython_c2py_conv_function_pointer,
    }


def _get_cython_py2c_conv():
    def cython_py2c_conv_function_pointer(t, ts):
        t = t[1]
        argnames = []
        argcts = []
        argdecls = []
        argbodys = []
        argrtns = []
        for n, argt in t[1][2]:
            argnames.append(n)
            decl, body, rtn, _ = ts.cython_c2py(n, argt, proxy_name="c_" + n,
                                                cached=False)
            argdecls.append(decl)
            #argdecls.append("cdef {0} {1}".format(cython_pytype(argt), "c_" + n))
            argbodys.append(body)
            argrtns.append(rtn)
            argct = ts.cython_ctype(argt)
            argcts.append(argct)
        rtnname = 'rtn'
        rtnprox = 'c_' + rtnname
        rtncall = 'call_' + rtnname
        while rtnname in argnames or rtnprox in argnames:
            rtnname += '_'
            rtnprox += '_'
        rtnct = ts.cython_ctype(t[2][2])
        argdecls = indent(argdecls)
        argbodys = indent(argbodys)
        #rtndecl, rtnbody, rtnrtn = cython_py2c(rtnname, t[2][2], proxy_name=rtnprox)
        #rtndecl, rtnbody, rtnrtn = cython_py2c(rtnname, t[2][2], proxy_name=rtncall)
        rtndecl, rtnbody, rtnrtn = ts.cython_py2c(rtncall, t[2][2],
                                                  proxy_name=rtnprox)
        if rtndecl is None and rtnbody is None:
            rtnprox = rtnname
        rtndecl = indent([rtndecl])
        rtnbody = indent([rtnbody])
        returns_void = (t[2][2] == 'void')
        if returns_void:
            rtnrtn = ''
        s = ('cdef {rtnct} {{proxy_name}}({arglist}):\n'
             '{argdecls}\n'
             '{rtndecl}\n'
             '    global {{var}}\n'
             '{argbodys}\n')
        s += '    {{var}}({pyarglist})\n' if returns_void else \
             '    {rtncall} = {{var}}({pyarglist})\n'
        s += ('{rtnbody}\n'
              '    return {rtnrtn}\n')
        arglist = ", ".join(["{0} {1}".format(*x) for x in zip(argcts, argnames)])
        pyarglist=", ".join(argrtns)
        s = s.format(rtnct=rtnct, arglist=arglist, argdecls=argdecls,
                     rtndecl=rtndecl, argbodys=argbodys, rtnprox=rtnprox,
                     pyarglist=pyarglist, rtnbody=rtnbody, rtnrtn=rtnrtn,
                     rtncall=rtncall)
        return s, False

    return {
        # Has tuple form of (body or return,  return or False)
        # base types
        'char': ('{var}_bytes = {var}.encode()', '(<char *> {var}_bytes)[0]'),
        ('char', '*'): ('{var}_bytes = {var}.encode()', '<char *> {var}_bytes'),
        (('char', '*'), '*'): ('cdef char * {var}_bytes_\n'
                               '{var}_bytes = {var}[0].encode()\n'
                               '{var}_bytes_ = {var}_bytes\n'
                               '{proxy_name} = &{var}_bytes_',
                               '{proxy_name}'),
        'uchar': ('{var}_bytes = {var}.encode()', '(<unsigned char *> {var}_bytes)[0]'),
        ('uchar', '*'): ('{var}_bytes = {var}.encode()', '<unsigned char *> {var}_bytes'),
        'str': ('{var}_bytes = {var}.encode()', 'std_string(<char *> {var}_bytes)'),
        'int16': ('<short> {var}', False),
        'int32': ('<int> {var}', False),
        #('int32', '*'): ('&(<int> {var})', False),
        ('int32', '*'): ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
        'int64': ('<long long> {var}', False),
        'uint16': ('<unsigned short> {var}', False),
        'uint32': ('<{t.cython_ctype}> long({var})', False),
        #'uint32': ('<unsigned long> {var}', False),
        #('uint32', '*'): ('cdef unsigned long {proxy_name}_ = {var}', '&{proxy_name}_'),
        ('uint32', '*'): ('cdef unsigned int {proxy_name}_ = {var}', '&{proxy_name}_'),
        'uint64': ('<unsigned long long> {var}', False),
        'float32': ('<float> {var}', False),
        'float64': ('<double> {var}', False),
        ('float64', '*'): ('cdef double {proxy_name}_ = {var}', '&{proxy_name}_'),
        'float128': ('<long double> {var}', False),
        'complex128': ('{extra_types}py2c_complex({var})', False),
        'bool': ('<bint> {var}', False),
        'void': ('NULL', False),
        ('void', '*'): ('NULL', False),
        'file': ('{extra_types}PyFile_AsFile({var})[0]', False),
        ('file', '*'): ('{extra_types}PyFile_AsFile({var})', False),
        # template types
        'map': ('{proxy_name} = {t.cython_pytype}({var}, not isinstance({var}, {t.cython_cytype}))',
                '{proxy_name}.map_ptr[0]'),
        'dict': ('dict({var})', False),
        'pair': ('{proxy_name} = {t.cython_pytype}({var}, not isinstance({var}, {t.cython_cytype}))',
                 '{proxy_name}.pair_ptr[0]'),
        'set': ('{proxy_name} = {t.cython_pytype}({var}, not isinstance({var}, {t.cython_cytype}))',
                '{proxy_name}.set_ptr[0]'),
        'vector': ((
            '# {var} is a {t.type}\n'
            'cdef int i{var}\n'
            'cdef int {var}_size\n'
            'cdef {t.cython_npctypes[0]} * {var}_data\n'
            '{var}_size = len({var})\n'
            'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == {t.cython_nptype}:\n'
            '    {var}_data = <{t.cython_npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
            '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = {var}_data[i{var}]\n'
            'else:\n'
            '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = <{t.cython_npctypes[0]}> {var}[i{var}]\n'),
            '{proxy_name}'),     # FIXME There might be improvements here...
        ('vector', 'char', 0): ((
            '# {var} is a {t.type}\n'
            'cdef int i{var}\n'
            'cdef int {var}_size\n'
            'cdef {t.cython_npctypes[0]} * {var}_data\n'
            '{var}_size = len({var})\n'
            'if isinstance({var}, np.ndarray) and (<np.ndarray> {var}).descr.type_num == <int> {t.cython_nptype}:\n'
            '    {var}_data = <{t.cython_npctypes[0]} *> np.PyArray_DATA(<np.ndarray> {var})\n'
            '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        {proxy_name}[i{var}] = {var}[i{var}]\n'
            'else:\n'
            '    {proxy_name} = {t.cython_ctype}(<size_t> {var}_size)\n'
            '    for i{var} in range({var}_size):\n'
            '        _ = {var}[i{var}].encode()\n'
            '        {proxy_name}[i{var}] = deref(<char *> _)\n'),
            '{proxy_name}'),
        TypeMatcher(('vector', MatchAny, '&')): CYTHON_PY2C_CONV_VECTOR_REF,
        TypeMatcher((('vector', MatchAny, 0), '&')): CYTHON_PY2C_CONV_VECTOR_REF,
        TypeMatcher((('vector', MatchAny, '&'), 0)): CYTHON_PY2C_CONV_VECTOR_REF,
        TypeMatcher((('vector', MatchAny, '&'), 'const')): CYTHON_PY2C_CONV_VECTOR_REF,
        TypeMatcher((('vector', MatchAny, 'const'), '&')): CYTHON_PY2C_CONV_VECTOR_REF,
        TypeMatcher(((('vector', MatchAny, 0), 'const'), '&')): CYTHON_PY2C_CONV_VECTOR_REF,
        TypeMatcher(((('vector', MatchAny, 0), '&'), 'const')): CYTHON_PY2C_CONV_VECTOR_REF,
        # refinement types
        'nucid': ('nucname.zzaaam({var})', False),
        'nucname': ('nucname.name({var})', False),
        TypeMatcher((('enum', MatchAny, MatchAny), '*')): \
            ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
        TypeMatcher((('int32', ('enum', MatchAny, MatchAny)), '*')): \
            ('cdef int {proxy_name}_ = {var}', '&{proxy_name}_'),
        'function_pointer': cython_py2c_conv_function_pointer,
    }
