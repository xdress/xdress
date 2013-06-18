"""
Inserts doxygen documentatoin into python docstrings. This is done using
the xml export capabilities of doxygen. The docstrings are inserted into
the desc dictionary for each function/class and will then be merged with
standard auto-docstrings as well as any user input from sidecar files.

This module is available as an xdress pluging by the name
``xdress.doxygen``.

:author: Spencer Lyon <spencerlyon2@gmail.com>

Usage Details
=============
"""
from __future__ import print_function
import os
import sys
from subprocess import call

# this is the meat of the template doxyfile template returned by: doxygen -g
# NOTE: I have changed a few things like no html/latex generation.

# NOTE: Also, there are three placeholders for format: project, output,
#       src_dir
_doxyfile_content =\
"""
DOXYFILE_ENCODING      = UTF-8
PROJECT_NAME           = "{project}"
PROJECT_NUMBER         = "0.1"
OUTPUT_DIRECTORY       = {output}
CREATE_SUBDIRS         = NO
OUTPUT_LANGUAGE        = English
BRIEF_MEMBER_DESC      = YES
REPEAT_BRIEF           = YES
ALWAYS_DETAILED_SEC    = NO
INLINE_INHERITED_MEMB  = NO
FULL_PATH_NAMES        = YES
SHORT_NAMES            = NO
JAVADOC_AUTOBRIEF      = NO
QT_AUTOBRIEF           = NO
MULTILINE_CPP_IS_BRIEF = NO
INHERIT_DOCS           = YES
SEPARATE_MEMBER_PAGES  = NO
TAB_SIZE               = 4
OPTIMIZE_OUTPUT_FOR_C  = NO
OPTIMIZE_OUTPUT_JAVA   = NO
OPTIMIZE_FOR_FORTRAN   = NO
OPTIMIZE_OUTPUT_VHDL   = NO
MARKDOWN_SUPPORT       = YES
AUTOLINK_SUPPORT       = YES
BUILTIN_STL_SUPPORT    = NO
CPP_CLI_SUPPORT        = NO
SIP_SUPPORT            = NO
IDL_PROPERTY_SUPPORT   = YES
DISTRIBUTE_GROUP_DOC   = NO
SUBGROUPING            = YES
INLINE_GROUPED_CLASSES = NO
INLINE_SIMPLE_STRUCTS  = NO
TYPEDEF_HIDES_STRUCT   = NO
LOOKUP_CACHE_SIZE      = 0
EXTRACT_ALL            = NO
EXTRACT_PRIVATE        = NO
EXTRACT_PACKAGE        = NO
EXTRACT_STATIC         = NO
EXTRACT_LOCAL_CLASSES  = YES
EXTRACT_LOCAL_METHODS  = NO
EXTRACT_ANON_NSPACES   = NO
HIDE_UNDOC_MEMBERS     = NO
HIDE_UNDOC_CLASSES     = NO
HIDE_FRIEND_COMPOUNDS  = NO
HIDE_IN_BODY_DOCS      = NO
INTERNAL_DOCS          = NO
CASE_SENSE_NAMES       = NO
HIDE_SCOPE_NAMES       = NO
SHOW_INCLUDE_FILES     = YES
FORCE_LOCAL_INCLUDES   = NO
INLINE_INFO            = YES
SORT_MEMBER_DOCS       = YES
SORT_BRIEF_DOCS        = NO
SORT_MEMBERS_CTORS_1ST = NO
SORT_GROUP_NAMES       = NO
SORT_BY_SCOPE_NAME     = NO
STRICT_PROTO_MATCHING  = NO
GENERATE_TODOLIST      = YES
GENERATE_TESTLIST      = YES
GENERATE_BUGLIST       = YES
GENERATE_DEPRECATEDLIST= YES
MAX_INITIALIZER_LINES  = 30
SHOW_USED_FILES        = YES
SHOW_FILES             = YES
SHOW_NAMESPACES        = YES
QUIET                  = NO
WARNINGS               = YES
WARN_IF_UNDOCUMENTED   = YES
WARN_IF_DOC_ERROR      = YES
WARN_NO_PARAMDOC       = NO
WARN_FORMAT            = "$file:$line: $text"
INPUT                  = {src_dir}
INPUT_ENCODING         = UTF-8
RECURSIVE              = NO
EXCLUDE_SYMLINKS       = NO
EXAMPLE_RECURSIVE      = NO
FILTER_SOURCE_FILES    = NO
SOURCE_BROWSER         = NO
INLINE_SOURCES         = NO
STRIP_CODE_COMMENTS    = YES
REFERENCED_BY_RELATION = NO
REFERENCES_RELATION    = NO
REFERENCES_LINK_SOURCE = YES
USE_HTAGS              = NO
VERBATIM_HEADERS       = YES
ALPHABETICAL_INDEX     = YES
COLS_IN_ALPHA_INDEX    = 5
GENERATE_HTML          = NO
HTML_OUTPUT            = html
HTML_FILE_EXTENSION    = .html
HTML_COLORSTYLE_HUE    = 220
HTML_COLORSTYLE_SAT    = 100
HTML_COLORSTYLE_GAMMA  = 80
HTML_TIMESTAMP         = YES
HTML_DYNAMIC_SECTIONS  = NO
HTML_INDEX_NUM_ENTRIES = 100
GENERATE_DOCSET        = NO
DOCSET_FEEDNAME        = "Doxygen generated docs"
DOCSET_BUNDLE_ID       = org.doxygen.Project
DOCSET_PUBLISHER_ID    = org.doxygen.Publisher
DOCSET_PUBLISHER_NAME  = Publisher
GENERATE_HTMLHELP      = NO
GENERATE_CHI           = NO
BINARY_TOC             = NO
TOC_EXPAND             = NO
GENERATE_QHP           = NO
QHP_NAMESPACE          = org.doxygen.Project
QHP_VIRTUAL_FOLDER     = doc
GENERATE_ECLIPSEHELP   = NO
ECLIPSE_DOC_ID         = org.doxygen.Project
DISABLE_INDEX          = NO
GENERATE_TREEVIEW      = NO
ENUM_VALUES_PER_LINE   = 4
TREEVIEW_WIDTH         = 250
EXT_LINKS_IN_WINDOW    = NO
FORMULA_FONTSIZE       = 10
FORMULA_TRANSPARENT    = YES
USE_MATHJAX            = NO
MATHJAX_FORMAT         = HTML-CSS
MATHJAX_RELPATH        = http://cdn.mathjax.org/mathjax/latest
SEARCHENGINE           = YES
SERVER_BASED_SEARCH    = NO
EXTERNAL_SEARCH        = NO
SEARCHDATA_FILE        = searchdata.xml
GENERATE_LATEX         = NO
LATEX_OUTPUT           = latex
LATEX_CMD_NAME         = latex
MAKEINDEX_CMD_NAME     = makeindex
COMPACT_LATEX          = NO
PAPER_TYPE             = a4
PDF_HYPERLINKS         = YES
USE_PDFLATEX           = YES
LATEX_BATCHMODE        = NO
LATEX_HIDE_INDICES     = NO
LATEX_SOURCE_CODE      = NO
LATEX_BIB_STYLE        = plain
GENERATE_RTF           = NO
RTF_OUTPUT             = rtf
COMPACT_RTF            = NO
RTF_HYPERLINKS         = NO
GENERATE_MAN           = NO
MAN_OUTPUT             = man
MAN_EXTENSION          = .3
MAN_LINKS              = NO
GENERATE_XML           = YES
XML_OUTPUT             = xml
XML_PROGRAMLISTING     = YES
GENERATE_DOCBOOK       = NO
DOCBOOK_OUTPUT         = docbook
GENERATE_AUTOGEN_DEF   = NO
GENERATE_PERLMOD       = NO
PERLMOD_LATEX          = NO
PERLMOD_PRETTY         = YES
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = NO
EXPAND_ONLY_PREDEF     = NO
SEARCH_INCLUDES        = YES
SKIP_FUNCTION_MACROS   = YES
ALLEXTERNALS           = NO
EXTERNAL_GROUPS        = YES
EXTERNAL_PAGES         = YES
PERL_PATH              = /usr/bin/perl
CLASS_DIAGRAMS         = YES
HIDE_UNDOC_RELATIONS   = YES
HAVE_DOT               = NO
DOT_NUM_THREADS        = 0
DOT_FONTNAME           = Helvetica
DOT_FONTSIZE           = 10
CLASS_GRAPH            = YES
COLLABORATION_GRAPH    = YES
GROUP_GRAPHS           = YES
UML_LOOK               = NO
UML_LIMIT_NUM_FIELDS   = 10
TEMPLATE_RELATIONS     = NO
INCLUDE_GRAPH          = YES
INCLUDED_BY_GRAPH      = YES
CALL_GRAPH             = NO
CALLER_GRAPH           = NO
GRAPHICAL_HIERARCHY    = YES
DIRECTORY_GRAPH        = YES
DOT_IMAGE_FORMAT       = png
INTERACTIVE_SVG        = NO
DOT_GRAPH_MAX_NODES    = 50
MAX_DOT_GRAPH_DEPTH    = 0
DOT_TRANSPARENT        = NO
DOT_MULTI_TARGETS      = NO
GENERATE_LEGEND        = NO
DOT_CLEANUP            = YES
"""

# TODO: Allow user to pass additional doxygen arguments for more control

doxyfile = open('doxyfile', 'w')
doxyfile.write(_doxyfile_content.format(project='test',
                                        output='build',
                                        src_dir='src/'))
# FIXME: Get src_dir from rc.package
# FIXME: Maybe get output from rc.build?
doxyfile.close()

# call(['doxygen', 'doxyfile'])
"""
root = etree.parse('index.xml')
compounds = list(root.iterfind('.//compound'))

vars = []
funcs = []
classes = {}
class_names = []

class_list = filter(lambda i:i.attrib['kind'] == 'class', root.findall('compound'))

for kls in class_list:
    kls_name = kls.find('name').text
    class_names.append(kls_name)
    file_name = kls.attrib['refid']
    kls_dict = {'vars': [], 'methods': [], 'file_name': file_name}

    for mem in kls.iter('member'):
        mem_name = mem.find('name').text
        if mem.attrib['kind'] == 'variable':
            kls_dict['vars'].append(mem_name)
        elif mem.attrib['kind'] == 'function':
            kls_dict['methods'].append(mem_name)

    classes[kls_name] = kls_dict

# Now on to a single class
c1 = classes[classes.keys()[0]]
"""

def parse_class(class_dict):
    """
    Parses a single class and returns a dictionary of dictionaries
    containing all the data for that class.

    Parameters
    ----------
    class_dict : dict
        A dictionary containing the following keys:
        ['file_name', 'methods', 'vars']

    Returns
    -------
    data : dict
        A dictionary with all docstrings for instance variables and
        class methods. This object is structured as follows:

            data
                'protected-func'
                    'prot_func1'
                        arg_string
                        briefdescription
                        detaileddescription
                        type
                        definition

                'public-func'
                    'pub_func_1'
                        arg_string
                        briefdescription
                        detaileddescription
                        type
                        definition

                'protected-attrib'
                    'prot-attrib1'
                        briefdescription
                        detaileddescription
                        type
                        definition

        This means that data is a 3-level dictionary. The levels go as
        follows:

        1. data
            - keys: 'protected-func', 'public-func', and
            'protected-attrib'
            - values: dictionaries of attribute types
        2. dictionaries of attribute types
            - keys: attribute names
            - values: attribute dictionaries
        3. attribute dictionaries
            - keys: arg_string, briefdescription, detaileddescription,
            type, definition
            - values: strings containing the actual data we care about

    Notes
    -----

    The inner 'arg_string' key is only applicable to methods as it
    contains the function signature for the arguments.

    For methods, the type key has a value of the return type of the
    function.

    """
    c1 = class_dict
    fn1 = c1['file_name']
    croot = etree.parse(fn1 + '.xml')
    compd_def = croot.find('compounddef')
    data = {}
    for sec in compd_def.iter('sectiondef'):
        # Iterate over all sections in the compound
        sec_name = sec.attrib['kind']
        sec_dict = {}

        for mem in sec.iter('memberdef'):
            # Iterate over each member in the section
            mem_name = mem.find('name').text
            mem_dict = {}

            # Add type and definition signature.
            # NOTE: for class methods type is the type of the return value.
            mem_dict['type'] = mem.find('type').text
            mem_dict['definition'] = mem.find('definition').text

            # Find brief description:
            mem_bd = mem.find('briefdescription')
            try:
                mem_bdstr = mem_bd.find('para').text
            except AttributeError:
                mem_bdstr = ''

            mem_dict['briefdesctription'] = mem_bdstr

            # Find detailed description:
            mem_dd = mem.find('detaileddescription')
            try:
                mem_ddstr = mem_dd.find('para').text
            except AttributeError:
                mem_ddstr = ''

            mem_dict['detaileddescription'] = mem_ddstr

            # Try to find argstring (only applicable to methods)
            mem_argstr = mem.find('argsstring').text
            if mem_argstr is not None:
                mem_dict['arg_string'] = mem_argstr


            i = 1
            while mem_name in sec_dict.keys():
                if i > 1:
                    mem_name = mem_name[:-1]
                mem_name += str(i)
                i += 1
            sec_dict[mem_name] = mem_dict

        data[sec_name] = sec_dict

    return data
