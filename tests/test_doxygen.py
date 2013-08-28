from __future__ import print_function
import pprint
from collections import OrderedDict
from nose.tools import assert_equal
from nose.plugins.attrib import attr
from xdress.doxygen import class_docstr, func_docstr

unit = attr('unit')

car_dict = {'file_name': 'Cars.h',
 'kls_name': 'util::Car',
 'members': {'methods': ['Car',
                         'Car',
                         'Car',
                         'navigate',
                         'traffic',
                         'isValid',
                         '~Car'],
             'variables': ['nwheels', 'maxrpm', 'maxspeed', 'manufacturer']},
 'namespace': 'util',
 'protected-attrib': {'manufacturer': {'briefdescription': '',
                                       'definition': 'str util::Car::manufacturer',
                                       'detaileddescription': 'The manufacturer of the car. This could be anything from Saturn to Porche. ',
                                       'type': 'str'},
                      'maxrpm': {'briefdescription': '',
                                 'definition': 'double util::Car::maxrmp',
                                 'detaileddescription': 'The maximum rmp this car can attain',
                                 'type': 'double'},
                      'maxspeed': {'briefdescription': 'The top speed of the car',
                                   'definition': 'double util::Car::maxspeed',
                                   'detaileddescription': '',
                                   'type': 'double'},
                      'nwheels': {'briefdescription': 'The number of wheels on the car. ',
                                  'definition': 'uint util::Car::nwheels',
                                  'detaileddescription': '',
                                  'type': 'uint'}},
 'public-func': {'Car': {'arg_string': '()',
                         'args': None,
                         'briefdescription': 'Default constructor. ',
                         'definition': 'util::Car::Car',
                         'detaileddescription': 'A very simple car class that can do the basics. This car can navigate, get a traffic report, and verify that it is indeed a valid car. ',
                         'ret_type': None},
                 'Car1': {'arg_string': '(const Car &other)',
                          'args': OrderedDict({'other': {'type': 'const '}}),
                          'briefdescription': 'Copy constructor. This literally makes a clone of the Car that is passed in.',
                          'definition': 'util::Car::Car',
                          'detaileddescription': '',
                          'ret_type': None},
                 'Car2': {'arg_string': '(uint nwheels, str manufacturer)',
                          'args': OrderedDict({'manufacturer': {'type': 'str'},
                                   'nwheels': {'type': 'uint'}}),
                          'briefdescription': '',
                          'definition': 'util::Car::Car',
                          'detaileddescription': 'Construct a car by specifying how many wheels it should have and who the manufacturer is.',
                          'ret_type': None},
                 'isValid': {'arg_string': '()',
                             'args': None,
                             'briefdescription': 'Checks if the object is really a car. Basically sees that is has all the components of a car.',
                             'definition': 'bool util::Car::isValid',
                             'detaileddescription': '',
                             'ret_type': 'bool'},
                 'navigate': {'arg_string': '(str where, float32 howFast, Date when)',
                              'args': OrderedDict([('where', {'type': 'str'}),
                                                  ('howFast', {'type': 'float32'}),
                                                  ('when', {'type': 'Date'}),
                                                   ]),
                              'briefdescription': 'Has the car drive to a specified location',
                              'definition': 'std::vector< int32> util::Car::navigate',
                              'detaileddescription': '',
                              'ret_type': 'std::vector< uint32 >'},
                 'traffic': {'arg_string': '(std::vector< int32 > &coord) const',
                             'args': OrderedDict({'coord': {'type': 'std::vector< unit32 > const &'}}),
                             'briefdescription': '',
                             'definition': 'str util::Car::traffic',
                             'detaileddescription': 'Check the traffic at a given location. The input parameter is a vector of integers specifying the latitude and longitude of the position where the traffic should be checked.',
                             'ret_type': 'str'},
                 '~Car': {'arg_string': '()',
                          'args': None,
                          'briefdescription': 'A destructor. ',
                          'definition': 'hbs::Car::~Car',
                          'detaileddescription': '',
                          'ret_type': None}}}

@unit
def test_classdocstr():
    exp = \
"""A very simple car class that can do the basics. This car can
navigate, get a traffic report, and verify that it is indeed a valid
car.

Attributes
----------
nwheels (uint) : The number of wheels on the car.
maxrpm (double) : The maximum rmp this car can attain
maxspeed (double) : The top speed of the car
manufacturer (str) : The manufacturer of the car. This could be
    anything from Saturn to Porche.


Methods
-------
Car
~Car
isValid
navigate
traffic

Notes
-----
This class was defined in Cars.h

The class is found in the "util" namespace

"""
    actual = class_docstr(car_dict)
    print('-------- Expected Class docstring --------')
    print(exp)
    print('-------- Actual Class docstring --------')
    print(actual)

    # Strip whitespace before testing b/c editor config
    assert_equal(exp.strip(), actual.strip())

@unit
def test_funcdocstr():
    exp = \
"""Has the car drive to a specified location

Parameters
----------
where : str

howFast : float32

when : Date

Returns
-------
res1 : std::vector< uint32 >

"""
    actual = func_docstr(car_dict['public-func']['navigate'], is_method=True)
    print('-------- Expected Class docstring --------')
    print(exp)
    print('-------- Actual Class docstring --------')
    print(actual)

    # Strip whitespace before testing b/c editor config
    assert_equal(exp.strip(), actual.strip())
