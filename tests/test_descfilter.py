from __future__ import print_function
import os
from copy import deepcopy
import pprint
from nose.tools import assert_equal
from tools import unit
from xdress.typesystem import TypeMatcher
from xdress import descfilter as df

car_class = {
    'name': 'Car',
    'namespace': 'util',
    'parents': None,
    'attrs': {
        'nwheels': 'uint32',
        'maxrpm': 'float64',
        'maxspeed': 'float32',
        'manufacturer': 'str'},
    'methods': {
        ('Car',): None,
        ('~Car',): None,
        ('navigate', ('where', 'str'),
                     ('howFast', 'float32'),
                     ('when', 'Date')): ('vector', 'uint32'),
        ('traffic', ('coord', (('vector', 'int32', 'const'), '&'))): 'str',
        ('isValid',): 'bool',
        }
    }

plane_class = {
    'name': 'Plane',
    'namespace': 'util',
    'parents': None,
    'attrs': {
        'homeAirport': 'str',
        'maxrpm': 'float64',
        'maxspeed': 'float32',
        'manufacturer': 'str',
        'position': ('vector', 'float64')},
    'methods': {
        ('Plane',): None,
        ('Plane', ('homeAirport', 'str')): None,
        ('~Plane',): None,
        ('navigate', ('where', 'str'),
                     ('howFast', 'float32'),
                     ('when', 'Date')): ('vector', 'uint32'),
        ('land', ('coord', (('vector', 'int32', 'const'), '&'))): 'str',
        ('dogfight', ('who', 'Chopper'), ('why', 'str')): 'bool',
        ('isOnFire',): 'bool',
        }
    }

filt_types_list = ['str', (('vector', 'int32', 'const'), '&')]
filt_types_dict = {'Car': ['uint32'],
                   'Plane': ['Chopper', 'float32']}

@unit
def test_typefilter_list():
    skips = [TypeMatcher(i) for i in filt_types_list]
    car_class_copy = deepcopy(car_class)
    df.modify_desc(skips, car_class_copy)
    exp = {'name': 'Car', 'namespace': 'util', 'parents': None,
    'attrs': {
        'nwheels': 'uint32',
        'maxrpm': 'float64',
        'maxspeed': 'float32'},
    'methods': {
        ('Car',): None,
        ('~Car',): None,
        ('isValid',): 'bool'}
        }

    print('********\nCar class stuff (actual then expected):\n')
    pprint.pprint(car_class_copy)
    pprint.pprint(exp)
    assert_equal(car_class_copy, exp)

@unit
def test_typefilter_dict():
    skiptypes = {'Car': [TypeMatcher(i) for i in filt_types_dict['Car']],
                 'Plane': [TypeMatcher(i) for i in filt_types_dict['Plane']]}

    skips_car = skiptypes['Car']
    skips_plane = skiptypes['Plane']

    # Copy the dictionaries that are going to be changed in place
    car_class_copy = deepcopy(car_class)
    plane_class_copy = deepcopy(plane_class)

    df.modify_desc(skips_car, car_class_copy)
    df.modify_desc(skips_plane, plane_class_copy)

    exp_car = {
        'name': 'Car',
        'namespace': 'util',
        'parents': None,
        'attrs': {
            'maxrpm': 'float64',
            'maxspeed': 'float32',
            'manufacturer': 'str'},
        'methods': {
            ('Car',): None,
            ('~Car',): None,
            ('traffic', ('coord', (('vector', 'int32', 'const'), '&'))): 'str',
            ('isValid',): 'bool'}
    }

    exp_plane = {
        'name': 'Plane',
        'namespace': 'util',
        'parents': None,
        'attrs': {
            'homeAirport': 'str',
            'maxrpm': 'float64',
            'manufacturer': 'str',
            'position': ('vector', 'float64')},
        'methods': {
            ('Plane',): None,
            ('Plane', ('homeAirport', 'str')): None,
            ('~Plane',): None,
            ('land', ('coord', (('vector', 'int32', 'const'), '&'))): 'str',
            ('isOnFire',): 'bool'}
    }

    print('********\nCar class stuff (actual then expected):\n')
    pprint.pprint(car_class_copy)
    pprint.pprint(exp_car)
    print('********\nplane class stuff (actual then expected):\n')
    pprint.pprint(plane_class_copy)
    pprint.pprint(exp_plane)

    assert_equal(car_class_copy, exp_car)
    assert_equal(plane_class_copy, exp_plane)
