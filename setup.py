#!/usr/bin/env python
from __future__ import print_function 
import os
import sys
import subprocess

import configure

xdress_logo = """"""

def main_body():
    print(xdress_logo)
    configure.setup()

def main():
    success = False
    try:
        main_body()
        success = True
    finally:
        configure.final_message(success)

if __name__ == "__main__":
    main()
