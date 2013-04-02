#!/bin/bash
s3cmd -rvpFH --progress put /home/scopatz/xdress/docs/_build/html/* s3://xdress/
