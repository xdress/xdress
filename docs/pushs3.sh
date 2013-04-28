#!/bin/bash
s3cmd -rvpFHP --progress put /home/scopatz/xdress/docs/_build/html/* s3://xdress/
