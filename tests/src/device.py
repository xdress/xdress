mod = {'TwoNums': {
    'extra': {
        'max_instances_guess': 2,
        'pyx': """
    def call_from_c(self, x, y):
        rtn = (<c_pydevice.TwoNums *> self._inst).op(<double> x, <double> y)
        return float(rtn)
"""},
    },
    }
