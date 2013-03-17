"""Helper functions for bright API generation."""

def indent(s, n=4, join=True):
    """Indents all lines in the string or list s by n spaces."""
    spaces = " " * n
    lines = s.splitlines() if isinstance(s, basestring) else s
    if join:
        return '\n'.join([spaces + l for l in lines])
    else:
        return [spaces + l for l in lines]


def expand_default_args(methods):
    """This function takes a collection of method tuples and expands all of 
    the default arguments, returning a set of all methods possible."""
    methitems = set()
    for mkey, mrtn in methods:
        mname, margs = mkey[0], mkey[1:]
        havedefaults = [3 == len(arg) for arg in margs]
        if any(havedefaults):
            # expand default arguments  
            n = havedefaults.index(True)
            items = [((mname,)+tuple(margs[:n]), mrtn)] + \
                    [((mname,)+tuple(margs[:i]), mrtn) for i in range(n+1,len(margs)+1)]
            methitems.update(items)
        else:
            # no default args
            methitems.add((mkey, mrtn))
    return methitems
