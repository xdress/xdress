"""This module provides the architecture for creating and handling xdress plugins.

:author: Anthony Scopatz <scopatz@gmail.com>

The purpose of xdress is to be as modular and extensible as possible, allowing for
developers to build and execute their own tools as needed.  As such, xdress has
a very nimble plugin interface that easily handles run control, adding arguments to
the command line interface, setting up & validating the run control, command
execution, and teardown.  In fact, the entire xdress execution is based on this
plugin architecture.  You can be certain that this is well supported feature and
not some hack'd add on.

Writing Plugins
===============
Writing plugins is easy!  You simply need to have a variable named ``XDressPlugin``
in a module.  Say your module is called ``mymod`` and lives in a package ``mypack``,
then xdress would know this plugin by the name ``"mypack.mymod"``.  This is exactly
the same string that you would use to do an absolute import of ``mymod``.

To expose this plugin to an xdress execution, either add it to the ``plugins``
variable in your ``xdressrc.py`` file::

    from xdress.utils import DEFAULT_PLUGINS
    plugins = list(DEFAULT_PLUGINS) + ['mypack.mymod']

Or you can add it on the command line::

    ~ $ xdress --plugins xdress.stlwrap xdress.autoall xdress.cythongen mypack.mymod

Note that in both of the above cases we retain normal functionality by including
the default plugins that come with xdress.

The ``XDressPlugin`` variable must be callable with no arguments and return a
variable with certain attributes.  Normally this is done as a class but through
the magic of duck typing it doesn't have to be.  The ``Plugin`` class is provided
as a base class which implements a minimal, zero-work interface.  This is useful
for inheriting your modules plugin from.  You need only override the attributes
you want.  Again, inheriting from ``Plugin`` is suggested but not required.

Interface
---------
:requires: This is a list of module names or a function that returns such a list.
    The names in this list will be loaded and executed in order prior to this plugin.
    If multiple plugins require the same upstream plugin, the upstream on will only
    be run once.
:defaultrc: This is a dictionary or run control instance that maps run control
    parameters to their default values if they are otherwise not specified.  To
    make a parameter have to be given by the user, set the value to the singleton
    ``xdress.utils.NotSpecified``.  Parameters with the same name in different
    plugins will clobber each other, with the last plugin's value being ultimately
    assigned.  The exception to this is if a later plugin's parameter value is
    ``NotSpecified`` then the previous plugin value will be retained.  See the
    ``RunControl`` class for more details.  Generally it is not advised for two
    plugins to share run control parameter names unless you *really* know what
    you are doing.
:rcupdaters: This may be a dict, another mapping, a function which returns a mapping.
    The keys are the string names of the run control parameters.  The values
    are callables which indicate how to update or merge two rc parameters with
    this key. The callable should take two instances and return a copy that
    represents the merger, e.g. ``lambda old, new: old + new``.  One useful example
    is for paths.  Normally you want new paths to prepend old ones::

        rcupdaters = {'includes': lambda old, new: list(new) + list(old)}

    If a callable is not supplied for an rc parameter then the the default
    behaviour is to simply override the old value with the new one.
:rcdocs: This may be a dict, another mapping, a function which returns a mapping.
    The keys are the string names of the run control parameters.  The values
    are docstrings for the rc parameters.
:update_argparser(parser):  This is method that takes an argparse.ArgumentParser()
    instance and modifies it in-place.  This allows for run control parameters to be
    exposed as command line arguments and options.  Default arguments in
    ``parser.add_argument()`` values should not be given, or should only be set to
    ``Not Specified``.  This is to prevent collisions with the run controller.
    Default values should instead be given in the plugin's ``defaultrc``.  Thus
    argument names or the ``dest`` keyword argument should match the keys in
    ``defaultrc``.
:setup(rc): Performs all setup tasks needed for this plugin.  This may include
    validation and munging of the run control object (rc) as well as creating
    directories and files in the OS environment.  If needed, the rc should be
    modified in-place so that changes propagate to other plugins and further calls
    on this plugin. This should return None.
:execute(rc): Performs the heavy lifting of the plugin, which may require a run
    controller.If needed, the rc should be modified in-place so that changes
    propagate to other plugins and further calls on this plugin. This should
    return None.
:teardown(rc): Performs any cleanup tasks needed by the plugin, including removing
    temporary files.  If needed, the rc should be modified in-place so that changes
    propagate to other plugins and further calls on this plugin. This should
    return None.
:report_debug(rc):  Generates and returns a message to report in the ``debug.txt``
    file in the event that execute() fails and additional debugging information is
    requested.  This message is a string.


Example
-------
Here is simple, if morbid, plugin example::

    from xdress.plugins import Plugin

    class XDressPlugin(Plugin):
        '''Which famous person was executed?'''

        # everything should require base, it is useful!
        requires = ('xdress.base',)

        defaultrc = {
            'choices': ['J. Edgar Hoover', 'Hua Mulan', 'Leslie'],
            'answer': 'Joan of Arc',
            }

        rcupdaters = {'choices': lambda old, new: list(new) + list(old)}

        rcdocs = {'choices': "Possible answers.",
                  'answer': "The correct answer"}

        def update_argparser(self, parser):
            # Note, no 'default=' keyword arguments are given
            parser.add_argument('-c', '--choices', action='store', dest='choices',
                                nargs="+", help="famous people chocies")
            parser.add_argument('-a', '--answer', action='store', dest='answer',
                                help="person who was executed")

        def setup(self, rc):
            '''Ensures that Joan of Arc is a choice.'''
            if 'Joan of Arc' not in rc.choices:
                rc.choices.append('Joan of Arc')

        def execute(self, rc):
            '''Kills Joan...'''
            if rc.answer == 'Joan of Arc':
                print('Joan has met an untimely demise!')
            else:
                raise ValueError('Joan of Arc was executed, not ' + rc.answer)

        def report_debug(self, rc):
            return "the possible choices were " + str(rc.choices)


Plugins API
===========
"""
import os
import io
import sys
import warnings
import importlib
import argparse
import textwrap

from .utils import RunControl, NotSpecified, nyansep

if sys.version_info[0] >= 3:
    basestring = str

class Plugin(object):
    """A base plugin for other xdress pluigins to inherit.
    """

    requires = ()
    """This is a sequence of strings, or a function which returns such, that
    lists the module names of other plugins that this plugin requires.
    """

    defaultrc = {}
    """This may be a dict, RunControl instance, or other mapping or a function
    which returns any of these.  The keys are string names of the run control
    parameters and the values are the associated default values.
    """

    rcupdaters = {}
    """This may be a dict, another mapping, a function which returns a mapping.
    The keys are the string names of the run control parameters.  The values
    are callables which indicate how to update or merge two rc parameters with
    this key. The callable should take two instances and return a copy that
    represents the merger, e.g. ``lambda old, new: old + new``.  One useful example
    is for paths.  Normally you want new paths to prepend old ones::

        rcupdaters = {'includes': lambda old, new: list(new) + list(old)}

    If a callable is not supplied for an rc parameter then the the default
    behaviour is to simply override the old value with the new one.
    """

    rcdocs = {}
    """This may be a dict, another mapping, a function which returns a mapping.
    The keys are the string names of the run control parameters.  The values
    are docstrings for the rc parameters.
    """

    def __init__(self):
        """The __init__() method may take no arguments or keyword arguments."""
        pass

    def update_argparser(self, parser):
        """This method takes an argparse.ArgumentParser() instance and modifies
        it in-place.  This allows for run control parameters to be modified from
        as command line arguments.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser to be updated.  Arguments defaults should not be given, or
            if given should only be ``xdress.utils.Not Specified``.  This is to
            prevent collisions with the run controller.  Default values should
            instead be given in this class's ``defaultrc`` attribute or method.
            Argument names or the ``dest`` keyword argument should match the keys
            in ``defaultrc``.

        """
        pass

    def setup(self, rc):
        """Performs all setup tasks needed for this plugin.  This may include
        validation and munging of the run control object as well as creating
        the portions of the OS environment.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        """
        pass

    def execute(self, rc):
        """Performs the actual work of the plugin, which may require a run controller.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        """
        pass

    def teardown(self, rc):
        """Performs any cleanup tasks needed by the plugin.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        """
        pass

    def report_debug(self, rc):
        """A message to report in the event that execute() fails and additional
        debugging information is requested.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        Returns
        -------
        message : str or None
            A debugging message to report.  If None is returned, this plugin is
            skipped in the debug output.

        """
        pass


class Plugins(object):
    """This is a class for managing the instantiation and execution of plugins.

    The execution and control of plugins should happen in the following order:

    1. ``build_cli()``
    2. ``merge_rcs()``
    3. ``setup()``
    4. ``execute()``
    5. ``teardown()``
    6. ``exit()``

    """

    def __init__(self, modnames, loaddeps=True):
        """Parameters
        ----------
        modnames : list of str
            The module names where the plugins live.  Plugins must have the name
            'XDressPlugin' in the these modules.
        loaddeps: bool, optional
            Flag for automatically loading dependencies, should only be False in
            a limited set of circumstances.

        """
        self.plugins = []
        self.modnames = []
        self._load(modnames, loaddeps=loaddeps)
        self.parser = None
        self.rc = None
        self.rcdocs = {}
        self.warnings = []

    def _load(self, modnames, loaddeps=True):
        for modname in modnames:
            if modname in self.modnames:
                continue
            mod = importlib.import_module(modname)
            plugin = mod.XDressPlugin()
            req = plugin.requires() if callable(plugin.requires) else plugin.requires
            req = req if loaddeps else ()
            self._load(req, loaddeps=loaddeps)
            self.modnames.append(modname)
            self.plugins.append(plugin)

    def build_cli(self):
        """Builds and returns a command line interface based on the plugins.

        Returns
        -------
        parser : argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser("Generates XDress API",
                    conflict_handler='resolve', argument_default=NotSpecified)
        for plugin in self.plugins:
            plugin.update_argparser(parser)
        self.parser = parser
        return parser

    def _setshowwarning(self):
        def showwarning(message, category, filename, lineno, file=None, line=None):
            if self.rc.debug:
                debugmsg = "{0}: '{1}' from {2}:{3}"
                debugmsg = debugmsg.format(category.__name__, message,
                                           filename, lineno)
                self.warnings.append(debugmsg)
            printmsg = "WARNING: {0}: {1}"
            printmsg = printmsg.format(category.__name__, message)
            print(printmsg)
        warnings.showwarning = showwarning

    def merge_rcs(self):
        """Finds all of the default run controllers and returns a new and
        full default RunControl() instance.  This has also merged all of
        the rc updaters in the process."""
        rc = RunControl()
        rcdocs = self.rcdocs
        for plugin in self.plugins:
            drc = plugin.defaultrc
            if callable(drc):
                drc = drc()
            rc._update(drc)
            uprc = plugin.rcupdaters
            if callable(uprc):
                uprc = uprc()
            rc._updaters.update(uprc)
            docs = plugin.rcdocs
            if callable(docs):
                docs = docs()
            rcdocs.update(docs)
        self.rc = rc
        self._setshowwarning()
        return rc

    def setup(self):
        """Performs all plugin setup tasks."""
        rc = self.rc
        try:
            for plugin in self.plugins:
                plugin.setup(rc)
        except Exception as e:
            self.exit(e)

    def execute(self):
        """Preforms all plugin executions."""
        rc = self.rc
        try:
            for plugin in self.plugins:
                plugin.execute(rc)
        except Exception as e:
            self.exit(e)

    def teardown(self):
        """Preforms all plugin teardown tasks."""
        rc = self.rc
        try:
            for plugin in self.plugins:
                plugin.teardown(rc)
        except Exception as e:
            self.exit(e)

    def exit(self, err=0):
        """Exits the process, possibly printing debug info."""
        rc = self.rc
        if rc.debug:
            import traceback
            sep = nyansep + '\n\n'
            msg = u''
            if err != 0:
                msg += u'{0}xdress failed with the following error:\n\n'.format(sep)
                msg += traceback.format_exc()
            if len(self.warnings) > 0:
                warnmsg = u'\n{0}xdress issued the following warnings:\n\n{1}\n\n'
                warnmsg = warnmsg.format(sep, "\n".join(self.warnings))
                msg += warnmsg
            msg += '\n{0}Run control run-time contents:\n\n{1}\n\n'.format(sep,
                                                                    rc._pformat())
            for plugin in self.plugins:
                plugin_msg = plugin.report_debug(rc) or ''
                if 0 < len(plugin_msg):
                    msg += sep
                    msg += plugin_msg
            with io.open(os.path.join(rc.builddir, 'debug.txt'), 'wt') as f:
                f.write(msg)
            if err != 0:
                raise
        else:
            if err == 0:
                sys.exit()
            else:
                sys.exit('ERROR: ' + str(err))

def summarize_rcdocs(modnames, headersep="=", maxdflt=2000):
    """For a list of plugin module names, return a rST string that
    summarizes the docstrings for all run control parameters.
    """
    nods = "No docstring provided."
    template = ":{0!s}: {1!s}, *default:* {2}."
    docstrs = []
    tw = textwrap.TextWrapper(width=80, subsequent_indent=" "*4)
    for modname in modnames:
        moddoc = str(modname)
        moddoc += "\n"+ headersep * len(moddoc) + "\n"
        plugins = Plugins([modname], loaddeps=False)  # get a lone plugin
        plugins.merge_rcs()
        rc = plugins.rc
        rcdocs = plugins.rcdocs
        for key in sorted(rc._dict.keys()):
            dflt = getattr(rc, key)
            rdflt = repr(dflt)
            rdflt = rdflt if len(rdflt) <= maxdflt else "{0}.{1} instance".format(
                    dflt.__class__.__module__, dflt.__class__.__name__)
            rcdoc = template.format(key, rcdocs.get(key, nods), rdflt)
            moddoc += "\n".join(tw.wrap(rcdoc)) + '\n'
        docstrs.append(moddoc)
    return "\n\n\n".join(docstrs)
