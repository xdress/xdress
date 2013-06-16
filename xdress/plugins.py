"""This module provides the architechture for creating and handling xdress plugins.

:author: Anthony Scopatz <scopatz@gmail.com>

"""
import os
import sys
import importlib
import argparser

from .utils import RunControl, NotSpecified

if sys.version_info[0] >= 3:
    basestring = str

class Plugin(object):
    """A base plugin for other xdress pluigins to inherit.
    """

    defaultrc = {}
    """This may be a dict, RunControl instance, or othet mapping or a function
    which returns any of these.  The keys are string names of the run control 
    parameters and the values are the associated default values.
    """

    def __init__(self):
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
            prevent collisions with the run controler.  Default values should 
            instead be given in this class's ``defaultrc`` attribute or method.
            Argument names or ``dest``s should match the keys in ``defaultrc``.

        """
        pass

    def setup(self, rc)
        """Performs all setup tasks needed for this plugin.  This may include 
        validation and munging of the run control object as well as creating 
        the portions of the OS environment.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        """
        pass

    def execute(self, rc)
        """Performs the actual work of the plugin, which may require a run controler.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        """
        pass

    def teardown(self, rc)
        """Performs any cleanup tasks needed by the plugin.

        Parameters
        ----------
        rc : xdress.utils.RunControl

        """
        pass
    

class Plugins(object):
    """This is a class for managing the instantiation and execution of plugins.

    The execution and control of plugins should happen in the following order:

    1. ``build_cli()``
    2. ``merge_defaultrcs()``
    3. ``setup()``
    4. ``execute()``
    5. ``teardown()``
       
    """

    def __init__(self, modnames):
        """Parameters
        ----------
        modnames : list of str
            The module names where the plugins live.  Plugins must have the name
            'XDressPlugin' in the these modules.

        """
        plugins = []
        for modname in modnames:
            mod = importlib.import_module(modname)
            plugin = mod.XDressPlugin()
            plugins.append(plugin)
        self.plugins = plugins
        self.parser = None
        self.rc = None

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

    def merge_defaultrcs(self):
        """Finds all of the default run controlers and returns a new and 
        full default RunControl() instance."""
        rc = RunControl()
        for plugin in plugins:
            drc = plugin.defaultrc
            if callable(drc):
                drc = drc()
            rc._update(drc)
        self.rc = rc
        return rc

    def setup(self):
        """Preforms all plugin setup tasks."""
        rc = self.rc
        for plugin in plugins:
            plugin.setup(rc)

    def execute(self):
        """Preforms all plugin executions."""
        rc = self.rc
        for plugin in plugins:
            plugin.execute(rc)

    def teardown(self):
        """Preforms all plugin teardown tasks."""
        rc = self.rc
        for plugin in plugins:
            plugin.teardown(rc)
