"""
AutoSplit Studio - Krita plugin.
Registers the docker that splits a character image into parts via ComfyUI SAM3.
"""
from krita import Krita, DockWidgetFactory, DockWidgetFactoryBase
from .autosplit_docker import AutoSplitDocker

Krita.instance().addDockWidgetFactory(
    DockWidgetFactory(
        "autosplit_studio",
        DockWidgetFactoryBase.DockRight,
        AutoSplitDocker,
    )
)
