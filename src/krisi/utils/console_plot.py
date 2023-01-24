from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin


class plotextMixin(JupyterMixin):
    def __init__(self, data, plotting_function, title=""):
        self.decoder = AnsiDecoder()
        self.title = title
        self.data = data
        self.plotting_function = plotting_function

    def __rich_console__(self, console, options):
        self.width = options.max_width or console.width
        self.height = options.height or console.height
        canvas = self.plotting_function(self.data, self.width, self.height, self.title)
        self.rich_canvas = Group(*self.decoder.decode(canvas))
        yield self.rich_canvas
