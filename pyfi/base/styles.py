import itables
from itables import init_notebook_mode
import itables.options as opt

def set():
    init_notebook_mode(all_interactive=True)
    opt.maxBytes = 0
    opt.style = "table-layout:auto;width:auto"
    opt.classes="display nowrap compact"
    opt.column_filters="footer"
    opt.order=[0, 'desc']
    opt.lengthMenu = [20, 50]