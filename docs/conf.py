master_doc = 'index'
nbsphinx_execute = "never"

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import DePass

project = 'DePass'
author = 'lwyx'
copyright = 'zhanglabNKU'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',             
    'sphinx.ext.autosummary',         
    'sphinx.ext.intersphinx',         
    'sphinx.ext.viewcode',            
    'sphinx_autodoc_typehints',       
    'nbsphinx',                       
    'IPython.sphinxext.ipython_console_highlighting',
]

autosummary_generate = True
autoclass_content = "both"
html_show_sourcelink = False
autodoc_inherit_docstrings = True
set_type_checking_flag = True
nbsphinx_allow_errors = True
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', '**.ipynb_checkpoints']


autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
    "torch_cluster",
    "torch_spline_conv",
    "rpy2",
    "mclust",
]


html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# html_static_path = ['_static']
html_css_files = ["readthedocs-custom.css"]
# html_logo = "_static/logo.png"  # 如果有 logo 再启用
