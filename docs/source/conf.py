# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
sys.path.insert(0, "../")
sys.path.insert(0, "../../")

project = 'StructureFactorUtils'
copyright = '2023, Henry Ding'
author = 'Henry Ding'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    "nbsphinx"
]
autosummary_generate = True 

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# Looks for objects in external projects
intersphinx_mapping = {
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}
