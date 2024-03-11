import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "lsstools"
copyright = "2024, lsstools"
author = "Ruiyang Zhao"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []
default_role = "py:obj"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/zhaoruiyang98/lsstools",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_context = {"default_mode": "light"}
html_copy_source = False

autosummary_generate = True
autosummary_ignore_module_all = False
latex_engine = "xelatex"
numpydoc_validation_checks = {
    "all",
    "GL01",
    "GL08",
    "ES01",
    "PR01",
    "PR07",
    "RT01",
    "RT03",
    "SA04",
    "SA01",
    "EX01",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}
