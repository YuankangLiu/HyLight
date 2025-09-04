# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

# local
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + '/src/hylightpy' )

# GitHub Actions
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#print(">>> sys.path[0]:", sys.path[0])

#print(">>> Sphinx is using:", hylightpy.__file__)
#importlib.reload(hylightpy)

project = 'HyLight'
copyright = '2025, Yuankang Liu, Tom Theuns'
author = 'Yuankang Liu, Tom Theuns'
release = '0.0.11'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
