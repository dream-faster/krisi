# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Krisi - Forecasting Evaluation and Exploration"
copyright = "2022, Myalo UG - Daniel Szemerey, Mark Aron Szulyovszky"
author = "Myalo UG - Daniel Szemerey, Mark Aron Szulyovszky"
release = "0.0.1"

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.coverage",
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "logo_white.svg"
html_theme = "sphinx_material"
html_static_path = ["_static"]

# Set link name generated in the top bar.
html_title = "Krisi"

# Material theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    "nav_title": "Krisi - Forecasting Evaluation & Exploration",
    # Set you GA account ID to enable tracking
    "google_analytics_account": "G-Z22E374SDG",
    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    "base_url": "https://dream-faster.github.io/krisi",
    # Set the color and the accent color
    "color_primary": "yellow",
    "color_accent": "light-yellow",
    # Set the repo location to get a badge with stats
    "repo_url": "https://github.com/dream-faster/krisi/",
    "repo_name": "Krisi",
    # Visible levels of the global TOC; -1 means unlimited
    "globaltoc_depth": 3,
    # If False, expand all TOC entries
    "globaltoc_collapse": False,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
    "nav_links": [
        {
            "href": "get_started",
            "title": "Get Started",
            "internal": True,
        },
        {
            "href": "api",
            "title": "API",
            "internal": True,
        },
        {
            "href": "contribute",
            "title": "Contribute",
            "internal": True,
        },
        {
            "href": "https://github.com/dream-faster/krisi/issues",
            "title": "Submit Issues",
            "internal": False,
        },
        {
            "href": "https://dreamfaster.ai",
            "title": "About us",
            "internal": False,
        },
    ],
}
