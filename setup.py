from setuptools import setup, find_packages
setup(
    name = "virtualecologist",
    version = "0.0.4",
    #packages = find_packages(),
    description = "Optimize transect length in ecological surveys.",

    packages = ['virtualecologist'],
      #long_description="""""",
    #scripts = ['virtual_ecologist.py'],

    # installed or upgraded on the target machine
    install_requires = ["csv", "pandas", "numpy",\
    "scipy", "matplotlib", "prettytable"],

    include_package_data = True,

    package_data = {
        # include *.txt or *.rst files:
        '': ['*.txt', '*.rst'],
        # include *.msg files:
        '': ['*.msg'],
        # include subdirectory containing example datasets:
        'virtualecologist': ['data/*.csv'],
        },

    # metadata for PyPi
    author = "Ray",
    author_email = "rblick.ecol@gmail.com",
    license = "PSF",
    keywords = "ecology model virtual ecologist",
    url = "#",   # project home page, if any
)
