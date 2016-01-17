from setuptools import setup, find_packages
setup(
    name = "virtualecologist",
    version = "0.1",
    #packages = find_packages(),

    packages = ['virtualecologist','tests'],
      long_description="""\
      Rationalise transect length in ecological surveys. Quantify
      your error rate between surveys, or with a collegue, to determine
      the optimal transect length in an ecological monitoring programme.
      """,
    #scripts = ['virtual_ecologist.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ["csv", "pandas", "numpy", "math",\
    "scipy", "matplotlib", "prettytable", "re"],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        '': ['*.msg'],
    },
    data_files=[('example', ['example/fulldata.csv', 'example/pilotdata.csv'])]

    # metadata for upload to PyPI
    author = "Ray",
    author_email = "rblick.ecol@gmail.com",
    description = "",
    license = "PSF",
    keywords = "ecology model virtual ecologist",
    url = "#",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
