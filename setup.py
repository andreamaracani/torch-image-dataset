from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Image datasets for Pytorch.'
LONG_DESCRIPTION = 'Some classes that help to load images, as dataset, for Pytorch projects.'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="torch-image-dataset", 
        version=VERSION,
        author="Andrea Maracani",
        author_email="<andrea.maracani@iit.it>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'pytorch', 'dataset', 'image'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)