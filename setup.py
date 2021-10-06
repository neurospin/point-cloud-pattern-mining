import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='moving_averages',
                 # from python 3.6, dictionary order is preserved during execution
                 python_requires='>3.6',

                 version='0.0.40',
                 description='Tools for the calculation of moving averages of 3D objects',
                 author='Marco Pascucci - NeuroSpin (CEA)',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author_email='marpas.paris@gmail.com',
                 url='',
                 packages=['moving_averages', 'moving_averages.distance'],
                 install_requires=['plotly==4.14.3', 'scipy>=0.22'],

                 classifiers=[
                     "Programming Language :: Python",
                     "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
                     "Operating System :: OS Independent",
                 ]
                 )
