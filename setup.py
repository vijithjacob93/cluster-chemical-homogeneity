from setuptools import setup

setup(name='clusterchemistry',
      version='1.0',
      description='This repository consists of the code used for a project studying the chemical homogeneity of open clusters in the Milky Way. It consists of a pipeline that finds cluster members using stellar kinematics for a large number of open clusters, and a novel method to re-derive abundance uncertainties for stars.',
      url='https://github.com/vijithjacob93/cluster_chemical_homogeneity.git',
      author='Vijith Jacob Poovelil',
      author_email='vijith.jacob93@gmail.com',
      license='pip',
      packages=['clusterchemistry'],
      zip_safe=False)