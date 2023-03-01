from setuptools import setup

setup(
    name='CombinatorialGymnasium',
    version='0.0.0.2',
    packages=['combinatorial_problems'],
    url='',
    license='GPL',
    author='Jose E. Aguilar Escamilla',
    author_email='jose.efraim.a.e@gmail.com',
    description='A combinatorial gymnasium for testing Reinforcement Learning.',
    install_requires=[
        "gymnasium",
        "pygame",
        "numpy",
    ]
)
