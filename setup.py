from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='pypot',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'pypot': ["data/*.csv"],
    },
)
