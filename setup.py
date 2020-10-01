#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

setup_requirements = []

test_requirements = []

setup(
    author="Stefan Heid",
    author_email='stefan.heid@upb.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Simple OpenAI Gym implementation of a simple water treatment plant. The gym is created with constrained reinforcement learning in mind.",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='OpenAI Gym, Reinforcement Learning',
    name='constraint_water_treatment_gym',
    packages=find_packages(include=['constraint_water_treatment_gym', 'constraint_water_treatment_gym.*']),
    setup_requires=setup_requirements,
    extras_require=dict(examples=['stable-baselines3']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/stheid/constraint_water_treatment_gym',
    version='0.1.0',
    zip_safe=False,
)
