==============================
Constraint Water Treatment Gym
==============================


.. image:: https://img.shields.io/pypi/v/constraint_water_treatment_gym.svg
        :target: https://pypi.python.org/pypi/constraint_water_treatment_gym

.. image:: https://img.shields.io/travis/stheid/constraint_water_treatment_gym.svg
        :target: https://travis-ci.com/stheid/constraint_water_treatment_gym



OpenAI Gym implementation of a simple water treatment plant. The gym is created with constrained reinforcement learning in mind.


* Free software: GNU General Public License v3


Installation
------------

For installing the Environment run
::

    pip install constraint_water_treatment_gym


To be able to run all examples and experiments install
::

    pip install constraint_water_treatment_gym[examples]


Usage
-----
.. code-block:: python

    import gym

    if __name__ == '__main__':
        env = gym.make('constraint_water_treatment_gym:distillation-plant-v0')

        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())  # pick three continous control actions randomly
        env.close()



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
