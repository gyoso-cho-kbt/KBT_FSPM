=======================
README for Growth-Wheat
=======================

This is Growth-Wheat model, a mechanistic model of wheat growth.


Prerequisites
=============

* To run the model:
    * Python >= 2.7, http://www.python.org/
    * Pandas >= 0.18.0, http://pandas.pydata.org/
    * Respi-Wheat, https://sourcesup.renater.fr/projects/respi-wheat/ (or send an email to respi-wheat-request@groupes.renater.fr)
* To build the documentation: Sphinx >= 1.1.3, http://sphinx-doc.org/
* To run the tests: 
    * Nose >= 1.3.0, http://nose.readthedocs.org/
    * NumPy >= 1.11.0, http://www.numpy.org/
* To get code coverage testing: Coverage >= 3.6b3, http://nedbatchelder.com/code/coverage/


Installing
==========

Use ``setup.py``::

   python setup.py install

To install in develop mode::

   python setup.py develop


Reading the docs
================

After installing::

   python setup.py build_sphinx

Then, direct your browser to ``_build/html/index.html``.


Testing
=======

To run the tests, use::

    nosetests


Contact
=======

Please send a mail to growth-wheat@groupes.renater.fr.


Contributing
============

#. Check for open issues or open a fresh issue to start a discussion around a
   feature idea or a bug: https://sourcesup.renater.fr/forum/?group_id=2213
#. If you feel uncomfortable or uncertain about an issue or your changes, feel
   free to email growth-wheat@groupes.renater.fr.
