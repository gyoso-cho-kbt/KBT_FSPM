=========================
README for Senesc-Wheat
=========================

This is the Senesc-Wheat model, a model of senescence adapted to wheat.


Prerequisites
=============

* To run the model: 
    * Python >= 2.7, http://www.python.org/
    * Pandas >= 0.14.0, http://pandas.pydata.org/
* To build the documentation: Sphinx >= 1.1.3, http://sphinx-doc.org/
* To run the tests with Nose:
    * Nose >= 1.3.0, http://nose.readthedocs.org/
    * NumPy >= 1.7.2, http://www.numpy.org/
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

With Nose::

    nosetests
    
Without Nose, install the package (see section "Installing") and run::

    python test/unit/test_senescwheat.py


Contact
=======

Please send a mail to senesc-wheat@groupes.renater.fr.


Contributing
============

#. Check for open issues or open a fresh issue to start a discussion around a
   feature idea or a bug: https://sourcesup.renater.fr/tracker/?group_id=1792.
#. If you feel uncomfortable or uncertain about an issue or your changes, feel
   free to email senesc-wheat@groupes.renater.fr.
