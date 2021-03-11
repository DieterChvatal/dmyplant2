dMyplant
========

Python Interface to INNIO MyPlant.

Easy access to INNIO Assets via Myplant API. Download Pandas Dataframes
on a predefined Validation Fleet. Calculate demonstrated Reliability
based on Success run Method and Lipson Equality approach as described in
A.Kleyners Paper "Reliability Demonstration in Product Validation
Testing".

figure:: header.png
:alt: 

Installation
------------

**Windows:**

.. code:: sh

*inside Anaconda or Miniconda prompt:*

..1. git clone https://github.com/DieterChvatal/dmyplant.git::

*cd into the folder and run*

2. python setup.py develop

*this creates links to this package in the Anaconda package location.
to remove these links use*

3. python setup.py develop --uninstall

*Now you can modify and extend the package in place ...*

Usage example
-------------

*create an input.csv file with your myplant assets, e.g.*::

.. code::
code::n;Validation Engine;serialNumber;val start;oph@start;starts@start
0;POLYNT - 2 (1145166-T241) --> Sept;1145166;12.10.2020;31291;378
1;REGENSBURG;1175579;14.09.2020;30402;1351
2;ROCHE PENZBERG KWKK;1184199;27.04.2020;25208;749
3;ECOGEN ENERGY SYSTEMS BVBA;1198719;15.10.2020;28583;711
4;BMW REGENSBURG M3;1243360;17.08.2020;63893;2016
5;REGENSBURG;1243362;07.09.2020;62765;;
6;ABINSK;1250575;15.06.2020;758;;
7;PROSPERITY WEAVING MILLS LTD - 1 (1351388-X243);1250578;12.10.2020;0;352
8;SOTERNIX RENOVE;1310773;25.09.2020;18439;1218
9;BMW MÜNCHEN;1319133;31.08.2020;4532;581




Release History
---------------

-  0.0.1
-  Work in progress

Meta
----

Your Name – dieter.chvatal@innio.com

Distributed under the MIT license. See ``LICENSE`` for more information.

`https://github.com/DieterChvatal/dmyplant2 <https://github.com/DieterChvatal/>`__


Contributing
------------

1. Fork it (https://github.com/DieterChvatal/dmyplant2)
2. Create your feature branch (``git checkout -b feature/fooBar``)
3. Commit your changes (``git commit -am 'Add some fooBar'``)
4. Push to the branch (``git push origin feature/fooBar``)
5. Create a new Pull Request

