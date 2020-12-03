=========
Changelog
=========

Version 0.1
===========

#Release version Changelog

a) Parser:
- added remount since while rerunning the code error was returned
- "pointsXD = {1 : [], 0 : []}" changed to "pointsXD = {0 : [], 1 : []}"
- all data had to be cast to float since parsing output was string
- packaged parser into function beacuse execution analysis relies on it


a) Normalization:
- packed it into a function
- added rounding numbers to two decimal points
- first three lines of code are not necessary 

b) Last cell:
- Can we delete?

- thanks to changes to parser i automated your part and added number of points vs Time

a) classifyAPoint:
- printing option so that we can choose if we want to print the blocks of points

