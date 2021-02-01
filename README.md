# Geometric Deep Learning for Security
## About the Project

This project represents a geometric deep learning approach to automated classification of common weakness enumerations in Java source code. The approach uses pytorch geometric and graph convelutional neural networks to achieve near-perfect accuracy across the Juliet Test Suite and the OWASP Benchmark. 

See Juliet and OWASP:

* <https://samate.nist.gov/SRD/testsuite.php>
* <https://owasp.org/www-project-benchmark/>

## Installation

```
git clone https://github.com/garrett-partenza-us/faastGDL
cd graphcwe
pip3 install -r requirements.txt
```

## Usage

To train Juliet model:
```
cd src
python3 run_juliet.py 89
```

To train OWASP model:
```
cd src
python3 run_owasp.py 89
```

## Author
Garrett A. Partenza. 
Towson University Software Engineering Laboratory. 
gparte1@students.towson.edu. 

## License
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

