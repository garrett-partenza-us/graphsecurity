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
or to run all CWE's
```
cd src
python3 run_juliet.py all
```

To train OWASP model:
```
cd src
python3 run_owasp.py 89
```
or to run all CWE's
```
cd src
python3 run_owasp.py all
```
### Screenshots

### Features

## Code

[![Build Status](https://qa.nuxeo.org/jenkins/buildStatus/icon?job=/nuxeo/addons_nuxeo-sample-project-master)](https://qa.nuxeo.org/jenkins/job/nuxeo/job/addons_nuxeo-sample-project-master/)

### Content

Description, sub-modules organization...

### Requirements

See [CORG/Compiling Nuxeo from sources](http://doc.nuxeo.com/x/xION)

Sample: <https://github.com/nuxeo/nuxeo/blob/master/nuxeo-distribution/README.md>

### Limitations

Sample: <https://github.com/nuxeo-archives/nuxeo-features/tree/master/nuxeo-elasticsearch>

### Build

    mvn clean install

Build options:

* ...

### Deploy (how to install build product)

Direct to MP package if any. Otherwise provide steps to deploy on Nuxeo Platform:

 > Copy the built artifacts into `$NUXEO_HOME/templates/custom/bundles/` and activate the `custom` template.

## Resources (Documentation and other links)

## Contributing / Reporting issues

Link to JIRA component (or project if there is no component for that project). Samples:

* [Link to component](https://jira.nuxeo.com/issues/?jql=project%20%3D%20NXP%20AND%20component%20%3D%20Elasticsearch%20AND%20Status%20!%3D%20%22Resolved%22%20ORDER%20BY%20updated%20DESC%2C%20priority%20DESC%2C%20created%20ASC)
* [Link to project](https://jira.nuxeo.com/secure/CreateIssue!default.jspa?project=NXP)

## License

[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html)

## About Nuxeo

Nuxeo Platform is an open source Content Services platform, written in Java. Data can be stored in both SQL & NoSQL databases.

The development of the Nuxeo Platform is mostly done by Nuxeo employees with an open development model.

The source code, documentation, roadmap, issue tracker, testing, benchmarks are all public.

Typically, Nuxeo users build different types of information management solutions for [document management](https://www.nuxeo.com/solutions/document-management/), [case management](https://www.nuxeo.com/solutions/case-management/), and [digital asset management](https://www.nuxeo.com/solutions/dam-digital-asset-management/), use cases. It uses schema-flexible metadata & content models that allows content to be repurposed to fulfill future use cases.

More information is available at [www.nuxeo.com](https://www.nuxeo.com).
