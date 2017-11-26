# MOVED TO SEPARATE REPO - https://github.com/jimmyhmiller/account-number/



# Account Number Parser

A program made to parse account numbers from a file. See [Code Kata](http://codingdojo.org/kata/BankOCR/) for details.

## Usage

In order to run the program you need to choose a scenario (corresponding to the sections in the Kata). If a file is not supplied, the default file for that scenario will be chosen.

```bash
lein run scenario-1 [file]
lein run scenario-2 [file]
lein run scenario-3 [file]
```

## Code Structure Overview

The core functionality of the app lies fittingly enough in `account-number.core`. Core includes all the methods that are needed to parse the ascii file, validate it, and produce error messages.

 `account-number.main` is the main entry point to applications and just combines the functionality from core to achieve the different scenario goals.

Both spec and utils are used at dev time rather than at runtime. Spec contains various clojure.spec definitions to provide basic santity checks around functions. There are many more properties that could be specified, but this was mainly written after the fact to ensure I hadn't made mistakes. It found a couple minor ones and saved me some time as I refactored some code.

Utils mainly consists of printing and generating. From utils, you can generate random files that can be used for testing purposes. It can generate valid, invalid, ill-formed, and off-by-one account numbers. Off-by-one account number generation should make implementing scenario-4 easier. It provides a very quick way to have some examples to test on.
