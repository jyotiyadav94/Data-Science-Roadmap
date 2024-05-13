# Python coding challenge

Hi and welcome to the faire.ai’s engineering challenge that we hope you will 
enjoy to develop with your best skills and creativity. The following sections 
describe the problem you have to solve and what we expect from your solution.

Imagine you’re working for a fintech company that collects information from 
bank accounts of people. The company wants to implement a new function that
computes the average balance for a collection of accounts , and it 
provides you with the following datasets:

* a dataset containing all the transactions currently in possession for 
  some bank accounts. The avaliable fields are the id of the account 
  to which each transaction pertains (`account_id`), the moment at 
  which each transaction was made (`value_timestamp`), and the amount 
  of the transaction (`amount`).
* a dataset containing information about the bank accounts. The available
  fields here are the id of each account (`account_id`), the time at 
  which that account was created in the company systems (`creation_timestamp`),
  and the account balance value at `creation_timestamp`
* a dataset that specifies for each account at which date to compute the 
  average balance. The fields here are the id of the account to consider
  (`account_id`), and the time at which the result is required for each
  account (`reference_timestamp`).

Your task is to build the function that, for each account, computes the average
value of the over the 90 days before the `reference_timestamp`, i.e.

```math
\mbox{average\_booked\_balance} = \frac{\sum_{\mbox{day} \in D} \mbox{balance}_\mbox{day}}{90}
```

where

```math
D = \{\mbox{days between (reference\_timestamp - 90days) and (reference\_timestamp)}\}
```

For example if we have

| account_id | reference_timestamp     |
|------------|-------------------------|
| ac_1       | 2017-03-31 23:59:59.999 |
| ac_2       | 2017-04-15 23:59:59.999 |

the function should return:
* the average value of the balance observed each day between 
  `2016-12-31 23:59:59.999` and `2017-03-31 23:59:59.999` for `ac_1`
* the average value of the balance observed each day between
  `2017-01-15 23:59:59.999` and `2017-04-15 23:59:59.999` for `ac_2`.

Multiple factors contributes to the overall difficulty of the challenge, such as

1. For each account the balance is known only at the `creation_timestamp`, so
   the balance at other days have to be computed using the transactions.
2. If a day has more than one transaction, one has to decide at which time 
   compute the balance for that date.
3. Some accounts do not have transactions for a long period of time, and this
   should be reflected in the average booked balance result.
4. `creation_timestamp` can be either before or after the `reference_timestamp`.
5. Different accounts can have different `reference_timestemp` values. 


## How to

This repo contains all the material required to run the challenge. Specifically

* `challenge/average_booked_balance.py` contains `average_booked_balance_from`,
  an empty function you are required to implement with this challenge. You can
  code your solution in any way you like as long as you maintain the 
  `average_booked_balance_from` name and signature.
* `tests/fixtures` contains the datasets discussed in the main body of the 
   challenge. The file 
   `account_booked_balance_mean_3mo_reference_timestamps.csv` 
   also contains the value of the average booked balance for each account 
   computed using our proprietary algorithm for reference and testing purposes
* `tests/test_average_booked_balance.py` contains a unit test that verifies
   that the results obtained by your function match with the ones computed
   with our proprietary one. To run the test you need to have `pytest` 
   installed, and then run on a shell
   ```shell
   pytest
   ```
* `requirements.txt` contains the libraries to install to run this challenge.
  The list shipped with the repository is not exhaustive, and you can add
  other libraries if required.

## Evaluation criteria

With this challenge we want to understand how you write code to address a complex 
problem with limited time available. Specifically we want to understand

* how you deal with a problem that has a multitude of edge cases;
* how close your solution is to production-ready code.  

Be aware that one entire team went through multiple iterations to build
the function that we use today to address this challenge. As such, we don't
require that you replicate our same exact results in 
`tests/test_average_booked_balance.py` verbatim. We will consider the challenge
a success if you understand the problem nuances, and then propose a solution
that you believe to be robust enough both in terms of logic and Python 
language.
