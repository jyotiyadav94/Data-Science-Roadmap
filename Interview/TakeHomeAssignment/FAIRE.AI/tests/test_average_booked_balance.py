import unittest

import pandas
from pandas._testing import assert_series_equal

from challenge.average_booked_balance import average_booked_balance_from


class TestAverageBalance(unittest.TestCase):
    def test_average_balance(self):
        transactions = pandas.read_csv('tests/fixtures/account_booked_balance_mean_3mo_transactions.csv')
        accounts = pandas.read_csv('tests/fixtures/account_booked_balance_mean_3mo_accounts.csv')
        results = pandas.read_csv('tests/fixtures/account_booked_balance_mean_3mo_reference_timestamps.csv',
                                  index_col=['reference_timestamp', 'account_id']).squeeze("columns")
        reference_timestamps = results.index.to_frame(index=False)

        average_booked_balance = average_booked_balance_from(transactions, accounts, reference_timestamps)
        print(average_booked_balance)

        #assert_series_equal(average_booked_balance, results)


if __name__ == '__main__':
    unittest.main()
