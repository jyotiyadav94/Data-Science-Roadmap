import pandas



def days_diff(date1, date2):
    """
    Calculates the difference in days between two dates.

    Args:
        date1 (datetime): The first date.
        date2 (datetime): The second date.

    Returns:
        int: The difference in days between the two dates.
    """
    return (date2 - date1).days




def merge_daily_transactions(df):
    """
    Groups transactions in a DataFrame by account and date, and returns the sum of amounts.

    Args:
        df (pandas.DataFrame): A DataFrame containing transaction data.

    Returns:
        pandas.DataFrame: A DataFrame with the total amount for each account and date.
    """
    # Convert value_timestamp to datetime object.
    df['value_timestamp'] = pandas.to_datetime(df['value_timestamp'])
    # Group transactions by account_id and value_timestamp.
    return df.groupby(['account_id', pandas.Grouper(key='value_timestamp', freq='D')])['amount'].sum().reset_index()





def calculate_avg(df_t, balance_at_creation, account_creation_date, date, end_date):

    """
    Calculates the average booked balance value for a given period of time.

    Args:
        df_t (pandas.DataFrame): A DataFrame containing transaction data.
        balance_at_creation (float): The account balance at creation.
        account_creation_date (datetime): The date the account was created.
        date (datetime): The start date for the period.
        end_date (datetime): The end date for the period.

    Returns:
        pandas.Series: A Series containing the average booked balance value.
    """

    df = df_t.copy()
    df.reset_index(inplace=True, drop=True)
    breakage_index = 0
    total_transaction_sum = 0
    prev_transactionAmt = 0
    days_beforeTransactionFound = 0
    account_balance = balance_at_creation

    # Remove the transactions before account creation date.
    for i in range(df.shape[0]):
        if df['value_timestamp'].iloc[i] < account_creation_date:
            index_afterAccountCreation = i
            df = df[index_afterAccountCreation+1:]
            break

    # Calculate the net transactions before we start calculating the average.
    for i in range(df.shape[0]):
        if df['value_timestamp'].iloc[i] > date:
            breakage_index = i
            days_beforeTransactionFound = days_diff(date, df['value_timestamp'].iloc[i])+1
            break
        prev_transactionAmt += df['amount'].iloc[i]
    account_balance += prev_transactionAmt
    total_transaction_sum = (account_balance)*days_beforeTransactionFound

    df = df[breakage_index:].reset_index(drop=True)

    for i in range(df.shape[0]-1):
        # Once the date exceeds end_date, we reject the next transactions.
        if df['value_timestamp'].iloc[i+1] >= end_date:
            temp_date = df['value_timestamp'].iloc[i]
            temp_date_next = df['value_timestamp'].iloc[i+1]
            temp_amount = df['amount'].iloc[i]
            account_balance = account_balance+temp_amount
            total_transaction_sum += days_diff(temp_date, end_date)*account_balance
            break
        temp_date = df['value_timestamp'].iloc[i]
        temp_date_next = df['value_timestamp'].iloc[i+1]
        temp_amount = df['amount'].iloc[i]
        account_balance = account_balance+temp_amount
        total_transaction_sum += days_diff(temp_date, temp_date_next)*account_balance
    net_days = days_diff(date, end_date)
    return total_transaction_sum/net_days





def average_booked_balance_from(transactions: pandas.DataFrame,
                                accounts: pandas.DataFrame,
                                reference_timestamps: pandas.DataFrame) -> pandas.Series:
    """
    Computes the average booked balance for each account at the specified reference timestamps.

    :param transactions: pandas dataframe containing the transactions from a collection of accounts
    :param accounts: pandas dataframe containing a collection of accounts together with their balance when they
        were first added to our systems.
    :param reference_timestamps: pandas dataframe with the timestamp at which to compute the average booked balance for
        each account. Different accounts might have different reference timestamps.
    :return: a pandas series where the index is a multindex containing the reference timestamp and the account id, and the
        values are the average booked balances.
    """
    reference_timestamps['reference_timestamp'] = pandas.to_datetime(reference_timestamps['reference_timestamp'])
    accounts['creation_timestamp'] = pandas.to_datetime(accounts['creation_timestamp'])

    # merge accounts and reference datasets
    account_creation_reference = pandas.merge(accounts, reference_timestamps, on='account_id', how='inner')

    transactions['value_timestamp'] = pandas.to_datetime(transactions['value_timestamp'])
    average_booked_balance_value=[]
    transactions = merge_daily_transactions(transactions)
    for i in range(account_creation_reference.shape[0]):
      account_id = account_creation_reference['account_id'].iloc[i]
      creation_ts = account_creation_reference['creation_timestamp'].iloc[i]
      balance_at_creation = account_creation_reference['balance_at_creation'].iloc[i]
      reference_timestamp = account_creation_reference['reference_timestamp'].iloc[i]
      days_before = pandas.Timedelta(days=90)
      begin_reference_timestamp = reference_timestamp-days_before
      test_trans = transactions[transactions['account_id'] == account_id]
      result=calculate_avg(test_trans, balance_at_creation, creation_ts, begin_reference_timestamp, reference_timestamp)
      average_booked_balance_value.append(result)
    return pandas.Series(average_booked_balance_value)

