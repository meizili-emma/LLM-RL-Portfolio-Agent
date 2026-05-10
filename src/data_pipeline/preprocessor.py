
import pandas as pd 
from typing import Type
from stockstats import StockDataFrame as Sdf
import numpy as np 
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from src.config import TECHNICAL_INDICATOR


class GroupByScaler(BaseEstimator, TransformerMixin):
    """Sklearn-like scaler that scales considering groups of data.

    In the financial setting, this scale can be used to normalize a DataFrame
    with time series of multiple tickers. The scaler will fit and transform
    data for each ticker independently.
    """

    def __init__(
            self, 
            by: str = 'date', 
            scaler: Type[BaseEstimator] = RobustScaler, 
            columns: list = None, 
            scaler_kwargs: dict = None):
        """Initializes GoupBy scaler.

        Args:
            by: Name of column that will be used to group.
            scaler: Scikit-learn scaler class to be used.
            columns: List of columns that will be scaled.
            scaler_kwargs: Keyword arguments for chosen scaler.
        """
        self.scalers = {}  # dictionary with scalers
        self.by = by
        self.scaler = scaler
        self.columns = columns
        self.scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs

    def fit(self, X, y=None):
        # if columns aren't specified, considered all numeric columns
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object", "datetime64[ns]"]).columns
        # fit one scaler for each ticker 
        for value in X[self.by].unique():
            X_group = X.loc[X[self.by] == value, self.columns]
            self.scalers[value] = self.scaler(**self.scaler_kwargs).fit(X_group)
        return self

    def transform(self, X, y=None):
        # apply scaler for each ticker 
        X = X.copy()
        for value in X[self.by].unique():
            select_mask = X[self.by] == value
            X.loc[select_mask, self.columns] = self.scalers[value].transform(
                X.loc[select_mask, self.columns]
            )
        return X


def _rolling_scale_column(column: pd.Series, window: int) -> pd.Series:
    """Helper function to apply rolling normalization to a single column."""
    rolling_median = column.rolling(window=window, min_periods=window).median()
    rolling_q25 = column.rolling(window=window, min_periods=window).quantile(0.25)
    rolling_q75 = column.rolling(window=window, min_periods=window).quantile(0.75)
    rolling_iqr = rolling_q75 - rolling_q25
    # Scale the column. Replace infinite values resulting from zero IQR with NaN.
    scaled_column = (column - rolling_median) / rolling_iqr
    return scaled_column.replace([np.inf, -np.inf], np.nan)


def rolling_window_scaler(
    df: pd.DataFrame, 
    window: int = 8, 
    columns_to_exclude: list[str] = None
) -> pd.DataFrame:
    """
    Performs rolling window normalization on the numerical columns of a DataFrame,
    grouped by ticker.

    Args:
        df (pd.DataFrame): Input DataFrame with a ['date', 'ticker'] multi-index.
        window (int): The size of the rolling window (e.g., 8 weeks).
        columns_to_exclude (List[str]): List of columns to not normalize.

    Returns:
        pd.DataFrame: The normalized DataFrame.
    """
    if columns_to_exclude is None:
        columns_to_exclude = []
    df_normalized = df.copy()
    df_normalized = df_normalized.sort_values(['ticker', 'date'])
    numeric_cols = df.select_dtypes(include=np.number).columns
    cols_to_scale = [col for col in numeric_cols if col not in columns_to_exclude]
    df_normalized[cols_to_scale] = df_normalized.groupby('ticker')[cols_to_scale].transform(
        lambda x: _rolling_scale_column(x, window)
    )
    df_normalized.dropna(inplace=True)
    df_normalized = df_normalized.sort_values(['date', 'ticker'], ignore_index=True)
    df_normalized.index = df_normalized['date'].factorize()[0]
    return df_normalized


class FeatureEngineer:

    def __init__(
            self,
            use_technical_indicators: bool = True,
            technical_indicators: list = TECHNICAL_INDICATOR,
            use_fundamental_indicators: bool = False,
            use_news_indicators: bool = False,
            use_turbulence: bool = False):
        self.use_technical_indicators = use_technical_indicators
        self.use_fundamental_indicators = use_fundamental_indicators
        self.use_news_indicators = use_news_indicators
        self.use_turbulence = use_turbulence
        self.technical_indicators = technical_indicators 

    def preprocess_data(
            self, 
            market_df: pd.DataFrame,
            start: str, 
            end: str
    ) -> pd.DataFrame: 
        """
         Args:
         market_df (pd.DataFrame): Daily based OHLCV data, along with date and ticker.
         start (str): The start date for the whole dataset. 
         end (str): The end date for the whole dataset. 
         """
        market_df = self.clean_data(market_df)
        if self.use_technical_indicators: 
            tech_df = self.add_technical_indicator(market_df)
            print("Successfully added technical indicators")
        if self.use_turbulence:
            turbulence_df = self.add_turbulence(market_df)
            print("Successfully added turbulence index")
        df = self.resample_data(market_df)
        df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]
        df = pd.merge(df, tech_df, on=['date', 'ticker'], how='left')
        df = pd.merge(df, turbulence_df, on='date')
        df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'macd',
                 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
                 'close_60_sma', 'turbulence']]
        df.index = df['date'].factorize()[0]
        return df 

    def resample_data(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        all_tickers = list(df['ticker'].unique())
        all_weekly_market_data = pd.DataFrame()
        for ticker in tqdm(all_tickers):
            single_market_data = df[df['ticker']==ticker]
            single_market_data = single_market_data.set_index('date').sort_index()
            single_weekly_market_data = single_market_data.resample('W-FRI').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
                })
            single_weekly_market_data['ticker'] = ticker
            all_weekly_market_data = pd.concat([all_weekly_market_data, single_weekly_market_data])
        all_weekly_market_data = all_weekly_market_data.reset_index().sort_values(['date', 'ticker'], ignore_index=True)
        return all_weekly_market_data

    def clean_data(
            self, 
            df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
        merged_closes = df.pivot_table(index='date', columns='ticker', values='close')
        merged_closes = merged_closes.dropna(axis=1, how='all')
        tics = merged_closes.columns
        df = df[df.ticker.isin(tics)]
        return df 
    
    def add_technical_indicator(
            self, 
            data: pd.DataFrame
            ) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(by=["ticker", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.ticker.unique()
        for indicator in self.technical_indicators:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.ticker == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["ticker"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.ticker == unique_ticker[i]]["date"].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                        )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[["ticker", "date", indicator]], on=["ticker", "date"], how="left")
        df = df.sort_values(by=["date", "ticker"])
        df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        return df
    
    def add_turbulence(
            self, 
            data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="ticker", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()
        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min() :].dropna(axis=1)
            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
    