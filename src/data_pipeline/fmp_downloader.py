# src/data_pipeline/fmp_downloader.py

import os
import io 
import logging
import csv
import time 
import requests
import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .. import config 

FMP_API_KEY=config.FMP_API_KEY
BASE_URL_V3 = config.BASE_URL_V3
BASE_URL_V4 = config.BASE_URL_V4
BASE_URL_STABLE = config.BASE_URL_STABLE
DEFAULT_TIMEOUT = config.DEFAULT_TIMEOUT if hasattr(config, 'DEFAULT_TIMEOUT') else 10
FORCE_REDOWNLOAD = config.FORCE_REDOWNLOAD


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FMP_Downloader:
    """A robust class to download and manage financial data from FMP."""

    def __init__(
            self, 
            api_key: str
            ):
        if not api_key:
            raise ValueError("FMP_API_KEY is required.")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': "MPhil-Project/1.0"})
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.raw_data_path = self.project_root / "data" / "raw"
        self.interim_data_path = self.project_root / "data" / "interim"
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.interim_data_path.mkdir(parents=True, exist_ok=True)

    def __enter__( self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    @retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True)
    def fetch_api(
        self,
        base_url: str = BASE_URL_V3,
        path: str = "",
        query_vars: Dict[str, Any] = None
        ) -> Any:
        """ A robust, single-call function for the FMP API with retries and logging. """
        if query_vars is None:
            query_vars = dict()
        query_vars['apikey'] = FMP_API_KEY 
        return_var = None
        try:
            full_url = f"{base_url}{path}"
            response = requests.get(full_url, params=query_vars, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            if not response.content:
                logging.warning(f"Response from {full_url} is empty.")
                return []
            if query_vars.get('datatype') == 'csv':
                content = response.content.decode("utf-8")
                reader = csv.DictReader(io.StringIO(content))
                return_var = [row for row in reader]
            else:
                return_var = response.json()
        except requests.exceptions.Timeout:
            logging.error(f"Connection to {full_url} timed out.")
            raise 
        except requests.exceptions.ConnectionError:
            logging.error(f"Connection to {full_url} failed.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred for {full_url}: {e}")
            raise 
        return return_var
    
    def _process_raw_market_data(
            self, 
            raw_df: pd.DataFrame
            ) -> pd.DataFrame:
        """
        Processes a raw market data DataFrame from FMP for a single stock.
        
        This function calculates fully adjusted OHLC prices, selects the
        necessary columns, and returns a clean, analysis-ready DataFrame.

        Args:
        raw_df (pd.DataFrame): The raw DataFrame from the FMP downloader,
        expected to contain columns: ['date', 'open', 'high', 'low',
        'close', 'adjClose', 'volume', 'unadjustedVolume','change',
        'changePercent', 'vwap', 'label', 'changeOverTime', 'ticker'].

        Returns:
        pd.DataFrame: A clean DataFrame with a standard integer index and columns:
        ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume'].
        All price columns are fully adjusted for splits and dividends.
        """
        df = raw_df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['adj_ratio'] = np.where(df['close'] == 0, 1, df['adjClose'] / df['close'])
        df['open'] = df['open'] * df['adj_ratio']
        df['high'] = df['high'] * df['adj_ratio']
        df['low'] = df['low'] * df['adj_ratio']
        df['close'] = df['adjClose'] 
        final_df = df[[
            'date',
            'ticker',
            'open',
            'high',
            'low',
            'close',
            'volume' 
            ]].copy()
        # Display Standard: 2 decimal places; Calculation Standard: 4-6 decimal places 
        return final_df.round(4)
    
    def _get_sp500_history(self, force_redownload: bool = FORCE_REDOWNLOAD) -> pd.DataFrame:
        """ Check locally first. Then, Fetch from FMP. Saves the full history of S&P 500 historical constituent changes under raw path. """
        file_dir = self.raw_data_path / 'constituent'
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / 'sp500_constituent_history.csv'
        if file_path.exists() and not force_redownload: 
            logging.info(f"Loading existing data from {file_path}")
            data = pd.read_csv(file_path)
        else: 
            data = self.fetch_api(path='historical/sp500_constituent')
            if not data:
                logging.error("No data returned from FMP API for S&P 500 historical constituents.")
                return pd.DataFrame()
            data = pd.DataFrame(data)
            data.to_csv(file_path, index=False)
            logging.info(f"Data saved to {file_path}")
        return data     
    

    def _get_sp500_current(self, force_redownload: bool = FORCE_REDOWNLOAD) -> pd.DataFrame:
        """ Check locally first. Then, Fetch from FMP. Saves the full current of S&P 500 current constituents under raw path. """
        file_dir = self.raw_data_path / 'constituent'
        file_dir.mkdir(parents=True, exist_ok=True)
        file_path = file_dir / 'sp500_constituent_current.csv'
        if file_path.exists() and not force_redownload: 
            logging.info(f"Loading existing data from {file_path}")
            data = pd.read_csv(file_path)
        else: 
            data = self.fetch_api(path='sp500_constituent')
            if not data:
                logging.error("No data returned from FMP API for S&P 500 current constituents.")
                return pd.DataFrame()
            data = pd.DataFrame(data)
            data.to_csv(file_path, index=False)
            logging.info(f"Data saved to {file_path}")
        return data  
    
    def _bulk_profile_to_df(self, bulk_profile: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert a list of company profile dictionaries to a DataFrame.
        Please note that lookahead bias might be introduced if the profile data is used for analysis, 
        as it's data on present day, instead of point-in-time.
        """
        if not bulk_profile:
            logging.warning("No profile data provided.")
            return pd.DataFrame()
        if not isinstance(bulk_profile, list):
            raise TypeError("bulk_profile must be a list of dictionaries.")
        bulk_profile = self.fetch_api(
            path=f'profile/{",".join(bulk_profile)}'
            )
        profile_df = pd.DataFrame(bulk_profile)
        profile_df = profile_df[['symbol', 'companyName', 'industry', 'sector', 'exchange', 'exchangeShortName', 'cusip', 
                             'website', 'description', 'ceo', 'cik', 'ipoDate', 'address', 'city', 'state', 'zip']].copy()
        profile_df.rename(columns={'symbol': 'ticker'}, inplace=True)
        return profile_df 
    
    def get_sp500_on_date(self, target_date: str) -> pd.DataFrame:
        """ Get S&P 500 tickers on a specific date. """
        target_date = pd.to_datetime(target_date)
        sp_current = self._get_sp500_current()
        if sp_current.empty:    
            logging.error("No current S&P 500 data available.")
            return set()
        constituents = set(sp_current['symbol'].dropna())
        sp_history = self._get_sp500_history()
        if sp_history.empty:
            logging.error("No historical S&P 500 data available.")
            return set()    
        sp_history['date'] = pd.to_datetime(sp_history['date'])
        sp_history.sort_values(by=['date'], inplace=True, ascending=False)
        for _, row in sp_history.iterrows():
            change_date = row['date']
            if change_date <= target_date:
                break
            added_ticker = row['symbol']
            removed_ticker = row['removedTicker']
            if added_ticker and pd.notna(added_ticker):
                constituents.discard(added_ticker)
            if removed_ticker and pd.notna(removed_ticker):
                constituents.add(removed_ticker)
        constituents = self._bulk_profile_to_df(list(constituents))
        return constituents
    
    def _select_portfolio_by_sector_helper(self, 
                                           df: pd.DataFrame, 
                                           portfolio_size: int = 20) -> list[str]:
        """
        A helper method to select a portfolio of a given size from a DataFrame of candidates,
        ensuring sector diversification based on proportional allocation.

        Args:
        df (pd.DataFrame): DataFrame with tickers as index and columns ['sector', 'avg_trade_capital'].
        portfolio_size (int): The desired number of stocks in the portfolio.

        Returns:
        list[str]: A list of the selected ticker symbols.
        """
        print("\n--- Starting Sector-Balanced Stock Selection ---")
        sector_counts = df['sector'].value_counts()
        sector_proportions = sector_counts / len(df)
        ideal_alloc = sector_proportions * portfolio_size
        sector_allocation = ideal_alloc.apply(math.floor).astype(int)
        remainder = portfolio_size - sector_allocation.sum()
        if remainder > 0:
            largest_remainders = (ideal_alloc - sector_allocation).nlargest(remainder).index
            sector_allocation[largest_remainders] += 1
        print("\n--- Final Proportional Sector Allocation ---")
        print(sector_allocation[sector_allocation > 0])
        final_selection: list[str] = []
        df_sorted = df.sort_values(by='avg_trade_capital', ascending=False)
        for sector, count in sector_allocation.items():
            if count > 0:
                sector_stocks = df_sorted[df_sorted['sector'] == sector]
                top_stocks = sector_stocks.head(count)
                final_selection.extend(top_stocks.index.tolist())
        print("\n--- Final Selected Portfolio ---")
        print(f"Total tickers: {len(final_selection)}")
        print(final_selection)
        return final_selection
    
    def _select_portfolio_by_top_n_plus_diversification_fill_helper(self, 
                                                                    df: pd.DataFrame, 
                                                                    top_n: int = 16,
                                                                    portfolio_size: int = 20) -> list[str]:
                                     
        """
        A helper method to select a portfolio of a given size from a DataFrame of candidates,
        prioritizing the top N by average trading capital and filling the rest with sector diversification.

        Args:
        df (pd.DataFrame): DataFrame with tickers as index and columns ['sector', 'avg_trade_capital'].
        top_n (int): The number of top stocks by average trading capital to prioritize.
        portfolio_size (int): The desired number of stocks in the portfolio.

        Returns:
        list[str]: A list of the selected ticker symbols.
        """
        print("\n--- Starting top-N-plus-diversification-fill potfolio selection ---")
        ranked_df = df.sort_values(by='avg_trade_capital', ascending=False)
        top_stocks = ranked_df.head(top_n)
        remaining_slots = portfolio_size - len(top_stocks)
        final_portfolio = top_stocks.copy()
        if remaining_slots==0:
            print(f"\n--- Selected top {portfolio_size} stocks by average trading capital ---")
            print(final_portfolio.index.tolist())
            return final_portfolio.index.tolist()
        print(f"\n--- Filling remaining {remaining_slots} slots for diversification ---")
        remaining_df = ranked_df.drop(top_stocks.index)
        represented_sectors = set(top_stocks['sector'])
        for ticker, row in remaining_df.iterrows():
            if len(final_portfolio)==portfolio_size:
                break
            if row['sector'] not in represented_sectors:
                final_portfolio = pd.concat([final_portfolio, row.to_frame().T])
                represented_sectors.add(row['sector'])
                print(f"Added {ticker} to represent '{row['sector']}' sector.")
        slots_to_fill_after_diversification = portfolio_size - len(final_portfolio)
        if slots_to_fill_after_diversification > 0:
            best_of_the_rest = remaining_df[~remaining_df.index.isin(final_portfolio.index)].head(slots_to_fill_after_diversification)
            final_portfolio = pd.concat([final_portfolio, best_of_the_rest])
            print(f"\nFilled remaining slots with best-of-the-rest: {best_of_the_rest.index.tolist()}")
        final_selection = final_portfolio.index.tolist()
        print("\n--- Final Selected Portfolio ---")
        print(f"Total tickers: {len(final_selection)}. Including: ")
        print(final_selection)
        return final_selection

    
    def _chunk_tickers(self, tickers: List[str], chunk_size: int = 5) -> List[List[str]]:
        """ A helper method to split a list into smaller chunks. """
        return [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    def get_market_data(
        self,
        tickers: List[str],
        start: str,
        end: str
    ) -> pd.DataFrame:
        """ Fetches historical market data in parallel by chunking tickers to respect the API's 5-ticker limit. """
        print(f"Preparing to download market data for {len(tickers)} tickers...")
        ticker_chunks = self._chunk_tickers(tickers, chunk_size=5)
        yearly_periods = pd.date_range(start=start, end=end, freq='YS')
        if pd.to_datetime(start) not in yearly_periods:
            yearly_periods = yearly_periods.insert(0, pd.to_datetime(start))
        print(f'Total periods: {len(yearly_periods)}')
        tasks = []
        for chunk in ticker_chunks:
            ticker_str = ",".join(chunk)
            for i in range(len(yearly_periods)):
                period_start = yearly_periods[i]
                if i + 1 < len(yearly_periods):
                    period_end = yearly_periods[i+1] - pd.DateOffset(days=1)
                else:
                    period_end = pd.to_datetime(end)
                final_period_end = min(period_end, pd.to_datetime(end))
                tasks.append({
                    'path': f'historical-price-full/{ticker_str}',
                    'query_vars': {
                    'from': period_start.strftime('%Y-%m-%d'),
                    'to': final_period_end.strftime('%Y-%m-%d')
                    }
                    })
        print(f'Total tasks: {len(tasks)}')
        all_results = []
        cpu_count = os.cpu_count()
        # Set max_workers to 10 times the CPU count for I/O bound tasks. 
        max_workers = cpu_count * 10 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.fetch_api, BASE_URL_V3, task['path'], task['query_vars']): task 
                for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Fetching market data chunks"):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                except Exception as e:
                    task = future_to_task[future]
                    logging.error(f"Chunk failed after all retries for path {task['path']}: {e}")
        if not all_results:
            logging.warning("No market data was downloaded.")
            return pd.DataFrame()
        processed_dfs = []
        for result_chunk in all_results:
            if 'historicalStockList' in result_chunk:
                for stock_data in result_chunk['historicalStockList']:
                    raw_df = pd.DataFrame(stock_data['historical'])
                    raw_df['ticker'] = stock_data['symbol']
                    processed_df = self._process_raw_market_data(raw_df)
                    processed_dfs.append(processed_df)
        final_df = pd.concat(processed_dfs, ignore_index=True)
        rows_with_nan_volume = final_df[final_df['volume'].isna()]
        print('\n✅ Data Diagnosis:')
        print(rows_with_nan_volume)
        print(f"\n✅ Successfully processed and combined data for {len(final_df['ticker'].unique())} tickers.")
        final_df.sort_index(inplace=True)
        return final_df
    
    def select_portfolio_by_sector(self, 
                                   start_date_str: str, 
                                   portfolio_size: int = 20, 
                                   reference_week: int = 52, 
                                   force_redownload: bool = FORCE_REDOWNLOAD
    )-> pd.DataFrame:
         """
         Selects a portfolio of a given size, referencing historical weeks from a given day,
         ensuring sector diversification.

         Args:
         start_date_str (str): The desired date from which to build the portfolio. 
         portfolio_size (int): The desired number of stocks in the portfolio.
         reference_weeks (int): The number of weeks to calculate average daily trading capital for each ticker to decide liquidity for selection.

         Returns:
         df (pd.DataFrame): The selected tickers general information, including sectors, companyName, cusip, etc.  
         """
         portfolio_file_dir = self.interim_data_path / 'portfolio'
         portfolio_file_dir.mkdir(parents=True, exist_ok=True)
         portfolio_file_path = portfolio_file_dir / f'sp500_{start_date_str}_{portfolio_size}_{reference_week}.csv'
         if portfolio_file_path.exists() and not force_redownload: 
             logging.info(f"Loading existing portfolio constituents from {portfolio_file_path}")
             return pd.read_csv(portfolio_file_path)
         else:
             sp500_constituents_on_date_dir = self.raw_data_path / 'constituent'
             sp500_constituents_on_date_dir.mkdir(parents=True, exist_ok=True)
             sp500_constituents_on_date_path = sp500_constituents_on_date_dir / f'sp500_constituent_{start_date_str}.csv'
             if sp500_constituents_on_date_path.exists() and not force_redownload:
                  logging.info(f"Loading existing sp500 constituents from {sp500_constituents_on_date_path}.")
                  sp500_constituents= pd.read_csv(sp500_constituents_on_date_path)
             else:
                  sp500_constituents = self.get_sp500_on_date(start_date_str)
                  logging.info(f"Saving sp500 constituents to {sp500_constituents_on_date_path}.")
                  sp500_constituents.to_csv(sp500_constituents_on_date_path, index=False)
             date_weeks_ago = pd.to_datetime(start_date_str) - pd.DateOffset(weeks=reference_week)
             date_weeks_ago_str = date_weeks_ago.strftime('%Y-%m-%d')
             market_data_on_date_path = sp500_constituents_on_date_dir / f'sp500_market_{start_date_str}_{reference_week}.parquet'
             if market_data_on_date_path.exists() and not force_redownload:
                  logging.info(f"Loading existing market data from {market_data_on_date_path}.")
                  market_data = pd.read_parquet(market_data_on_date_path)
             else:
                  market_data = self.get_market_data(tickers=list(sp500_constituents['ticker']),
                                                     start=date_weeks_ago_str,
                                                     end=start_date_str)
                  logging.info(f"Saving market data to {market_data_on_date_path}.")
                  market_data.to_parquet(market_data_on_date_path, index=False)
             market_data['trade_capital'] = market_data['close'] * market_data['volume']
             avg_trading_capital = market_data.groupby('ticker', as_index=True)['trade_capital'].mean()
             sector_data = sp500_constituents[['ticker', 'sector']].copy()
             sector_data.set_index('ticker', inplace=True)
             df_combined = sector_data.join(avg_trading_capital)
             df_combined.rename(columns={'trade_capital': 'avg_trade_capital'}, inplace=True)
             ## selected_tickers = self._select_portfolio_by_sector_helper(df_combined, portfolio_size)
             selected_tickers = self._select_portfolio_by_top_n_plus_diversification_fill_helper(df_combined, 15, portfolio_size)
             selected_tickers_df = sp500_constituents[sp500_constituents['ticker'].isin(selected_tickers)]
             selected_tickers_df.reset_index(drop=True)
             logging.info(f"Saving selected portfolio data to {portfolio_file_path}.")
             selected_tickers_df.to_csv(portfolio_file_path, index=False)
             return selected_tickers_df 
         
    def portfolio_market_data_loader(self,
                                     start_date_str: str = '2025-01-01',
                                     portfolio_size: int = 20, 
                                     reference_week: int = 52, 
                                     warmup_weeks: int = 52, 
                                     end_date_str: str = '2025-08-15',
                                     force_redownload: bool = FORCE_REDOWNLOAD) -> None: 
        """ Check locally first. Then, Fetch from FMP. Saves the full current of S&P 500 current constituents under processed path."""
        file_dir = self.interim_data_path / 'market'
        file_dir.mkdir(parents=True, exist_ok=True)
        warmup_date = pd.to_datetime(start_date_str) - pd.DateOffset(weeks=warmup_weeks)
        warmup_date_str = warmup_date.strftime('%Y-%m-%d')
        file_path = file_dir / f'sp500_{warmup_date_str}_{start_date_str}_{end_date_str}_{portfolio_size}_{reference_week}.parquet'
        if file_path.exists() and not force_redownload:
            logging.info(f"Loading existing data from {file_path}.")
            print(f'\n✅ Loading existing data from {file_path}.')
        else:
            portfolio_universe = self.select_portfolio_by_sector(start_date_str, portfolio_size, reference_week)
            market_data = self.get_market_data(list(portfolio_universe['ticker']), warmup_date_str, end_date_str) 
            if len(market_data) != 0:
                market_data.to_parquet(file_path, index=False)
                logging.info(f"Data saved to {file_path}")
                print(f'\n✅ Data saved to {file_path}.')


    def _fetch_news_paginated(self, tickers_str: str, from_date: str, to_date: str) -> List[Dict]:
        """
        A helper method that handles pagination to fetch all news for a given
        ticker chunk and date range.
        """
        all_articles = []
        page = 0
        while True:
            try:
                params = {
                    'tickers': tickers_str,
                    'from': from_date,
                    'to': to_date,
                    'limit': 1000, # Max limit
                    'page': page
                }
                articles_on_page = self.fetch_api(path="stock_news", query_vars=params)
                if not articles_on_page:
                    break 
                all_articles.extend(articles_on_page)
                page += 1
                time.sleep(0.2) # Small polite delay between pages
            except Exception as e:
                logging.error(f"Failed to fetch page {page} for tickers {tickers_str}: {e}")
                break 
        return all_articles

    def get_news_data(self, tickers: List[str], start_date: str, end_date: str, force_redownload: bool = FORCE_REDOWNLOAD) -> pd.DataFrame:
        """
        Downloads and caches all historical news for a list of tickers and a date range.
        It chunks tickers and time periods for efficient, parallel downloading.
        """
        cache_path = self.raw_data_path / "news" / f"sp500_{start_date}_to_{end_date}_{len(tickers)}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not force_redownload:
            logging.info(f"Loading news data from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        logging.info("Starting historical news download...")
        # --- Task Generation ---
        ticker_chunks = self._chunk_tickers(tickers, chunk_size=5)
        date_ranges = pd.date_range(start=start_date, end=end_date, freq='AS') # Annual Start frequency
        start_ts = pd.to_datetime(start_date) - pd.DateOffset(weeks=1)  # Start one week before to include the first day
        end_ts = pd.to_datetime(end_date)           
        date_ranges = pd.DatetimeIndex([start_ts]).append(date_ranges).append(pd.DatetimeIndex([end_ts])) # Ensure start and end dates are included
        date_ranges = date_ranges.drop_duplicates()
        tasks = []
        for chunk in ticker_chunks:
            for i in range(len(date_ranges)-1):
                period_start = date_ranges[i]
                if i==len(date_ranges) - 2:
                    period_end = end_ts
                else:
                    period_end = date_ranges[i+1] - pd.DateOffset(days=1)
                tasks.append({
                    'tickers_str': ",".join(chunk),
                    'from_date': period_start.strftime('%Y-%m-%d'),
                    'to_date': period_end.strftime('%Y-%m-%d')
                })
        # --- Parallel Execution ---
        all_results = []
        cpu_count = os.cpu_count()
        # Set max_workers to 10 times the CPU count for I/O bound tasks. 
        max_workers = cpu_count * 10 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._fetch_news_paginated, task['tickers_str'], task['from_date'], task['to_date']): task
                for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Fetching news"):
                try:
                    result = future.result()
                    if result:
                        all_results.extend(result)
                except Exception as e:
                    logging.error(f"Task failed after all retries: {e}")
        if not all_results:
            logging.warning("No news data was downloaded.")
            return pd.DataFrame()
        # --- Processing and Saving ---
        df = pd.DataFrame(all_results)
        df.rename(columns={'symbol': 'ticker', 'publishedDate': 'date', 'site': 'source', 'title': 'headline', 'text': 'content'}, inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None).dt.date 
        # Filter again to the exact date range and select final columns
        final_df = df[
            (df['date'] >= start_ts.date()) & 
            (df['date'] <= end_ts.date())][['date', 'ticker', 'headline', 'content', 'source']].copy()
        final_df.drop_duplicates(inplace=True)
        final_df.sort_values(by=['date', 'ticker'], inplace=True)
        logging.info(f"Downloaded a total of {len(final_df)} news articles.")
        df.dropna(inplace=True)
        df.to_parquet(cache_path, index=False)
        logging.info(f"News data saved to {cache_path}")
        return final_df
    
    def get_earnings_transcripts(
            self,
            tickers: list[str],
            start_date: str,
            end_date: str,
            force_redownload: bool = False
            ) -> pd.DataFrame:
        """
        Downloads and caches all earnings call transcripts for a list of tickers
        over a specified date range.
        """
        cache_path = self.raw_data_path / "fundamentals" / f"sp500_{start_date}_to_{end_date}_{len(tickers)}.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not force_redownload:
            logging.info(f"Loading earnings transcripts from cache: {cache_path}")
            return pd.read_parquet(cache_path)
        logging.info("Starting historical earnings transcript download...")
        period_start = pd.to_datetime(start_date) - pd.DateOffset(months=3)
        quarterly_periods = pd.date_range(start=period_start, end=end_date, freq='Q')
        tasks = []
        for ticker in tickers:
            for period in quarterly_periods:
                tasks.append({
                    'path': 'earning-call-transcript',
                    'query_vars': {
                        'symbol': ticker,
                        'year': period.year,
                        'quarter': period.quarter
                    }
                })
        all_transcripts = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {
                executor.submit(self.fetch_api, BASE_URL_STABLE, task['path'], task['query_vars']): task 
                for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Fetching transcripts"):
                try:
                    result = future.result()
                    # The API returns a list, usually with one transcript
                    if result:
                        transcript_data = result[0]
                        all_transcripts.append({
                            'ticker': transcript_data['symbol'],
                            'quarter': transcript_data['period'],
                            'year': transcript_data['year'],
                            'date': transcript_data['date'],
                            'content': transcript_data['content']
                        })
                except Exception as e:
                    # It's common for a company to not have a transcript for a given quarter, so we log as info
                    task_info = future_to_task[future]['query_vars']
                    logging.info(f"Could not fetch transcript for {task_info['symbol']} Q{task_info['quarter']} {task_info['year']}. It may not exist.")
        if not all_transcripts:
            logging.warning("No earnings transcripts were downloaded.")
            return pd.DataFrame()
        df = pd.DataFrame(all_transcripts)
        df.sort_values(by=['date', 'ticker'], inplace=True)
        df.drop_duplicates(subset=['ticker', 'quarter', 'year'], inplace=True)
        df.to_parquet(cache_path, index=False)
        logging.info(f"✅ Earnings transcripts saved to {cache_path}")
        return df
        
    def get_treasury_rates(
            self, 
            start: str, 
            end: str
            ) -> pd.DataFrame:
        treasury_rates_data = self.fetch_api(
            base_url=BASE_URL_STABLE,
            path='treasury-rates',
            query_vars={'from': start, 'to': end
                        })
        if len(treasury_rates_data) == 0:
            logging.info('No treasury rates data returned.')
            return pd.DataFrame()
        cache_path = self.raw_data_path / 'macro' / f'treasury_rates_{start}_{end}.csv'
        treasury_rates_df = pd.DataFrame(treasury_rates_data)
        treasury_rates_df.to_csv(cache_path, index=False)
        logging.info(f'✅ Treasury rates data saved to {cache_path}')
        return treasury_rates_df 
    
    def get_realGDP(
            self, 
            start: str, 
            end: str
            ) -> pd.DataFrame:
        realGDP = self.fetch_api(
            base_url=BASE_URL_STABLE,
            path='economic-indicators',
            query_vars={'name': 'realGDP', 'from': start, 'to': end})
        if len(realGDP) == 0:
            logging.info('No realGDP data returned.')
            return pd.DataFrame()
        cache_path = self.raw_data_path / 'macro' / f'realGDP_{start}_{end}.csv'
        readlGDP_df = pd.DataFrame(realGDP)
        readlGDP_df.to_csv(cache_path, index=False)
        logging.info(f'✅ Treasury rates data saved to {cache_path}')
        return readlGDP_df
    
    def get_economic_indicators(
            self, 
            name: str,
            start: str, 
            end: str
            ) -> pd.DataFrame:
        macro_data = self.fetch_api(
            base_url=BASE_URL_STABLE,
            path='economic-indicators',
            query_vars={'name': name, 'from': start, 'to': end})
        if len(macro_data) == 0:
            logging.info(f'No {name} data returned.')
            return pd.DataFrame()
        cache_path = self.raw_data_path / 'macro' / f'{name}_{start}_{end}.csv'
        macro_df = pd.DataFrame(macro_data)
        macro_df.to_csv(cache_path, index=False)
        logging.info(f'✅ {name} data saved to {cache_path}')
        return macro_df

    def get_batch_statements(
            self, 
            tickers: list[str],
            path: str = 'ratios',
            period: str = 'quarter', 
            limit: int = 50) -> pd.DataFrame:
        """ Fetches fundamental ratios for a list of tickers in parallel."""
        tasks = [{'symbol': ticker, 'period': period, 'limit': limit} for ticker in tickers]
        all_ratios = []
        cpu_count = os.cpu_count() 
        max_workers = cpu_count * 10 
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_api, BASE_URL_STABLE, path=path, query_vars=task): task for task in tasks}
            for future in tqdm(as_completed(future_to_ticker), total=len(tasks), desc=f"Fetching {path}"):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        df = pd.DataFrame(result)
                        df['ticker'] = ticker
                        all_ratios.append(df)
                except Exception as e:
                    print(f"Failed to fetch {path} for {ticker}: {e}")
        if not all_ratios:
            return pd.DataFrame()
        df_all = pd.concat(all_ratios, ignore_index=True)
        df_all.to_csv(self.raw_data_path / 'fundamentals' / f'{path}.csv')
        return df_all 
 


if __name__ == "__main__":
    print(f"--- Running Market Data Loader ---")
    start_date_str = '2019-05-01'
    end_date_str = '2025-08-15'
    portfolio_size = 20
    reference_week = 52
    warmup_weeks = 52

    with FMP_Downloader(FMP_API_KEY) as downloader:
        downloader.portfolio_market_data_loader(
        start_date_str=start_date_str,
        portfolio_size=portfolio_size,
        reference_week=reference_week, 
        warmup_weeks=warmup_weeks,
        end_date_str=end_date_str)