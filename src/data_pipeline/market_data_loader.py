# src/data_pipeline/market_data_loader.py

import os
from pathlib import Path
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Tuple

import tempfile
import atexit
import shutil

# --- IMPORTANT: Set Cache Directory BEFORE yfinance import ---
# Use a temporary, isolated cache for each run 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMP_CACHE_BASE_DIR = PROJECT_ROOT / "data" / ".temp_caches"
TEMP_CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
temp_cache_dir = tempfile.mkdtemp(dir=TEMP_CACHE_BASE_DIR)
atexit.register(shutil.rmtree, temp_cache_dir, ignore_errors=True)
os.environ['XDG_CACHE_HOME'] = temp_cache_dir 

# Now, import yfinance. It will use our clean, temporary cache.
import yfinance as yf 
from .. import config 

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "market"
CACHE_FILE = RAW_DATA_DIR / f"market_original_daily_{config.STOCK_POOL_NAME}.parquet"



def diagnose_data_quality(df: pd.DataFrame, missing_threshold: float = 0.03) -> Tuple[int, List[str]]:
    """
    Analyzes the raw DataFrame for missing values and identifies low-quality stocks.

    Args:
        df (pd.DataFrame): The raw DataFrame from yfinance.
        missing_threshold (float): The percentage of missing days above which a stock is flagged.

    Returns:
        Tuple[int, List[str]]: A tuple containing the number of dropped dates and a list of low-quality tickers.
    """
    print("\n--- Running Data Quality Diagnostics ---")
    
    # 1. Overall dropped dates
    initial_rows = len(df)
    clean_df = df.dropna()
    final_rows = len(clean_df)
    dropped_dates_count = initial_rows - final_rows
    print(f"Total trading days downloaded: {initial_rows}")
    print(f"Total clean trading days (all stocks have data): {final_rows}")
    print(f"Number of dates dropped due to missing data in at least one stock: {dropped_dates_count} ({dropped_dates_count/initial_rows:.2%})")

    # 2. Per-stock missing dates
    missing_days_per_stock = df['Close'].isnull().sum()
    low_quality_stocks = set()
    
    print("\n--- Per-Stock Missing Data Report ---")
    for ticker, missing_days in missing_days_per_stock.items():
        if missing_days > 0:
            missing_pct = missing_days / initial_rows
            print(f"Ticker: {ticker}, Missing Days: {missing_days} ({missing_pct:.2%})")
            if missing_pct > missing_threshold:
                low_quality_stocks.add(ticker)

    # Per-stock Zero Price check
    print("\n--- Per-Stock Zero Price Report ---")
    price_cols = ['Open', 'High', 'Low', 'Close']
    stacked_df = df.stack(level=1)
    has_zero_price = (stacked_df[price_cols] == 0).any(axis=1)
    zero_days_per_ticker = has_zero_price.groupby('Ticker').sum()
    tickers_with_zeros = zero_days_per_ticker[zero_days_per_ticker > 0]

    if not tickers_with_zeros.empty:
        for ticker, zero_days in tickers_with_zeros.items():
            print(f"Ticker: {ticker}, Days with at least one Zero Price (O,H,L,C): {zero_days}")
            low_quality_stocks.add(ticker)
    else:
        print("No zero price values found.")

    if low_quality_stocks:
        print(f"\n⚠️ Low-quality stocks to consider removing (>{missing_threshold:.0%} missing data): {low_quality_stocks}")
    else:
        print("\n✅ All stocks passed the data quality threshold.")
        
    return dropped_dates_count, low_quality_stocks


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    reraise=True
)
def fetch_data_from_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """A robust wrapper around yf.download() that lets yfinance handle the session."""
    print(f"Attempting to download data for {len(tickers)} tickers...")
    # Ensure tickers are unique and non-empty
    tickers = list(set(ticker for ticker in tickers if ticker))
    if not tickers:
        raise ValueError("No valid tickers provided.")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, ignore_tz=True)
    if df.empty:
        raise IOError("No data returned from yfinance. Retrying...")
    print("✅ Download successful.")
    return df 

def get_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    stock_pool_name: str, 
    force_redownload: bool = False
) -> pd.DataFrame:
    """Downloads, validates, and caches daily OHLCV data."""
    raw_data_dir = PROJECT_ROOT / "data" / "raw" / "market"
    cache_filepath = raw_data_dir / f"market_original_daily_{stock_pool_name}.parquet"
    cache_filepath.parent.mkdir(parents=True, exist_ok=True)

    if cache_filepath.exists() and not force_redownload:
        print(f"✅ Loading market data from cache: {cache_filepath}")
        return pd.read_parquet(cache_filepath)

    print("ℹ️ No cache found or redownload forced. Fetching data from yfinance...")
    raw_df = fetch_data_from_yfinance(tickers, start_date, end_date)
    
    # --- Run Diagnostics and Clean Data ---
    if raw_df is not None: 
        _, low_quality_tickers = diagnose_data_quality(raw_df)
    
        if low_quality_tickers:
            print(f"\nWarning: The following tickers have high rates of missing data: {low_quality_tickers}. Consider removing them from your config.")

        clean_df = raw_df.dropna()

        print(f"\n💾 Saving cleaned raw market data to Parquet cache: {cache_filepath}")
        clean_df.to_parquet(cache_filepath)
        return clean_df
    else: 
        return None 



if __name__ == "__main__":
    print(f"--- Running Market Data Loader ---")
    market_data = get_market_data(
        tickers=config.ACTIVE_TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        stock_pool_name=config.STOCK_POOL_NAME,
        force_redownload=config.FORCE_DOWNLOAD 
    )
    if market_data is not None: 
        print("\n--- Final Data Info ---")
        market_data.info()
    else: 
        print("\n--- No Data Fetched ---")



