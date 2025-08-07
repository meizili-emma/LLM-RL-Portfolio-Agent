# src/data_pipeline/stock_selector.py

import pandas as pd
import numpy as np 
import requests
from bs4 import BeautifulSoup
import time
from .market_data_loader import fetch_data_from_yfinance

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# URL for a revision of the page from late 2016 to represent the start of 2017
URL_2008 = "https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=259875087" # 24 December 2008
URL_2009 = "https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=329730967" # 4 December 2009 
URL_2011 = "https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=444285954" # 11 August 2011
URL_2013 = "https://en.wikipedia.org/w/index.php?title=S%26P_100&oldid=587014388" # 20 December 2013

SP100_URLS = {
    2009: URL_2008,
    2010: URL_2009,
    2012: URL_2011,
    2014: URL_2013,
    }


def scrape_sp100_constituents(start_year: int) -> list[str]:
    """Scrapes the S&P 100 constituent list for a specific year from Wikipedia."""

    print(f"--- Scraping S&P 100 constituents for start of {start_year} from {SP100_URLS[start_year]} ---")

    headers = {'User-Agent': 'MPhil-PortfolioAgent-Scraper/0.1'}
    try:
        response = requests.get(SP100_URLS[start_year], headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download page: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    constituents_table = soup.find('table', {'class': 'wikitable sortable'})
    if not constituents_table:
        print("⚠️ 'wikitable sortable' class not found. Trying fallback for older page format...")
        constituents_table = soup.find('table', {'class': 'wikitable'})
    if not constituents_table:
        print("❌ Could not find the constituents table on the page with any known selector.")
        return []

    tickers = []
    for row in constituents_table.tbody.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) > 0:
            ticker = cells[0].text.strip()
            if ticker:
                tickers.append(ticker)

    print(f"✅ Found {len(tickers)} tickers in the historical list.")
    return sorted(tickers)


def select_portfolio_universe(start_year: int, constituent_list: list[str], portfolio_size: int = 20) -> list[str]:
    """
    Selects a diversified portfolio based on a dynamic liquidity filter and proportional sector allocation.
    """
    
    # --- 1. Fetch liquidity data for the prior year ---
    liquidity_year = start_year - 1
    print(f"\nFetching liquidity data for {liquidity_year}...")
    liquidity_df = fetch_data_from_yfinance(constituent_list, start=f"{liquidity_year}-01-01", end=f"{liquidity_year}-12-31")
    
    # --- 2. Apply Percentile Liquidity Filter ---
    dollar_volume = liquidity_df['Close'] * liquidity_df['Volume']
    avg_dollar_volume = dollar_volume.mean().dropna()
    liquidity_threshold = avg_dollar_volume.quantile(0.20)
    liquid_tickers = avg_dollar_volume[avg_dollar_volume >= liquidity_threshold].index.tolist()
    print(f"\nFiltered to {len(liquid_tickers)} tickers using a 20th percentile liquidity threshold.")
    
    # --- 3. Get Sector Info and Create Candidate DataFrame ---
    print("\nFetching sector information for liquid tickers...")
    candidates = []
    for ticker in liquid_tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            if sector != 'Unknown':
                candidates.append({
                    'ticker': ticker,
                    'sector': sector,
                    'volume': avg_dollar_volume[ticker]
                })
                print(f"  - {ticker}: {sector}")
            time.sleep(0.5)
        except Exception:
            print(f"Could not get info for {ticker}, skipping.")
    candidates_df = pd.DataFrame(candidates)

    # --- 4. Apply Proportional Sector Allocation ---
    # Calculate the proportion of each sector in the liquid universe
    sector_counts = candidates_df['sector'].value_counts()
    sector_proportions = sector_counts / len(candidates_df)
    # Calculate the ideal, fractional allocation
    ideal_alloc = sector_proportions * portfolio_size
    # Take the floor as the base allocation
    sector_allocation = ideal_alloc.apply(np.floor).astype(int)
    # Calculate how many stocks are still needed
    remainder = portfolio_size - sector_allocation.sum()
    # Distribute the remainder to the sectors with the largest fractional parts
    if remainder > 0:
        largest_remainders = (ideal_alloc - sector_allocation).nlargest(remainder).index
        sector_allocation[largest_remainders] += 1
    
    print("\n--- Proportional Sector Allocation ---\n")
    print(sector_allocation)
    
    final_selection = []
    for sector, count in sector_allocation.items():
        if count > 0:
            sector_stocks = candidates_df[candidates_df['sector'] == sector]
            top_stocks = sector_stocks.sort_values(by='volume', ascending=False).head(count)
            final_selection.extend(top_stocks['ticker'].tolist())
    portfolio_df = candidates_df[candidates_df['ticker'].isin(final_selection)][['ticker', 'sector']]

    spy_row = pd.DataFrame([{'ticker': 'SPY', 'sector': 'Index ETF'}])
    final_portfolio_df = pd.concat([portfolio_df, spy_row], ignore_index=True)

    print("\n--- Final Selected Portfolio ---")
    print(f"Total tickers: {len(final_portfolio_df)}")
    
    return final_portfolio_df


def acquire_portfolio_universe(start_year: int, portfolio_size: int = 20) -> list[str]:
    """Acquires the S&P 100 portfolio universe for a given start year, scraping the constituents and selecting"""

    print("Starting S&P 100 portfolio selection...")
    constituents = scrape_sp100_constituents(start_year)
    if constituents:
        final_portfolio = select_portfolio_universe(start_year, constituents, portfolio_size)
        print("\n--- Final Selected Portfolio ---")
        print(f"Total tickers: {len(final_portfolio)}")
        print(final_portfolio)
    else:
        print("No constituents found. Exiting.")
        return 

    output_dir = PROJECT_ROOT / "data" / "processed" / "universes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stock_universe_{start_year}.csv"
    
    final_portfolio.to_csv(output_path, index=False)
    
    print(f"\n✅ Final portfolio universe saved to: {output_path}")
    
    return final_portfolio


if __name__ == "__main__":

    start_year = 2014
    acquire_portfolio_universe(start_year)




# check market_data_loader.py for yfinance download function, the warmup setting. And the way to call this module, whether yfinance would be set as though

