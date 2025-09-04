
import pandas as pd
from pathlib import Path
import logging 
from tqdm import tqdm 
from src.llm_agents.news_analyst import NewsAnalyst


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_news_sources_whitelist(news_df: pd.DataFrame, threshold: float = 0.15) -> list:
    """
    Get a whitelist of news sources based on their frequency in the dataset.
    
    Args:
        news_df (pd.DataFrame): DataFrame containing news data with a 'source' column.
        threshold (float): Top percentage of news sources to include in the whitelist.
        
    Returns:
        list: List of news sources that meet the frequency threshold.
    """
    news_sources_count = news_df['source'].value_counts()
    threshold = int(len(news_sources_count) * threshold)
    return news_sources_count.nlargest(threshold).index.tolist()


def get_weekly_news_context(
    ticker: str,
    week_end_date: str,
    all_news_df: pd.DataFrame,
    whitelist: list[str],
    max_articles: int = 100,
    max_length: int = 512
) -> str:
    """
    Selects, filters, and formats news for a given stock and week into
    a single string for LLM analysis.
    """
    week_end = pd.to_datetime(week_end_date)
    week_start = week_end - pd.DateOffset(days=6)
    all_news_df['date'] = pd.to_datetime(all_news_df['date'])
    weekly_news = all_news_df[
        (all_news_df['ticker'] == ticker) &
        (all_news_df['date'] >= week_start) &
        (all_news_df['date'] <= week_end)
    ].copy()
    
    if weekly_news.empty:
        return ""
    if len(weekly_news) > max_articles:
        whitelisted_news = weekly_news[weekly_news['source'].isin(whitelist)]
        if len(whitelisted_news) >= max_articles:
            selected_news = whitelisted_news.nlargest(max_articles, 'date')
        else:
            other_news = weekly_news[~weekly_news['source'].isin(whitelist)]
            needed = max_articles - len(whitelisted_news)
            selected_news = pd.concat([whitelisted_news, other_news.nlargest(needed, 'date')])
    else:
        selected_news = weekly_news

    context_parts = []
    for _, row in selected_news.iterrows():
        content = (str(row['headline']) + " " + str(row['content'] or "")).strip()
        truncated_content = content[:max_length]
        context_parts.append(f"Source: {row['source']} | Content: {truncated_content}")
    return "\n---\n".join(context_parts)


def generate_batch_news_analysis(
    start_date: str,
    end_date: str,
    portfolio_df: pd.DataFrame,
    news_df: pd.DataFrame,
    bull_analyst: NewsAnalyst,
    bear_analyst: NewsAnalyst,
    output_filepath: Path,
    rl_features_path: Path,
    news_whitelist: list
):
    """
    Orchestrates the generation of bull/bear LLM analysis for all tickers
    over a date range, with robust checkpointing and two final output files.
    """
    # --- Checkpointing based on the full analysis file ---
    completed_tasks = set()
    if output_filepath.exists() > 0:
        results_df = pd.read_parquet(output_filepath)
        completed_tasks = set(zip(results_df['ticker'], results_df['date']))
        logging.info(f"Loaded {len(completed_tasks)} already completed tasks.")
    else:
        results_df = pd.DataFrame()
    # --- Main Loop ---
    weekly_date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    tickers = portfolio_df['ticker'].unique().tolist()
    tasks_to_run = [
        (ticker, week_date.strftime('%Y-%-m-%d'))
        for week_date in weekly_date_range
        for ticker in tickers
        if (ticker, week_date.strftime('%Y-%-m-%d')) not in completed_tasks
    ]
    new_results = []
    for ticker, week_str in tqdm(tasks_to_run, desc="Generating Bull/Bear analysis"):
        company_profile = portfolio_df[portfolio_df['ticker'] == ticker].iloc[0]
        news_context = get_weekly_news_context(ticker, week_str, news_df, news_whitelist)
        bull_result = bull_analyst.analyze(
            ticker=ticker, description=company_profile['description'], news_text=news_context,
            start_date=(pd.to_datetime(week_str) - pd.DateOffset(days=6)).strftime('%Y-%m-%d'), end_date=week_str
        )
        bear_result = bear_analyst.analyze(
            ticker=ticker, description=company_profile['description'], news_text=news_context,
            start_date=(pd.to_datetime(week_str) - pd.DateOffset(days=6)).strftime('%Y-%m-%d'), end_date=week_str
        )
        flat_result = {
            'date': week_str,
            'ticker': ticker,
            'bull_directional_impact': bull_result.directional_impact,
            'bull_significance': bull_result.significance,
            'bull_justification': bull_result.justification,
            'bull_mechanism_channel': bull_result.mechanism.channel,
            'bull_mechanism_magnitude': bull_result.mechanism.magnitude,
            'bull_mechanism_confidence': bull_result.mechanism.confidence,
            'bear_directional_impact': bear_result.directional_impact,
            'bear_significance': bear_result.significance,
            'bear_justification': bear_result.justification,
            'bear_mechanism_channel': bull_result.mechanism.channel,
            'bear_mechanism_magnitude': bull_result.mechanism.magnitude,
            'bear_mechanism_confidence': bull_result.mechanism.confidence,
        }
        new_results.append(flat_result)
        if len(new_results) > 0 and len(new_results) % 20 == 0:
            temp_df = pd.DataFrame(new_results)
            combined_df = pd.concat([results_df, temp_df], ignore_index=True)
            combined_df.to_parquet(output_filepath)
            logging.info(f"Saved intermediate progress with {len(combined_df)} total results.")
    
    if new_results:
        temp_df = pd.DataFrame(new_results)
        final_df = pd.concat([results_df, temp_df], ignore_index=True)
        final_df.to_parquet(output_filepath)
        logging.info(f"✅ Final LLM news features saved to {output_filepath}")
        rl_feature_columns = [
            'date', 'ticker', 
            'bull_directional_impact', 'bull_significance',
            'bear_directional_impact', 'bear_significance'
            ]
        final_rl_df = final_df[rl_feature_columns]
        final_rl_df.to_parquet(rl_features_path)
        logging.info(f"✅ RL news feature file saved to {rl_features_path}")
    else:
        logging.info("No new analysis generated. Files are up to date.")