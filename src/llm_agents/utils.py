# src/llm_agents/utils.py

import pandas as pd

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