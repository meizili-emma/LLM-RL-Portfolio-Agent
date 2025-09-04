# src/data_pipeline/batch_news_analysis.py

from pathlib import Path
import pandas as pd
from langchain_ollama import OllamaLLM 
from src.llm_agents.news_analyst import NewsAnalyst
from src.data_pipeline.news_processing import get_news_sources_whitelist, generate_batch_news_analysis


start_date='2019-04-20'
end_date='2025-08-29'
portfolio_size=21
reference_week=52
warmup_weeks=52 

project_root=Path(__file__).resolve().parent.parent.parent

llm_news_analysis_dir = project_root / 'data' / 'interim' / 'news'
llm_news_analysis_dir.mkdir(parents=True, exist_ok=True)
output_filepath = llm_news_analysis_dir / f'sp500_{start_date}_to_{end_date}_{portfolio_size}_llm_news_analysis.parquet'
features_path = llm_news_analysis_dir / f'sp500_{start_date}_to_{end_date}_{portfolio_size}_rl_news_features.parquet'

portfolio_df = pd.read_csv(project_root / 'data' / 'interim' / 'portfolio' / f'sp500_{start_date}_{portfolio_size}_{reference_week}.csv')
news_df = pd.read_parquet(project_root / 'data' / 'raw' / 'news' / f'sp500_{start_date}_to_{end_date}_{portfolio_size}.parquet')
whitelist = get_news_sources_whitelist(news_df, threshold=0.15) 

llm = OllamaLLM(model='gpt-oss:20b')    
bull_analyst = NewsAnalyst(llm_client=llm, perspective="Bullish")
bear_analyst = NewsAnalyst(llm_client=llm, perspective="Bearish")


generate_batch_news_analysis(
    start_date=start_date,
    end_date=end_date,
    portfolio_df=portfolio_df,
    news_df=news_df,
    bull_analyst=bull_analyst,
    bear_analyst=bear_analyst,
    output_filepath=output_filepath,
    rl_features_path=features_path,
    news_whitelist=whitelist
)