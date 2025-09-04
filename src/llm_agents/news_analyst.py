# src/llm_agents/news_analyst.py

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import List, Literal
from .schemas import NewsAnalysis, CausalChannel, CausalMechanism

# --- Prompt Templates for each Persona ---
BULLISH_PROMPT_TEMPLATE = """
You are an optimistic, sharp-eyed, bullish analyst at a top-tier hedge fund. Your sole task is to find the most positive evidence in the following news and construct the strongest possible bullish thesis for the stock this week.

**Company Profile:**
Ticker: {ticker}
Description: {description}

**Permitted Causal Channels:**
{causal_channels}

**Your Task:**
1. Read all the news for the week of **{start_date} to {end_date} to identify the most dominant narratives**.
2. IGNORE ALL NEGATIVE NEWS. Focus exclusively on positive signals, product launches, positive analyst ratings, and bullish indicators.
3. Construct the strongest possible bullish case and populate the `NewsAnalysis` schema based on this optimistic view.
4. Your `perspective` must be "Bullish".
5. Select the primary causal channel from the permitted list that best describes this narrative.
6. Estimate the magnitude, confidence, directional impact, and significance of this event.
7. Write a concise, one-sentence justification for your analysis.
8. If there is no significant news, select the 'No_Significant_Event' channel and set all scores to 0.0.
9. Format your final output as a single JSON object.

**Example:**
News: "Apple unveils new Vision Pro, stock jumps 5% on hype. Analysts raise price targets for AAPL. Report: Apple faces minor supply chain issues in China."
            
Expected Output:
```json
{{
    "ticker": "AAPL",
    "mechanism": {{
        "channel": "Demand_Sentiment",
        "magnitude": 0.8,
        "confidence": 0.9
     }},
    "directional_impact": 0.7,
     "significance": 0.9,
    "justification": "Positive sentiment from the Vision Pro unveiling and analyst upgrades is the dominant narrative, outweighing minor supply chain concerns."
}}
 ```

**News Text:**
---
{news_text}
---

{format_instructions}
"""


BEARISH_PROMPT_TEMPLATE = """
You are a skeptical, sharp-eyed, bearish analyst at a top-tier hedge fund. Your sole task is to find the most negative evidence and potential risks in the following news and construct the strongest possible bearish thesis for the stock this week.

**Company Profile:**
Ticker: {ticker}
Description: {description}

**Permitted Causal Channels:**
{causal_channels}

**Your Task:**
1. Read all the news for the week of **{start_date} to {end_date} to identify the most dominant narratives**.
2. IGNORE ALL POSITIVE NEWS. Focus exclusively on risks, lawsuits, negative ratings, and bearish indicators.
3. Construct the strongest possible bearish case and populate the `NewsAnalysis` schema based on this pessimistic view.
4. Your `perspective` must be "Bearish".
5. Select the primary causal channel from the permitted list that best describes this narrative.
6. Estimate the magnitude, confidence, directional impact, and significance of this event.
7. Write a concise, one-sentence justification for your analysis.
8. If there is no significant news, select the 'No_Significant_Event' channel and set all scores to 0.0.
9. Format your final output as a single JSON object.

**Example:**
News: "Apple unveils new Vision Pro, stock jumps 5% on hype. Analysts raise price targets for AAPL. Report: Apple faces new supply chain issues in China."
Expected Output:
```json
{{
    "ticker": "AAPL",
    "perspective": "Bearish",
    "mechanism": {{
        "channel": "Supply_Chain_Logistics",
        "magnitude": 0.5,
        "confidence": 0.7
    }},
    "directional_impact": -0.3,
    "significance": 0.4,
    "justification": "Underlying supply chain issues in China for a key new product present a significant operational risk."
}}
 ```

**News Text:**
---
{news_text}
---

{format_instructions}
"""


class NewsAnalyst:
    """
    An LLM-based agent that analyzes weekly news from a specific perspective (Bullish or Bearish).
    """
    def __init__(self, llm_client: AzureChatOpenAI, perspective: Literal["Bullish", "Bearish"]):
        self.perspective = perspective
        self.parser = PydanticOutputParser(pydantic_object=NewsAnalysis)
        
        if self.perspective == "Bullish":
            template = BULLISH_PROMPT_TEMPLATE
        elif self.perspective == "Bearish":
            template = BEARISH_PROMPT_TEMPLATE
        else:
            raise ValueError("Perspective must be either 'Bullish' or 'Bearish'.")

        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["ticker", "description", "news_text", "start_date", "end_date"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "causal_channels": ", ".join(CausalChannel.__args__)
            },
        )
        self.chain = self.prompt_template | llm_client | self.parser

    def analyze(self, ticker: str, description: str, news_text: str, start_date: str, end_date: str) -> NewsAnalysis:
        """Analyzes a concatenated string of weekly news for a specific stock."""
        if not news_text or news_text.isspace():
            return self._get_neutral_analysis(ticker)
        
        try:
            # Pass the required perspective to the schema
            analysis_result = self.chain.invoke({
                "ticker": ticker, 
                "description": description,
                "news_text": news_text,
                "start_date": start_date,
                "end_date": end_date
            })
            # Ensure the perspective is set correctly in the output
            analysis_result.perspective = self.perspective
            return analysis_result
        except Exception as e:
            print(f"An error occurred during {self.perspective} analysis for {ticker}: {e}")
            return self._get_neutral_analysis(ticker)

    def _get_neutral_analysis(self, ticker: str) -> NewsAnalysis:
        """Returns a neutral/default analysis object."""
        return NewsAnalysis(
            ticker=ticker,
            perspective=self.perspective,
            mechanism=CausalMechanism(channel="No_Significant_Event", magnitude=0.0, confidence=1.0),
            directional_impact=0.0,
            significance=0.0,
            justification="No significant news catalysts were identified for this week."
        )

if __name__ == '__main__':
    from src import config # Assuming you have Azure config here
    
    llm = AzureChatOpenAI(
        api_key = config.AZURE_OPENAI_API_KEY,
        endpoint = config.AZURE_OPENAI_ENDPOINT,
        deployment_name = config.AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version = config.AZURE_OPENAI_API_VERSION,
        )
    
    bull_analyst = NewsAnalyst(llm_client=llm, perspective="Bullish")
    bear_analyst = NewsAnalyst(llm_client=llm, perspective="Bearish")

    sample_ticker = "MSFT"
    sample_desc = "Microsoft is a technology company..."
    sample_contents = "Microsoft beats earnings expectations on strong Azure growth. Regulators in the EU announce a new probe into Microsoft's cloud business."

    bull_result = bull_analyst.analyze(sample_ticker, sample_desc, sample_contents, "2025-08-14", "2025-08-21")
    print("\n--- Bull Case ---")
    print(bull_result.model_dump_json(indent=2))

    bear_result = bear_analyst.analyze(sample_ticker, sample_desc, sample_contents, "2025-08-14", "2025-08-21")
    print("\n--- Bear Case ---")
    print(bear_result.model_dump_json(indent=2))