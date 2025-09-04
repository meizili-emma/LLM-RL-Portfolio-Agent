# src/llm_agents/schemas.py

from pydantic import BaseModel, Field
from typing import List, Literal


CausalChannel = Literal[
    "Demand_Sentiment", "Supply_Chain_Logistics", "Competitive_Landscape",
    "Regulatory_Legal", "Macroeconomic_Influence", "Management_Strategy",
    "Earnings_Financials", "No_Significant_Event"
]


class CausalMechanism(BaseModel):
    channel: CausalChannel
    magnitude: float = Field(description="The magnitude of the event's importance (0.0 to 1.0).")
    confidence: float = Field(description="The model's confidence in this causal link (0.0 to 1.0).")


class NewsAnalysis(BaseModel):
    """The structured output for a single-perspective (Bull or Bear) analysis."""
    ticker: str
    perspective: Literal["Bullish", "Bearish"]
    mechanism: CausalMechanism
    directional_impact: float = Field(description="The expected price impact from this perspective (-1.0 to 1.0).")
    significance: float = Field(description="The market-moving significance of this narrative (0.0 to 1.0).")
    justification: str = Field(description="A brief, one-sentence justification for this perspective.")


class FinalThesis(BaseModel):
    """The final, synthesized output from the Chief Strategist."""
    final_directional_conviction: float = Field(description="The final synthesized directional view (-1.0 to 1.0).")
    final_risk_assessment: float = Field(description="The final synthesized risk level (0.0 for low, 1.0 for high).")
    final_significance: float = Field(description="The overall significance of all combined information (0.0 to 1.0).")
    synthesized_justification: str = Field(description="The final, synthesized one-sentence investment thesis for the week.")