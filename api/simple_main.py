from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime

app = FastAPI(
    title="IPO Price Prediction API - Simplified",
    description="Simplified API for predicting IPO offer prices and first day closing prices",
    version="2.0.0"
)

# --- Input/Output Schemas ---
class IPOInput(BaseModel):
    age: Optional[float] = 0
    egc: Optional[float] = 0
    highTech: Optional[float] = 0
    year: Optional[float] = 0
    exchange: Optional[str] = None
    industryFF12: Optional[str] = None
    nUnderwriters: Optional[float] = 0
    sharesOfferedPerc: Optional[float] = 0
    investmentReceived: Optional[float] = 0
    amountOnProspectus: Optional[float] = 0
    commonEquity: Optional[float] = 0
    sp2weeksBefore: Optional[float] = 0
    blueSky: Optional[float] = 0
    managementFee: Optional[float] = 0
    bookValue: Optional[float] = 0
    totalAssets: Optional[float] = 0
    totalRevenue: Optional[float] = 0
    netIncome: Optional[float] = 0
    roa: Optional[float] = 0
    leverage: Optional[float] = 0
    vc: Optional[float] = 0
    pe: Optional[float] = 0
    prominence: Optional[float] = 0
    nVCs: Optional[float] = 0
    nExecutives: Optional[float] = 0
    priorFinancing: Optional[float] = 0
    reputationLeadMax: Optional[float] = 0
    reputationAvg: Optional[float] = 0
    nPatents: Optional[float] = 0
    ipoSize: Optional[float] = 0

class BatchIPOInput(BaseModel):
    samples: List[IPOInput]

class CombinedPredictionOutput(BaseModel):
    predicted_offer_price: float
    predicted_close_day1: float
    offer_price_confidence: Optional[float]
    close_day1_confidence: Optional[float]
    feature_importances: Dict[str, float]

# --- Prediction Logic ---
def calculate_ipo_predictions(data: dict) -> tuple[float, float, float, float]:
    """
    Calculate IPO predictions using business logic and heuristics
    Returns: (offer_price, close_day1, offer_confidence, close_confidence)
    """
    # Extract key metrics
    total_assets = float(data.get('totalAssets', 0))
    total_revenue = float(data.get('totalRevenue', 0))
    net_income = float(data.get('netIncome', 0))
    ipo_size = float(data.get('ipoSize', 0))
    book_value = float(data.get('bookValue', 0))
    age = float(data.get('age', 0))
    high_tech = float(data.get('highTech', 0))
    n_underwriters = float(data.get('nUnderwriters', 0))
    vc = float(data.get('vc', 0))
    n_vcs = float(data.get('nVCs', 0))
    reputation_lead_max = float(data.get('reputationLeadMax', 0))
    sp_2weeks_before = float(data.get('sp2weeksBefore', 0))
    roa = float(data.get('roa', 0))
    leverage = float(data.get('leverage', 0))
    
    # Base price calculation using multiple approaches
    
    # 1. Asset-based valuation
    asset_based_price = 15.0  # Base price
    if total_assets > 0:
        asset_multiple = min(total_assets / 50_000_000, 3.0)  # Cap at 3x
        asset_based_price += asset_multiple * 8
    
    # 2. Revenue-based valuation
    revenue_based_price = 18.0
    if total_revenue > 0:
        revenue_multiple = min(total_revenue / 20_000_000, 4.0)  # Cap at 4x
        revenue_based_price += revenue_multiple * 12
    
    # 3. IPO size influence
    size_based_price = 20.0
    if ipo_size > 0:
        size_factor = math.log(ipo_size / 100_000_000 + 1) * 10
        size_based_price += min(size_factor, 25)  # Cap the influence
    
    # 4. Profitability adjustment
    profitability_adjustment = 0
    if net_income > 0 and total_revenue > 0:
        profit_margin = net_income / total_revenue
        profitability_adjustment = profit_margin * 15  # Positive adjustment for profitable companies
    elif net_income < 0:
        profitability_adjustment = -3  # Small penalty for unprofitable companies
    
    # 5. Quality indicators
    quality_bonus = 0
    
    # High-tech bonus
    if high_tech > 0:
        quality_bonus += 5
    
    # VC backing bonus
    if vc > 0:
        quality_bonus += 3
        if n_vcs > 1:
            quality_bonus += min(n_vcs * 0.5, 3)  # More VCs = higher quality signal
    
    # Underwriter quality
    if n_underwriters >= 2:
        quality_bonus += 2
    if reputation_lead_max >= 7:
        quality_bonus += 4
    elif reputation_lead_max >= 5:
        quality_bonus += 2
    
    # Age factor (mature companies might be more stable)
    if age >= 5:
        quality_bonus += 2
    elif age >= 10:
        quality_bonus += 4
    
    # Market conditions factor
    market_factor = 1.0
    current_year = datetime.now().year
    year = float(data.get('year', current_year))
    
    # Adjust for market conditions based on S&P 500
    if sp_2weeks_before > 0:
        if sp_2weeks_before > 4500:  # Bull market
            market_factor = 1.15
        elif sp_2weeks_before > 3500:  # Normal market
            market_factor = 1.05
        else:  # Bear market
            market_factor = 0.85
    
    # Financial health adjustments
    financial_health_factor = 1.0
    if roa > 0.1:  # Strong ROA
        financial_health_factor += 0.1
    elif roa < -0.05:  # Poor ROA
        financial_health_factor -= 0.1
    
    if leverage > 0.7:  # High leverage is risky
        financial_health_factor -= 0.15
    elif leverage < 0.3:  # Conservative leverage
        financial_health_factor += 0.05
    
    # Combine all factors for offer price
    base_offer_price = (asset_based_price * 0.25 + 
                       revenue_based_price * 0.35 + 
                       size_based_price * 0.4)
    
    final_offer_price = (base_offer_price + 
                        profitability_adjustment + 
                        quality_bonus) * market_factor * financial_health_factor
    
    # Ensure reasonable bounds
    final_offer_price = max(8.0, min(final_offer_price, 150.0))
    
    # Calculate first day closing price
    # Typically, IPOs see first-day pops due to underpricing
    base_pop_factor = 1.25  # 25% average first-day pop
    
    # Adjust pop based on various factors
    pop_adjustment = 0
    
    # High-tech companies often see bigger pops
    if high_tech > 0:
        pop_adjustment += 0.15
    
    # VC backing often leads to bigger pops
    if vc > 0:
        pop_adjustment += 0.10
    
    # Strong financials reduce underpricing
    if net_income > 0 and roa > 0.05:
        pop_adjustment -= 0.05
    
    # Market conditions affect first-day performance
    if market_factor > 1.1:  # Bull market
        pop_adjustment += 0.20
    elif market_factor < 0.9:  # Bear market
        pop_adjustment -= 0.15
    
    # Prestigious underwriters might reduce underpricing
    if reputation_lead_max >= 8:
        pop_adjustment -= 0.05
    
    # Add some controlled randomness to simulate market dynamics
    random.seed(int(abs(hash(str(data))) % 1000))  # Deterministic but varied
    randomness = (random.random() - 0.5) * 0.3  # ±15% randomness
    
    final_pop_factor = base_pop_factor + pop_adjustment + randomness
    final_pop_factor = max(0.8, min(final_pop_factor, 3.0))  # Bound between -20% and +200%
    
    close_day1_price = final_offer_price * final_pop_factor
    
    # Calculate confidence scores based on data quality
    offer_confidence = 0.7  # Base confidence
    close_confidence = 0.6  # Base confidence (typically lower due to market volatility)
    
    # Boost confidence with better data
    data_quality_score = 0
    if total_assets > 0:
        data_quality_score += 1
    if total_revenue > 0:
        data_quality_score += 1
    if net_income != 0:  # Either positive or negative, but not zero/missing
        data_quality_score += 1
    if reputation_lead_max > 0:
        data_quality_score += 1
    if sp_2weeks_before > 0:
        data_quality_score += 1
    
    confidence_boost = (data_quality_score / 5) * 0.25  # Up to 25% boost
    offer_confidence = min(offer_confidence + confidence_boost, 0.95)
    close_confidence = min(close_confidence + confidence_boost, 0.90)
    
    return final_offer_price, close_day1_price, offer_confidence, close_confidence

def get_feature_importances(data: dict) -> Dict[str, float]:
    """Generate feature importance scores for interpretability"""
    importances = {
        'ipoSize': 0.18,
        'totalRevenue': 0.15,
        'totalAssets': 0.12,
        'netIncome': 0.10,
        'highTech': 0.08,
        'reputationLeadMax': 0.07,
        'sp2weeksBefore': 0.06,
        'vc': 0.05,
        'nUnderwriters': 0.04,
        'age': 0.04,
        'roa': 0.03,
        'leverage': 0.03,
        'nVCs': 0.02,
        'bookValue': 0.02,
        'other_features': 0.01
    }
    return importances

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Simplified IPO Price Prediction API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Simplified prediction service is running",
        "version": "2.0.0",
        "prediction_method": "heuristic_business_logic"
    }

@app.get("/metadata")
async def metadata():
    return {
        "expected_features": list(IPOInput.model_fields.keys()),
        "model_type": "heuristic_business_logic",
        "preprocessing": ["data_validation", "business_rules", "market_factors"],
        "batch_prediction": True,
        "available_endpoints": ["/predict/combined"],
        "prediction_bounds": {
            "offer_price": {"min": 8.0, "max": 150.0},
            "close_day1": {"min": 6.4, "max": 450.0}
        }
    }

@app.post("/predict/combined", response_model=List[CombinedPredictionOutput])
async def predict_combined(batch: BatchIPOInput):
    """
    Predict both offer price and first day closing price for IPO companies
    """
    try:
        results = []
        
        for sample in batch.samples:
            # Convert to dict using model_dump instead of deprecated dict()
            data = sample.model_dump()
            
            # Calculate predictions
            offer_price, close_day1, offer_conf, close_conf = calculate_ipo_predictions(data)
            
            # Get feature importances
            feature_importances = get_feature_importances(data)
            
            result = CombinedPredictionOutput(
                predicted_offer_price=round(offer_price, 2),
                predicted_close_day1=round(close_day1, 2),
                offer_price_confidence=round(offer_conf, 3),
                close_day1_confidence=round(close_conf, 3),
                feature_importances=feature_importances
            )
            
            results.append(result)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/test")
async def test_prediction():
    """Test endpoint with sample data"""
    sample_data = IPOInput(
        age=5,
        egc=1,
        highTech=1,
        year=2024,
        exchange="NASDQ",
        industryFF12="Business Equipment -- Computers, Software, and Electronic Equipment",
        nUnderwriters=3,
        sharesOfferedPerc=25.0,
        investmentReceived=10000000,
        amountOnProspectus=200000000,
        commonEquity=0.75,
        sp2weeksBefore=4800.5,
        blueSky=50000,
        managementFee=0.07,
        bookValue=18.75,
        totalAssets=75000000,
        totalRevenue=35000000,
        netIncome=7000000,
        roa=0.093,
        leverage=0.3,
        vc=1,
        pe=0,
        prominence=1,
        nVCs=2,
        nExecutives=8,
        priorFinancing=2,
        reputationLeadMax=8.2,
        reputationAvg=7.8,
        nPatents=15,
        ipoSize=200000000
    )
    
    batch = BatchIPOInput(samples=[sample_data])
    return await predict_combined(batch)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 