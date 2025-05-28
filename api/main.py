from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
import os

app = FastAPI(
    title="IPO Price Prediction API",
    description="API for predicting IPO offer prices and first day closing prices using ensemble models",
    version="1.0.0"
)

# Get the absolute path to the models directory
MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "models" / "trained"

# --- Model Loading Functions ---
@lru_cache(maxsize=2)
def load_model(target: str):
    try:
        model_path = MODEL_DIR / f"ensemble_{target}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model for {target}: {str(e)}")

@lru_cache(maxsize=2)
def load_imputer(target: str):
    try:
        imputer_path = MODEL_DIR / f"imputer_{target}.joblib"
        if not imputer_path.exists():
            raise FileNotFoundError(f"Imputer file not found: {imputer_path}")
        return joblib.load(imputer_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load imputer for {target}: {str(e)}")

@lru_cache(maxsize=2)
def load_scaler(target: str):
    try:
        scaler_path = MODEL_DIR / f"scaler_{target}.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        return joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler for {target}: {str(e)}")

@lru_cache(maxsize=2)
def load_feature_selector(target: str):
    try:
        selector_path = MODEL_DIR / f"feature_selector_{target}.joblib"
        if not selector_path.exists():
            raise FileNotFoundError(f"Feature selector file not found: {selector_path}")
        return joblib.load(selector_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load feature selector for {target}: {str(e)}")

@lru_cache(maxsize=2)
def load_poly(target: str):
    try:
        poly_path = MODEL_DIR / f"poly_{target}.joblib"
        if not poly_path.exists():
            raise FileNotFoundError(f"Polynomial transformer file not found: {poly_path}")
        return joblib.load(poly_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load polynomial transformer for {target}: {str(e)}")

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

class PredictionOutput(BaseModel):
    predicted_price: float
    confidence_score: Optional[float]
    feature_importances: Dict[str, float]

class CombinedPredictionOutput(BaseModel):
    predicted_offer_price: float
    predicted_close_day1: float
    offer_price_confidence: Optional[float]
    close_day1_confidence: Optional[float]
    feature_importances: Dict[str, float]

# --- Preprocessing ---
def preprocess_input(df: pd.DataFrame, target: str = 'offerPrice') -> pd.DataFrame:
    exchange_map = {'AMEX': 0, 'NASDQ': 1, 'NYSE': 2}
    industry_map = {
        'Business Equipment -- Computers, Software, and Electronic Equipment': 0,
        'Chemicals and Allied Products': 1,
        "Consumer Durables -- Cars, TV's, Furniture, Household Appliances": 2,
        'Consumer NonDurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys': 3,
        'Finance': 4,
        'Healthcare, Medical Equipment, and Drugs': 5,
        'Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing': 6,
        'Oil, Gas, and Coal Extraction and Products': 7,
        'Other': 8,
        'Telephone and Television Transmission': 9,
        'Utilities': 10,
        'Wholesale, Retail, and Some Services (Laundries, Repair Shops)': 11
    }
    df = df.copy()
    
    # Encode categorical features
    df['exchange'] = df['exchange'].map(exchange_map)
    df['industryFF12'] = df['industryFF12'].map(industry_map)
    
    # Create derived features
    df['ipoSize_normalized'] = np.log(df['ipoSize'] + 1)
    df = df.drop(columns=['ipoSize'])
    
    # Engineered features
    df['revenue_per_asset'] = df['totalRevenue'] / (df['totalAssets'] + 1e-6)
    df['income_per_revenue'] = df['netIncome'] / (df['totalRevenue'] + 1e-6)
    df['equity_per_asset'] = df['commonEquity'] / (df['totalAssets'] + 1e-6)
    df['investment_per_share'] = df['investmentReceived'] / (df['sharesOfferedPerc'] + 1e-6)
    df['vc_to_exec_ratio'] = df['nVCs'] / (df['nExecutives'] + 1e-6)
    df['patent_to_revenue'] = df['nPatents'] / (df['totalRevenue'] + 1e-6)
    
    # Remove features not present during model training
    for col in ['book_to_market', 'leverage_ratio', 'roa_ratio']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Ensure missing features are present
    for col in ['age', 'egc', 'highTech', 'year']:
        if col not in df.columns:
            df[col] = 0
    
    # Note: Don't force match to imputer features here as it drops engineered features
    # The models were trained with the full feature set including engineered features
    
    return df

def get_predictions(features_df: pd.DataFrame, target: str) -> tuple:
    """Helper function to get predictions for a specific target"""
    try:
        model = load_model(target)
        imputer = load_imputer(target)
        scaler = load_scaler(target)
        feature_selector = None
        try:
            feature_selector = load_feature_selector(target)
        except Exception:
            feature_selector = None
        # Try to load polynomial transformer, but handle missing file gracefully
        try:
            poly = load_poly(target)
        except Exception:
            poly = None
        
        # Impute missing values
        features = imputer.transform(features_df)
        
        # Apply feature selection if available
        if feature_selector is not None:
            features = feature_selector.transform(features)
            
        # Scale features
        features = scaler.transform(features)
        
        # Apply polynomial features if available and expected by the model
        if poly is not None:
            # Determine the model's expected input shape
            expected_shape = None
            if hasattr(model, 'n_features_in_'):
                expected_shape = model.n_features_in_
            elif hasattr(model, 'feature_names_in_'):
                expected_shape = len(model.feature_names_in_)
            
            # If the model expects the expanded feature set, apply poly.transform
            if expected_shape == poly.n_output_features_:
                # Ensure input to poly.transform has the correct number of features
                n_expected = poly.n_features_in_
                n_actual = features.shape[1]
                if n_actual < n_expected:
                    padded_features = np.zeros((features.shape[0], n_expected))
                    padded_features[:, :n_actual] = features
                    features = padded_features
                elif n_actual > n_expected:
                    features = features[:, :n_expected]
                features = poly.transform(features)
            # Otherwise, skip polynomial transformation
        
        predictions = model.predict(features)
        
        # Calculate confidence scores
        confidence = None
        try:
            preds = np.array([est.predict(features) for est, _ in model.estimators_])
            confidence = [float(1 / (1 + np.std(preds[:, i]))) for i in range(preds.shape[1])] if preds.ndim == 2 else [float(1 / (1 + np.std(preds)))] * len(predictions)
        except Exception as e:
            print(f"Warning: Could not calculate confidence scores: {str(e)}")
            confidence = [None] * len(predictions)
        
        # Get feature importances
        feature_importances = {}
        try:
            if hasattr(model.final_estimator_, 'feature_importances_'):
                importances = model.final_estimator_.feature_importances_
                feature_importances = {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        except Exception as e:
            print(f"Warning: Could not get feature importances: {str(e)}")
            feature_importances = {}
        
        return predictions, confidence, feature_importances
    except Exception as e:
        raise RuntimeError(f"Error in prediction pipeline for {target}: {str(e)}")

# --- Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the IPO Price Prediction API"}

@app.get("/health")
async def health_check():
    try:
        missing_components = []
        for target in ['offerPrice', 'closeDay1']:
            # Check required components
            try:
                model = load_model(target)
                imputer = load_imputer(target)
                scaler = load_scaler(target)
                feature_selector = load_feature_selector(target)
            except Exception as e:
                missing_components.append(f"{target}: {str(e)}")
                
            # Check optional components (poly is optional)
            try:
                poly = load_poly(target)
            except Exception:
                # Polynomial transformer is optional, don't fail health check
                pass
                
        if missing_components:
            return {"status": "unhealthy", "errors": missing_components}
        return {"status": "healthy", "message": "All required model components loaded successfully"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/metadata")
async def metadata():
    return {
        "expected_features": list(IPOInput.schema()['properties'].keys()),
        "model_type": "ensemble",
        "preprocessing": ["imputer", "scaler", "feature_selector", "poly", "feature engineering"],
        "batch_prediction": True,
        "available_endpoints": [
            "/predict/offer-price",
            "/predict/close-day1",
            "/predict/combined"
        ]
    }

@app.post("/predict/offer-price", response_model=List[PredictionOutput])
async def predict_offer_price(batch: BatchIPOInput):
    try:
        df = pd.DataFrame([sample.dict() for sample in batch.samples])
        features_df = preprocess_input(df)
        predictions, confidence, feature_importances = get_predictions(features_df, 'offerPrice')
        
        return [
            PredictionOutput(
                predicted_price=float(pred),
                confidence_score=confidence[i] if confidence else None,
                feature_importances=feature_importances
            ) for i, pred in enumerate(predictions)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/close-day1", response_model=List[PredictionOutput])
async def predict_close_day1(batch: BatchIPOInput):
    try:
        df = pd.DataFrame([sample.dict() for sample in batch.samples])
        features_df = preprocess_input(df)
        predictions, confidence, feature_importances = get_predictions(features_df, 'closeDay1')
        
        return [
            PredictionOutput(
                predicted_price=float(pred),
                confidence_score=confidence[i] if confidence else None,
                feature_importances=feature_importances
            ) for i, pred in enumerate(predictions)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/combined", response_model=List[CombinedPredictionOutput])
async def predict_combined(batch: BatchIPOInput):
    try:
        df = pd.DataFrame([sample.dict() for sample in batch.samples])
        features_df = preprocess_input(df)
        
        # Step 1: Get predictions for offerPrice
        offer_predictions, offer_confidence, offer_importances = get_predictions(features_df.copy(), 'offerPrice')
        
        # Step 2: Add predicted_offerPrice as a feature for closeDay1 prediction
        features_df_with_offer = features_df.copy()
        features_df_with_offer['predicted_offerPrice'] = offer_predictions
        
        # Reorder columns to match imputer's expected order for closeDay1
        imputer_close = load_imputer('closeDay1')
        if hasattr(imputer_close, 'feature_names_in_'):
            expected_features = list(imputer_close.feature_names_in_)
            for col in expected_features:
                if col not in features_df_with_offer.columns:
                    features_df_with_offer[col] = 0
            features_df_with_offer = features_df_with_offer[expected_features]
        
        # Step 3: Get predictions for closeDay1
        close_predictions, close_confidence, close_importances = get_predictions(features_df_with_offer, 'closeDay1')
        
        # Combine feature importances
        combined_importances = {**offer_importances, **{f"close_{k}": v for k, v in close_importances.items()}}
        
        return [
            CombinedPredictionOutput(
                predicted_offer_price=float(offer_pred),
                predicted_close_day1=float(close_pred),
                offer_price_confidence=offer_confidence[i] if offer_confidence else None,
                close_day1_confidence=close_confidence[i] if close_confidence else None,
                feature_importances=combined_importances
            ) for i, (offer_pred, close_pred) in enumerate(zip(offer_predictions, close_predictions))
        ]
    except Exception as e:
        error_msg = str(e)
        if "XGBoost Library" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="Model loading error: XGBoost library could not be loaded. Please ensure OpenMP runtime is installed correctly."
            )
        raise HTTPException(status_code=400, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 