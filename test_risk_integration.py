#!/usr/bin/env python3
"""
Test script for Risk Factor Analysis integration
Tests the complete flow from AI service to backend storage
"""

import asyncio
import httpx
import json
import os
from datetime import datetime

# Test configuration
AI_SERVICE_URL = "http://localhost:8001"
BACKEND_URL = "http://localhost:8000"

# Sample risk factor text for testing
SAMPLE_RISK_TEXT = """
The company faces significant risks including intense competition in the technology sector, 
dependence on key personnel including the CEO and CTO, regulatory uncertainties around data privacy, 
cybersecurity threats to customer data, and market volatility affecting revenue streams. 
Revenue concentration with three major clients poses significant risk if any client relationship deteriorates. 
The company has limited operating history in international markets and may face challenges scaling operations. 
Intellectual property disputes are ongoing with two competitors, and rapid technological changes could make our products obsolete. 
Additionally, the company currently operates with negative cash flows and may require additional financing within 12 months.
Operating expenses have increased 40% year-over-year while revenue growth has slowed to 15%.
The regulatory environment for our industry is becoming more stringent with new compliance costs expected.
"""

async def test_ai_service_risk_analysis():
    """Test the AI service risk analysis endpoint"""
    print("🔬 Testing AI Service Risk Analysis...")
    
    request_data = {
        "samples": [
            {
                "risk_text": SAMPLE_RISK_TEXT,
                "company_name": "TechCorp Inc."
            }
        ]
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/analyze/risk-factors",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ AI Service Response:")
                print(f"   Risk Score: {result[0]['risk_score']}")
                print(f"   Risk Level: {result[0]['risk_level']}")
                print(f"   Critical Concerns: {len(result[0]['critical_concerns'])}")
                return result[0]
            else:
                print(f"❌ AI Service Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ AI Service Request Failed: {str(e)}")
        return None

async def test_backend_risk_analysis():
    """Test the backend standalone risk analysis endpoint"""
    print("\n🔗 Testing Backend Risk Analysis Integration...")
    
    request_data = {
        "risk_text": SAMPLE_RISK_TEXT,
        "company_name": "TechCorp Inc."
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/ipo/analyze-risk-factors",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Backend Response:")
                print(f"   Success: {result['success']}")
                print(f"   Risk Score: {result['risk_analysis']['risk_score']}")
                print(f"   Risk Level: {result['risk_analysis']['risk_level']}")
                print(f"   Fallback Used: {result['fallback_used']}")
                return result
            else:
                print(f"❌ Backend Error: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Backend Request Failed: {str(e)}")
        return None

async def test_health_endpoints():
    """Test health endpoints for both services"""
    print("\n🏥 Testing Service Health...")
    
    # Test AI Service
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{AI_SERVICE_URL}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ AI Service: {health.get('status', 'unknown')}")
                if 'risk_analysis' in health:
                    print(f"   Risk Analysis: {health['risk_analysis']['status']}")
            else:
                print(f"❌ AI Service Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ AI Service Unreachable: {str(e)}")
    
    # Test Backend
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{BACKEND_URL}/ipo/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Backend Service: {health.get('status', 'unknown')}")
            else:
                print(f"❌ Backend Health Check Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Backend Service Unreachable: {str(e)}")

async def test_complete_workflow():
    """Test the complete multistep form workflow with risk analysis"""
    print("\n🔄 Testing Complete Workflow...")
    
    # Sample complete form data
    form_data = {
        # Step 1: Registration
        "companyName": "TechCorp Inc.",
        "registrationNumber": "TC123456789",
        "email": f"test+{datetime.now().strftime('%Y%m%d_%H%M%S')}@techcorp.com",
        "password": "SecurePassword123!",
        
        # Step 2: Prediction Data
        "industryFF12": "Business Equipment -- Computers, Software, and Electronic Equipment",
        "exchange": "NASDQ",
        "highTech": "true",
        "egc": "true", 
        "vc": "true",
        "pe": "false",
        "prominence": "true",
        "age": "5",
        "year": "2024",
        "nUnderwriters": "3",
        "sharesOfferedPerc": "25.0",
        "investmentReceived": "50000000",
        "amountOnProspectus": "200000000",
        "commonEquity": "0.75",
        "sp2weeksBefore": "4800.5",
        "blueSky": "75000",
        "managementFee": "0.07",
        "bookValue": "22.50",
        "totalAssets": "150000000",
        "totalRevenue": "75000000",
        "netIncome": "15000000",
        "roa": "0.10",
        "leverage": "0.25",
        "nVCs": "3",
        "nExecutives": "12",
        "priorFinancing": "25000000",
        "reputationLeadMax": "8.5",
        "reputationAvg": "8.0",
        "nPatents": "25",
        "ipoSize": "200000000",
        
        # Step 3: Risk Analysis
        "additionalInfo": "Company seeking to expand international operations",
        "uploadPdf": False,
        "riskFactorsText": SAMPLE_RISK_TEXT
    }
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/ipo/submit-multistep-form",
                json=form_data
            )
            
            if response.status_code == 201:
                result = response.json()
                print("✅ Complete Workflow Success:")
                print(f"   User ID: {result['user']['id']}")
                print(f"   Prediction ID: {result['prediction']['id']}")
                print(f"   Predicted Offer Price: ${result['prediction'].get('predictedOfferPrice', 'N/A')}")
                print(f"   Predicted Close Day 1: ${result['prediction'].get('predictedCloseDay1', 'N/A')}")
                
                if result.get('riskAnalysis'):
                    risk = result['riskAnalysis']
                    print(f"   Risk Score: {risk.get('riskScore', 'N/A')}")
                    print(f"   Risk Level: {risk.get('riskLevel', 'N/A')}")
                    print(f"   Analysis Status: {risk.get('analysisStatus', 'N/A')}")
                
                return result
            else:
                print(f"❌ Complete Workflow Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
    except Exception as e:
        print(f"❌ Complete Workflow Request Failed: {str(e)}")
        return None

async def main():
    """Main test function"""
    print("🚀 Starting Risk Factor Analysis Integration Tests")
    print("=" * 60)
    
    # Check environment
    print(f"AI Service URL: {AI_SERVICE_URL}")
    print(f"Backend URL: {BACKEND_URL}")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Risk analysis may use fallback responses.")
    
    print("\n")
    
    # Run tests
    await test_health_endpoints()
    
    ai_result = await test_ai_service_risk_analysis()
    backend_result = await test_backend_risk_analysis()
    
    if ai_result and backend_result:
        print("\n🎯 Integration Test Results:")
        print(f"   AI Risk Score: {ai_result['risk_score']}")
        print(f"   Backend Risk Score: {backend_result['risk_analysis']['risk_score']}")
        print(f"   Scores Match: {ai_result['risk_score'] == backend_result['risk_analysis']['risk_score']}")
    
    # Test complete workflow if basic tests pass
    if backend_result and backend_result.get('success'):
        workflow_result = await test_complete_workflow()
    
    print("\n" + "=" * 60)
    print("✅ Integration tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 