import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
import sys
import os

def analyze_risk_factors(risk_text):
    prompt = f"""
    As an expert financial analyst, analyze the following IPO risk factors and provide a CONCISE assessment.
    
    Risk Factors to Analyze:
    {risk_text}
    
    Provide your analysis in this EXACT format (keep responses brief):
    Score: [number between 0-100]
    
    Score Breakdown:
    - Financial Health: [score/20]
    - Market Position: [score/20]
    - Regulatory: [score/20]
    - Management: [score/20]
    - Technology: [score/20]
    
    Top 3 Critical Concerns:
    1. [brief concern description - max 100 chars]
    2. [brief concern description - max 100 chars]
    3. [brief concern description - max 100 chars]
    """
    
    try:
        # Use environment variable for API key, fall back to hardcoded if not set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = "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"
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fixed model name
            messages=[
                {"role": "system", "content": "You are an expert financial analyst. Provide concise, structured risk assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500  # Limit response length
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing risk factors: {str(e)}")
        return None

def process_risk_analysis(df, company_name=None):
    # Create a list to store results
    results = []
    
    if company_name:
        # Filter for the specific company
        sample_df = df[df['issuer'].str.contains(company_name, case=False, na=False)]
        if sample_df.empty:
            print(f"\nCompany '{company_name}' not found in the dataset.")
            print("\nAvailable companies (first 10):")
            print(df['issuer'].head(10).to_string())
            return
    else:
        # If no company specified, sample 5 companies
        sample_df = df.sample(5)
    
    print("\n" + "="*80)
    print("IPO RISK ANALYSIS REPORT")
    print("="*80)
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        risk_text = row['rf']
        if pd.notna(risk_text):
            print(f"\nAnalyzing company: {row['issuer']}")
            analysis = analyze_risk_factors(risk_text)
            if analysis:
                # Extract score and concerns
                lines = analysis.split('\n')
                score = None
                score_breakdown = []
                concerns = []
                
                for line in lines:
                    if line.startswith('Score:'):
                        try:
                            score = float(line.split(':')[1].strip())
                        except:
                            pass
                    elif line.strip().startswith('- '):
                        score_breakdown.append(line.strip())
                    elif line.strip().startswith(('1.', '2.', '3.')):
                        concerns.append(line.strip())
                
                # Store results
                results.append({
                    'issuer': row['issuer'],
                    'risk_score': score,
                    'score_breakdown': '\n'.join(score_breakdown),
                    'main_concerns': '\n'.join(concerns)
                })
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
    
    # Display results in a formatted way
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    for result in results:
        print("\n" + "-"*80)
        print(f"COMPANY: {result['issuer']}")
        print(f"OVERALL RISK SCORE: {result['risk_score']}")
        print("\nRISK SCORE BREAKDOWN:")
        print(result['score_breakdown'])
        print("\nCRITICAL CONCERNS:")
        print(result['main_concerns'])
        print("-"*80)
    
    # Calculate and display average risk score
    if results:
        avg_score = sum(r['risk_score'] for r in results if r['risk_score'] is not None) / len(results)
        print("\n" + "="*80)
        print(f"AVERAGE RISK SCORE ACROSS SAMPLES: {avg_score:.2f}")
        print("="*80)

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('IPO_data_to_learn.csv')
    
    # Get company name from command line argument if provided
    company_name = None
    if len(sys.argv) > 1:
        company_name = ' '.join(sys.argv[1:])
    
    # Process risk analysis
    process_risk_analysis(df, company_name) 