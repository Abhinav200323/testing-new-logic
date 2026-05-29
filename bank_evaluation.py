import csv
from engine.scorer import compare

def run_evaluation():
    # Define test samples from a banking/KYC perspective
    # We define whether we generally expect this to be considered a valid match (True)
    # or a completely different entity (False)
    test_cases = [
        # --- NAMES ---
        {"category": "Name", "str_a": "Abhinav Kumar Baliyan", "str_b": "Abhinav K Baliyan", "expected_match": True},
        {"category": "Name", "str_a": "bank", "str_b": "bank", "expected_match": True},
        {"category": "Name", "str_a": "Elizabeth Taylor", "str_b": "Elizbeth Taylor", "expected_match": True},  # typo
        {"category": "Name", "str_a": "Robert Johnson", "str_b": "William Johnson", "expected_match": False}, # different person
        {"category": "Name", "str_a": "Mohammed Ali", "str_b": "Md. Ali", "expected_match": True},
        {"category": "Name", "str_a": "Jane Doe", "str_b": "Jane Smith", "expected_match": False},
        {"category": "Name", "str_a": "Rahul Kumar Sharma", "str_b": "Sharma Rahul Kumar", "expected_match": True}, # Reordered
        {"category": "Name", "str_a": "Priya Singh", "str_b": "Priyanka Singh", "expected_match": False}, # Different name sharing prefix
        {"category": "Name", "str_a": "A. P. J. Abdul Kalam", "str_b": "Avul Pakir Jainulabdeen Abdul Kalam", "expected_match": True}, # Initials expanded
        {"category": "Name", "str_a": "Sanjay Dutt", "str_b": "Sunjay Dat", "expected_match": True}, # Phonetic spelling variations
        # Domain specific / longer names (4+ words treated as address-like logic by scorer)
        {"category": "Name", "str_a": "Dr. Subrahmanyam Jaishankar", "str_b": "Dr S Jaishankar", "expected_match": True},
        {"category": "Name", "str_a": "Mr. Sachin Ramesh Tendulkar", "str_b": "Sachin R Tendulkar", "expected_match": True},
        {"category": "Name", "str_a": "Late Shri Rajiv Gandhi", "str_b": "Rajiv Gandhi", "expected_match": True},
        {"category": "Name", "str_a": "Shrimati Indira Priyadarshini Gandhi", "str_b": "Indira Gandhi", "expected_match": True},
        
        # --- ADDRESSES ---
        {"category": "Address", "str_a": "123 MG Road, Bangalore 560001", "str_b": "123 M.G. Rd, Bengaluru 560001", "expected_match": True},
        {"category": "Address", "str_a": "456 Park Avenue, NY 10022", "str_b": "456 Park Ave, New York 10022", "expected_match": True},
        {"category": "Address", "str_a": "Flat 4B, Sunset Apts, Mumbai", "str_b": "Apt 4B, Sunset Apartments, Mumbai", "expected_match": True},
        {"category": "Address", "str_a": "123 Main Street, Delhi", "str_b": "456 Main Street, Delhi", "expected_match": False}, # different house number
        {"category": "Address", "str_a": "789 Broadway St, SF", "str_b": "789 Market St, SF", "expected_match": False}, # different street
        {"category": "Address", "str_a": "Plot 42, Sector 15, Gurgaon", "str_b": "Plt 42, Sec 15, Gurugram", "expected_match": True}, # Abbreviations & city name change
        {"category": "Address", "str_a": "Shop No 5, Ground Floor, Linking Road, Bandra", "str_b": "Ground Floor, Shop 5, Linking Rd, Bandra", "expected_match": True}, # Reordered address
        {"category": "Address", "str_a": "10 Downing Street, London", "str_b": "10 Downing Street, Manchester", "expected_match": False}, # Different city entirely
        {"category": "Address", "str_a": "Block C, Vasant Kunj, New Delhi 110070", "str_b": "Blk-C, Vasant Kunj, ND 110070", "expected_match": True}, # Heavy abbreviations
        # Domain specific (Banking/KYC addresses with 4+ words)
        {"category": "Address", "str_a": "Branch Office, State Bank of India, Parliament Street, New Delhi 110001", "str_b": "SBI Branch, Parliament St, New Delhi 110001", "expected_match": True},
        {"category": "Address", "str_a": "HDFC Bank, Cyber City, Phase 2, Gurugram, Haryana", "str_b": "HDFC Cyber City Ph 2 Gurgaon", "expected_match": True},
        {"category": "Address", "str_a": "ICICI Bank ATM, Near Railway Station, Andheri East, Mumbai", "str_b": "ICICI ATM, Andheri East Railway Station, Mumbai", "expected_match": True},
        {"category": "Address", "str_a": "Axis Bank Ltd, Ground Floor, Maker Tower, Cuffe Parade, Mumbai", "str_b": "Axis Bank, Gr Flr, Maker Twr, Cuffe Parade, Mumbai", "expected_match": True},
        {"category": "Address", "str_a": "Punjab National Bank, Main Branch, Sector 17, Chandigarh", "str_b": "PNB Sector 17 Main Branch Chandigarh", "expected_match": True},
    ]

    results = []
    
    for case in test_cases:
        # Run the similarity engine
        result = compare(case["str_a"], case["str_b"])
        
        if isinstance(result, str):
            score = 0.0
            match_level = result
            predicted_match = False
        else:
            score = result["score"]
            match_level = result["match_level"]
            # We consider exact_match, probable_match, and possible_match as "True" (a match)
            # and no_match as "False"
            predicted_match = match_level in ["exact_match", "probable_match", "possible_match"]
        
        # Check if the engine's prediction matches our expectation
        is_correct = (predicted_match == case["expected_match"])
        
        results.append({
            "Category": case["category"],
            "String A": case["str_a"],
            "String B": case["str_b"],
            "Score": score,
            "Match Level": match_level,
            "Expected Match?": case["expected_match"],
            "Predicted Match?": predicted_match,
            "Is Correct?": "✅ Yes" if is_correct else "❌ No"
        })

    # Write to CSV
    csv_filename = "bank_evaluation_results.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["Category", "String A", "String B", "Score", "Match Level", "Expected Match?", "Predicted Match?", "Is Correct?"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Evaluation complete! Results saved to '{csv_filename}'.")
    
    # Print a quick summary to the console
    correct_count = sum(1 for r in results if "✅" in r["Is Correct?"])
    print(f"Accuracy: {correct_count}/{len(test_cases)} ({(correct_count/len(test_cases))*100:.1f}%)")

if __name__ == "__main__":
    run_evaluation()
