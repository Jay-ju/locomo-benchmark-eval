import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python calc_score.py <scored_results.json>")
        sys.exit(1)
        
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)
        
    with open(input_file, "r") as f:
        data = json.load(f)
        
    total = len(data)
    correct = 0
    wrong = 0
    errors = 0
    server_errors = 0
    missing_gt = 0
    
    for item in data:
        judge = item.get("judge_result", {})
        score = judge.get("score", 0.0)
        reason = judge.get("reasoning", "")
        answer = item.get("answer", "")
        
        # Check for server error in answer text
        if "500 Internal Server Error" in answer:
            server_errors += 1
            
        if "No gold" in reason or "No valid gold" in reason or "empty" in reason.lower():
             missing_gt += 1
             
        if score >= 0.7:  # Threshold for correctness
            correct += 1
        else:
            wrong += 1
            
    print(f"=== Evaluation Report: {input_file.name} ===")
    print(f"Total Questions: {total}")
    print(f"Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"Wrong: {wrong} ({wrong/total*100:.1f}%)")
    print("-" * 30)
    print(f"Server Errors (500): {server_errors}")
    print(f"Missing Ground Truth (Est.): {missing_gt}")
    
    valid_total = total - server_errors - missing_gt
    if valid_total > 0:
        print(f"Adjusted Accuracy (excluding errors/missing GT): {correct}/{valid_total} = {correct/valid_total*100:.1f}%")

if __name__ == "__main__":
    main()
