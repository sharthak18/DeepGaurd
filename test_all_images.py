import re
from pathlib import Path
from deepguard.detectors.image_detector import detect

def get_ground_truth(filename):
    match = re.search(r'image(\d+)', filename)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 5:
            return "FAKE"
        elif 6 <= num <= 13:
            return "REAL"
    return "UNKNOWN"

def main():
    test_dir = Path("testimage")
    if not test_dir.exists():
        print(f"Directory {test_dir} not found.")
        return
    
    images = sorted(list(test_dir.glob("image*.*")), key=lambda p: int(re.search(r'image(\d+)', p.name).group(1)))
    
    print(f"{'Filename':<15} | {'Ground Truth':<12} | {'Verdict':<10} | {'Fake Prob':<10} | {'Model Details'}")
    print("-" * 120)
    
    for img_path in images:
        gt = get_ground_truth(img_path.name)
        try:
            result = detect(img_path)
            verdict = result.verdict
            prob = f"{result.fake_probability*100:.1f}%"
            
            models_info = []
            for score in result.model_scores:
                short_name = score.model_id.split('/')[-1][:15]
                models_info.append(f"{short_name}:{score.label}({score.confidence*100:.1f}%)")
            
            print(f"{img_path.name:<15} | {gt:<12} | {verdict:<10} | {prob:<10} | {', '.join(models_info)}")
        except Exception as e:
            print(f"{img_path.name:<15} | {gt:<12} | ERROR: {e}")

if __name__ == "__main__":
    main()
