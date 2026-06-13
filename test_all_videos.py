import re
from pathlib import Path
from deepguard.detectors.video_detector import detect

def get_ground_truth(filename):
    match = re.search(r'video(\d+)', filename)
    if match:
        num = int(match.group(1))
        if 0 <= num <= 8:
            return "FAKE"
        elif 9 <= num <= 20:
            return "REAL"
    return "UNKNOWN"

def main():
    test_dir = Path("testvideo")
    if not test_dir.exists():
        print(f"Directory '{test_dir}' not found. Please create it and add your videos.")
        return
    
    videos = sorted(list(test_dir.glob("video*.*")), key=lambda p: int(re.search(r'video(\d+)', p.name).group(1)) if re.search(r'video(\d+)', p.name) else 999)
    
    if not videos:
        print(f"No videos found in '{test_dir}' starting with 'video'.")
        return

    print(f"{'Filename':<15} | {'Ground Truth':<12} | {'Verdict':<10} | {'Fake Prob':<10} | {'Frames Analysed'}")
    print("-" * 80)
    
    for vid_path in videos:
        gt = get_ground_truth(vid_path.name)
        try:
            result = detect(vid_path)
            verdict = result.verdict
            prob = f"{result.fake_probability*100:.1f}%"
            
            print(f"{vid_path.name:<15} | {gt:<12} | {verdict:<10} | {prob:<10} | {result.frames_analysed}")
            
            # Print frame details if you want to see exactly what triggered the fake response
            # for frame in result.frame_results:
            #     print(f"  Frame {frame.frame_index}: {frame.result.verdict} ({frame.result.fake_probability*100:.1f}%)")
                
        except Exception as e:
            print(f"{vid_path.name:<15} | {gt:<12} | ERROR: {e}")

if __name__ == "__main__":
    main()
