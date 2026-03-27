import asyncio
import os
import sys

from server.inference_engine import InferenceEngine

async def run_test():
    engine = InferenceEngine()
    
    # Check what files are available
    recordings_dir = r"d:\satyadrishti\recordings"
    files = [f for f in os.listdir(recordings_dir) if f.endswith(".wav")]
    
    if not files:
        print("No test audio files found!")
        return
        
    for fname in files:
        print(f"\n======================================")
        print(f"Testing {fname}...")
        path = os.path.join(recordings_dir, fname)
        
        with open(path, "rb") as f:
            audio_bytes = f.read()
            
        result = await engine.analyze_audio(audio_bytes)
        
        print(f"Verdict: {result.get('verdict')}")
        print(f"Biological Veto: {result.get('details', {}).get('biological_veto')}")
        print(f"Veto Reason: {result.get('details', {}).get('veto_reason')}")
        print(f"Explanation:")
        for exp in result.get('details', {}).get('explanation', []):
            print(f" - {exp}")

if __name__ == "__main__":
    asyncio.run(run_test())
