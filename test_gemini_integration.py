import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.annotator import LLMAnnotator

def run_gemini_test_suite():
    print("=" * 70)
    print("  GOOGLE GEMINI INTEGRATION & STRUCTURED OUTPUT VERIFICATION TEST  ")
    print("=" * 70)

    annotator = LLMAnnotator()
    print(f"[*] Configured Provider : {annotator.provider}")
    print(f"[*] Configured Model    : {annotator.model}")
    print(f"[*] Client Initialized  : {annotator.gemini_client is not None}")
    print(f"[*] Client Configured   : {annotator.gemini_client.is_configured() if annotator.gemini_client else False}\n")

    # 4 Representative Multilingual Test Cases
    test_samples = [
        {
            "id": "test_01",
            "lang": "Hinglish (Sarcasm)",
            "text": "Wah bhai wah, 3 ghante late flight aur bolte hain technical delay! Shabaash airlines!",
            "strategy": "reasoning_aware"
        },
        {
            "id": "test_02",
            "lang": "Marathi (Devanagari)",
            "text": "खूपच सुंदर चित्रपट आहे, सर्वांनी आवर्जून पाहावा! #मराठी",
            "strategy": "zero_shot"
        },
        {
            "id": "test_03",
            "lang": "Romanized Marathi (Sarcasm)",
            "text": "Wah re wah, 2 taas light geli aani he bill pathavtat!",
            "strategy": "few_shot"
        },
        {
            "id": "test_04",
            "lang": "Mixed Script (Latin + Devanagari)",
            "text": "Superb काम केलं आहे @team, proud of you all!",
            "strategy": "reasoning_aware"
        }
    ]

    results = []
    successful = 0
    failed = 0
    total_tokens = 0
    total_cost = 0.0

    print("--- Executing Live Test Invocations ---")
    for sample in test_samples:
        t0 = time.time()
        print(f"\n[+] Testing Sample {sample['id']} ({sample['lang']}) using Strategy: {sample['strategy']}...")
        print(f"    Text: \"{sample['text']}\"")
        
        # Invoke annotator
        ann = annotator.annotate_single_text(sample["text"], strategy=sample["strategy"], force_refresh=False)
        elapsed = round(time.time() - t0, 3)
        
        if ann and "sentiment" in ann and "toxicity" in ann:
            successful += 1
            total_tokens += ann.get("total_tokens", 0)
            total_cost += ann.get("estimated_cost", 0.0)
            print(f"    -> Status      : {ann.get('status', 'success').upper()} (in {elapsed}s)")
            print(f"    -> Sentiment   : {ann['sentiment']}")
            print(f"    -> Toxicity    : {ann['toxicity']}")
            print(f"    -> Sarcasm     : {ann['sarcasm']}")
            print(f"    -> Script Mixed: {ann['script_mixing']}")
            print(f"    -> Confidence  : {ann['confidence']}")
            print(f"    -> Explanation : {ann['explanation']}")
            print(f"    -> Tokens/Cost : {ann.get('total_tokens', 0)} tokens | ${ann.get('estimated_cost', 0):.6f}")
            print(f"    -> Cached      : {ann.get('cached', False)}")
        else:
            failed += 1
            print(f"    -> FAILED: Invalid response payload")

        results.append(ann)

    # Test Caching on repeat call
    print("\n--- Verifying Deterministic Cache Hit ---")
    cached_test = annotator.annotate_single_text(test_samples[0]["text"], strategy=test_samples[0]["strategy"], force_refresh=False)
    cache_hit = cached_test.get("cached", False)
    print(f"[*] Repeat Call Cache Hit: {cache_hit} (Instantaneous response without API latency)")

    print("\n" + "=" * 70)
    print("  TEST SUITE SUMMARY  ")
    print("=" * 70)
    print(f"• LLM Model Used        : {annotator.model}")
    print(f"• Total Tests Run       : {len(test_samples)}")
    print(f"• Successful Annotations: {successful}")
    print(f"• Failed Annotations    : {failed}")
    print(f"• Total Tokens Tracked  : {total_tokens}")
    print(f"• Estimated Total Cost  : ${total_cost:.6f}")
    print(f"• Cache Hit Verified    : {cache_hit}")
    print(f"• Cache File Location   : data/llm_annotations/llm_cache.json")
    print(f"• Strategy Logs Location: data/llm_annotations/gemini_*.jsonl")
    print("=" * 70)

if __name__ == "__main__":
    run_gemini_test_suite()
