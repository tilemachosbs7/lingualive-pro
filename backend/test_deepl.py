#!/usr/bin/env python3
"""Quick DeepL API test"""
import httpx
import asyncio

async def test_deepl():
    api_key = "8a8f0bbf-9c0b-4a5b-bc2d-0017b1d5bc5c:fx"
    endpoint = "https://api-free.deepl.com/v2/translate"
    
    headers = {
        "Authorization": f"DeepL-Auth-Key {api_key}",
        "Content-Type": "application/json",
    }
    
    test_cases = [
        ("Basic", {"text": ["Hello world"], "target_lang": "EL"}),
        ("With source_lang", {"text": ["Hello world"], "target_lang": "EL", "source_lang": "EN"}),
        ("With EN-US target", {"text": ["Γεια σου κόσμε"], "target_lang": "EN-US"}),
        ("With formality", {"text": ["Hello world"], "target_lang": "DE", "formality": "less"}),
        ("With split_sentences string", {"text": ["Hello world"], "target_lang": "EL", "split_sentences": "nonewlines"}),
        ("Empty text", {"text": [""], "target_lang": "EL"}),
        ("Empty array", {"text": [], "target_lang": "EL"}),
        ("Invalid formality for EN", {"text": ["Hello world"], "target_lang": "EN-US", "formality": "less"}),
    ]
    
    async with httpx.AsyncClient() as client:
        for name, payload in test_cases:
            print(f"\n{name}: {payload}")
            try:
                resp = await client.post(endpoint, headers=headers, json=payload)
                print(f"  Status: {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    trans = data.get('translations', [])
                    if trans:
                        print(f"  Translation: {trans[0].get('text', 'N/A')}")
                    else:
                        print(f"  No translations in response")
                else:
                    print(f"  Error: {resp.text[:300]}")
            except Exception as e:
                print(f"  Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepl())
