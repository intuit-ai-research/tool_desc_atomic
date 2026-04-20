import os, time, json, hashlib
from openai import OpenAI
import tempfile


class LlmWithCache:
    def __init__(self, model='gpt-4o-2024-08-06', api_key=None, cache_dir='.llm_cache'):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    def _generate_cache_key(self, messages, temperature, max_tokens, tools):
        """Create a unique hash key based on input parameters."""
        key_data = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'tools': tools
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

    def call(self, messages, temperature=0.7, max_tokens=10000, tools=None):
        key = self._generate_cache_key(messages, temperature, max_tokens, tools)

        try:
            ret = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
            )
            resp = json.loads(ret.model_dump_json())
            result = resp['choices'][0]['message']
            return result

        except Exception as e:
            print("LLM call failed:", e)
            return None


def test_llm_cache_with_fresh_cache():
    messages = [{'role':'user', 'content': 'which is greater, pi or 3.4?'}]

    with tempfile.TemporaryDirectory() as tmp_cache_dir:
        print(f"Using temporary cache directory: {tmp_cache_dir}")

        llm = LlmWithCache(cache_dir=tmp_cache_dir)

        print("Running first call (should trigger actual LLM)...")
        start = time.time()
        response1 = llm.call(messages)
        duration1 = time.time() - start
        print("First call time:", duration1)
        print("First response:", response1)

        print("\nRunning second call (should be from cache)...")
        start = time.time()
        response2 = llm.call(messages)
        duration2 = time.time() - start
        print("Second call time:", duration2)
        print("Second response:", response2)

        assert response1 == response2, "Cached response does not match original!"
        assert duration2 < duration1, "Second call wasn't faster (may not have used cache)"
        print("\nTest passed: Responses match and caching is functional.")

if __name__ == "__main__":
    test_llm_cache_with_fresh_cache()
