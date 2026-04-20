import json

def extract_all_json_fences(s: str) -> list[str]:
    """
    Return a list of every balanced ```json ... ``` content substring.

    Supports:
    - nesting: ```json ... ```json ... ``` ... ```
    - parallel blocks: ```json ... ``` ... ```json ... ```

    Assumptions (per your note):
    - fences are always properly balanced like parentheses.
    """
    open_fence = "```json"
    close_fence = "```"

    blocks: list[tuple[int, str]] = []  # [(open_index, content)]
    stack: list[tuple[int, int]] = []  # [(open_index, content_start_index)]

    i = 0
    n = len(s)
    while i < n:
        j = s.find("```", i)
        if j == -1:
            break

        if s.startswith(open_fence, j):
            content_start = j + len(open_fence)
            stack.append((j, content_start))
            i = content_start
            continue

        if stack:
            open_idx, content_start = stack.pop()
            blocks.append((open_idx, s[content_start:j]))
            i = j + len(close_fence)
            continue

        i = j + len(close_fence)

    blocks.sort(key=lambda t: t[0])
    return [content for _, content in blocks]


def json_loads(s, parse_with_key=None):
    if "```json" in s:
        blocks = extract_all_json_fences(s)
        if parse_with_key:
            for block in blocks:
                if parse_with_key in block and parse_with_key in json.loads(block):
                    return json.loads(block)    
            else:
                return json.loads(blocks[0])
        else:
            return json.loads(blocks[0])    
    start = None
    end = None
    for t, c in enumerate(s):
        if c == '{' and start is None:
            start = t
        if c == '}':
            end = t
    if start is None or end is None:
        return None
    try:
        return json.loads(s[start: end + 1])
    except:
        return None
    