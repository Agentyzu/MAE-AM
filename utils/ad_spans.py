import re


_TOKEN = re.compile(r"<\s*/?\s*ad\b[^>]*>", re.IGNORECASE)
_OPEN = re.compile(r"<\s*ad\s+id\s*=\s*[\"'](\d+)[\"']\s*>", re.IGNORECASE)
_CLOSE = re.compile(r"<\s*/\s*ad\s*>", re.IGNORECASE)


def span_instructions(indices, names, language):
    mapping = "\n".join(f'{k + 1}. id="{int(i)}": {names[i]}' for k, i in enumerate(indices))
    if language == "zh":
        return (
            "\n内部广告归属标注（交付前会移除，不计入广告长度）：\n"
            "将每条生成广告的全部文字（含广告商名称）包在 <ad id=\"对应ID\">广告文字</ad> 中。"
            "每个选中ID只用一次，不嵌套，不混合不同广告的文字；自然过渡和非广告内容放在标注外。"
            "保持原定的枚举或融合格式、拍卖顺序和目标长度，不输出JSON或代码围栏。\n" + mapping
        )
    return (
        '\nInternal ad attribution (removed before delivery; excluded from length):\n'
        'Wrap all generated text for each ad, including its advertiser name, in '
        '<ad id="CORRESPONDING_ID">advertising text</ad>. Use each selected ID once, '
        'without nesting or mixing advertisers. Keep organic text/transitions outside. '
        'Retain the requested E/I format, auction order and target lengths; no JSON or code fences.\n'
        + mapping
    )


def extract_ad_spans(raw_reply, indices):
    expected = [int(i) for i in indices]
    if len(set(expected)) != len(expected):
        raise ValueError("Selected advertiser indices must be unique")
    parts, stack, found, invalid, diagnostics = [], [], {}, set(), []
    length = cursor = 0
    for match in _TOKEN.finditer(raw_reply):
        between = raw_reply[cursor:match.start()]
        parts.append(between)
        length += len(between)
        token = match.group()
        opening = _OPEN.fullmatch(token)
        if opening:
            i = int(opening[1])
            if i not in expected:
                diagnostics.append(f"unknown_advertiser:{i}")
            if stack:
                invalid.update(x for x, _ in stack if x is not None)
                invalid.add(i)
                diagnostics.append("nested_ad_markers")
            stack.append((i, length))
        elif _CLOSE.fullmatch(token):
            if not stack:
                diagnostics.append("unmatched_closing_marker")
            else:
                i, start = stack.pop()
                if i in expected:
                    if i in found:
                        invalid.add(i)
                        diagnostics.append(f"duplicate_advertiser:{i}")
                    found[i] = (start, length)
        else:
            diagnostics.append("invalid_ad_marker")
            invalid.update(x for x, _ in stack if x is not None)
            stack.append((None, length))
        cursor = match.end()
    parts.append(raw_reply[cursor:])
    if stack:
        invalid.update(i for i, _ in stack if i is not None)
        diagnostics.append("unclosed_ad_marker")
    clean = "".join(parts)
    if re.search(r"<\s*/?\s*ad\b", clean, re.IGNORECASE):
        raise ValueError("Malformed ad annotation cannot be safely removed from the reply")
    records = []
    for i in expected:
        record = {"advertiser_index": i, "text": "", "start": None, "end": None}
        if i in invalid:
            record["status"] = "invalid"
        elif i not in found:
            record["status"] = "missing"
        else:
            start, end = found[i]
            text = clean[start:end]
            start += len(text) - len(text.lstrip())
            text = text.strip()
            record.update(text=text, start=start, end=start + len(text),
                          status="ok" if text else "empty")
        records.append(record)
    return clean, records, list(dict.fromkeys(diagnostics))
