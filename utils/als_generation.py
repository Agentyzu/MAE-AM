from copy import deepcopy

from utils.ad_spans import extract_ad_spans
from utils.text_length import count_text


def _examples(language, unit):
    cases = ([
        ("Fresh coffee roasted every morning and delivered to your door.", "fresh coffee; delivery", "Fresh coffee delivered"),
        ("Durable walking shoes with soft cushioning for daily comfort.", "walking shoes; cushioning", "Cushioned walking shoes"),
        ("Online language lessons with flexible times and expert teachers.", "language lessons; flexible", "Flexible language lessons")
    ] if language == "en" else [
        ("每日新鲜烘焙的咖啡，支持送货到家。", "新鲜咖啡，送货", "新鲜咖啡送到家"),
        ("步行鞋耐穿且柔软，适合日常行走。", "步行鞋，柔软耐穿", "柔软耐穿步行鞋"),
        ("在线语言课程，由专业教师授课，时间灵活。", "语言课，时间灵活", "灵活在线语言课")
    ])
    return "\n".join(
        f'Example: source={source}; limit={count_text(summary, unit)}; '
        f'<key_phrases>{keywords}</key_phrases><ad id="0">{summary}</ad>'
        for source, keywords, summary in cases)


def generate_dwls(model):
    context = deepcopy(model.messages)
    parts, records, failed = [], [], {}
    try:
        for k, index in enumerate(model.sigma[:model.q]):
            i, target = int(index), int(model.ad_len[k])
            source = model.content[i]
            if target == 0:
                raw, mode = f'<ad id="{i}"></ad>', "zero_length"
            elif count_text(source, model.length_unit) <= target:
                raw, mode = f'<ad id="{i}">{source}</ad>', "source_already_fits"
            else:
                if model.language == "en":
                    instruction = (
                        "Summarize this single advertisement faithfully. First list its key phrases in "
                        "<key_phrases>...</key_phrases>, then give only the ad summary inside "
                        f'<ad id="{i}">...</ad>. Use at most {target} words. Count nonempty '
                        "whitespace-separated segments, including the advertiser name if used; exclude tags. "
                        "Write in English. Do not pad or invent information.\n")
                else:
                    instruction = (
                        "请忠实概括这单条广告。先在<key_phrases>...</key_phrases>内列出关键词，"
                        f'再只在<ad id="{i}">...</ad>内给出摘要，最多{target}字。'
                        "按Unicode字符计，包含所用广告商名称、标点、数字、字母和内部空白，不计标注。"
                        "用中文回答，不凑字、不添加原文没有的信息。\n")
                prompt = (instruction + _examples(model.language, model.length_unit)
                          + f'\nAdvertiser: {model.ad_name[i]}\nSource: {source}\n'
                          + f'Actual summary ID: {i}; limit: {target} {model.length_unit}.')
                model.messages = []  # Summarize each ad independently.
                raw, mode = model.reply(prompt), "llm_summary"
            _, spans, diagnostics = extract_ad_spans(raw, [i])
            span = spans[0]
            records.append({"advertiser_index": i, "target_length": target,
                            "mode": mode, "raw_reply": raw,
                            "status": span["status"], "diagnostics": diagnostics})
            if span["status"] not in ("ok", "empty"):
                failed[i] = span["status"]
            tagged = f'<ad id="{i}">{span["text"]}</ad>'
            parts.append(f'{k + 1}. {tagged}' if model.is_structured else tagged)
    finally:
        model.messages = context
    separator = "\n" if model.is_structured else (" " if model.language == "en" else "；")
    reply = model._record_reply(separator.join(parts))
    for span in model.ad_spans:
        i = span["advertiser_index"]
        if i in failed:
            span.update(status=failed[i], start=None, end=None,
                        generated_length=None, length_error=None)
            model.span_diagnostics.append(f'als_summary_{failed[i]}:{i}')
    model.als_summaries = records
    return reply
