import re


_HAN = re.compile(r"[\u3400-\u9fff\U00020000-\U000323af]")
RULES = {
    "characters": "unicode_codepoints_including_punctuation_and_internal_whitespace",
    "words": "nonempty_whitespace_delimited_segments"
}


def dataset_settings(config, data):
    dataset = config.get("dataset")
    if dataset is None:
        texts = [row.get("content", row.get("ad_copy", "")) for row in data]
        dataset = ("ATVI" if any("ad_copy" in row for row in data)
                   or not any(_HAN.search(text) for text in texts) else "ADGEN")
    dataset = str(dataset).upper()
    if dataset not in ("ADGEN", "ATVI"):
        raise ValueError("dataset must be ADGEN or ATVI")
    return (dataset, "zh", "characters") if dataset == "ADGEN" else (dataset, "en", "words")


def count_text(text, unit):
    if unit == "characters":
        return len(text)
    if unit == "words":
        return len(text.split())
    raise ValueError("length unit must be characters or words")


def prefix_text(text, length, unit):
    if length < 0:
        raise ValueError("length must be nonnegative")
    if unit == "characters":
        return text[:length]
    if unit == "words":
        return " ".join(text.split()[:length])
    raise ValueError("length unit must be characters or words")
