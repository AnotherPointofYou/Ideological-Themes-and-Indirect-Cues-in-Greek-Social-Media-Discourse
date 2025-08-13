import math, re
import pandas as pd

def split_text_natural_or_equal(tokenizer, text: str, max_length: int = 512) -> list[str]:
    # tokenize and measure
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    total = len(token_ids)
    if total <= max_length:
        return [text]
    
    # chunks & target size
    n_chunks = math.ceil(total / max_length)
    target   = math.ceil(total / n_chunks)
    
    # split on sentence boundaries
    sentences = re.split(r'(?<=[\.\!\?;])\s+', text)
    if len(sentences) <= 1:
        # fallback to token‐wise slicing
        chunks = []
        for i in range(n_chunks):
            start = i * target
            end   = start + target
            ids   = token_ids[start:end]
            chunks.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
        return chunks
    
    # accumulate sentences into balanced chunks
    chunks, cur_ids, cur_len = [], [], 0
    def flush():
        nonlocal cur_ids, cur_len
        if cur_len:
            chunks.append(tokenizer.decode(cur_ids, skip_special_tokens=True).strip())
        cur_ids, cur_len = [], 0
    
    for sent in sentences:
        s_ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
        L = len(s_ids)
        if L > target:
            flush()
            for j in range(0, L, target):
                part = s_ids[j:j+target]
                chunks.append(tokenizer.decode(part, skip_special_tokens=True).strip())
        else:
            if cur_len + L > target and cur_len > 0:
                flush()
            cur_ids.extend(s_ids)
            cur_len += L
    
    flush()
    return chunks

def clean_text(cleaning_object, text: str) -> str:

    patterns_to_remove = [
    "διαφωνω",
    "συμφωνω",
    "νομοσχεδιο",
    "νομοσχεδιου"]

    repeated_pattern = re.compile(
        r'''
        \b\w*(\w)\1{2,}\b | # single character repeated 3+ times
        \b(\w{2})\2{2,}\b # a two-character sequence repeated 3+ times
        ''',
        flags=re.VERBOSE | re.UNICODE
    )
    
    literal_pattern = "|".join(map(re.escape, patterns_to_remove))
    
    text = re.sub(literal_pattern, "", text, flags=re.IGNORECASE)
    text = repeated_pattern.sub("", text)

    text = cleaning_object.remove_greek_stopwords(text)

    return text

def get_topic_words(model, topic, n_words=5):
    words_scores = model.get_topic(topic)
    return ", ".join([w for w, _ in words_scores[:n_words]])

def summarize_doc(df, model):
    # most frequent topic in this document
    dom_topic = df["topic"].value_counts().idxmax()
    # only the rows for that topic
    topic_rows = df[df["topic"] == dom_topic]
    # average probability (you could also take .max() instead of .mean())
    avg_prob = topic_rows["topic_prob"].mean()
    # top representative words
    words = get_topic_words(model, dom_topic, n_words=12)
    return pd.Series({
        "dominant_topic": dom_topic,
        "topic_prob":      avg_prob,
        "topic_words":     words
    })
