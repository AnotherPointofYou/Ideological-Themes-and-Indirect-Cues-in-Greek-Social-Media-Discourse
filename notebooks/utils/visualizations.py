from collections import Counter
from typing import List, Tuple, Any, Dict
import textwrap
from .text_analysis_functions import data_cleaning

def text_language_frequency(
    forests: List[Dict[str, List[Dict[str, Any]]]],
    top: int) -> Tuple[
                    List[Tuple[str,int]], # Latin
                    List[Tuple[str,int]], # Greek
                    List[Tuple[str,int]] # Mixed
                    ]:
    """
    Aggregate all comment bodies from all forests, normalize & classify them,
    then return three lists of (text, count) for Latin, Greek, and Mixed,
    filtering count>=2 and taking the top-N by frequency.
    """
    # collect and normalize every comment body
    all_bodies: List[str] = []
    for forest in forests:
        for c in forest.get("comments", []):
            raw = c.get("body", "")
            # norm = data_cleaning.normalize(raw)
            if raw:
                all_bodies.append(raw)

    # group by language groups
    groups: Dict[str, List[str]] = {"Latin": [], "Greek": [], "Mixed": []}
    for b in all_bodies:
        lang = data_cleaning.contains_mixed_latin_greek(b) # returns "Latin"/"Greek"/"Mixed"/None
        if lang in groups:
            groups[lang].append(b)

    # count, filter for non unique comments >=2, find top N frequent words per group
    def top_n(texts: List[str]) -> List[Tuple[str,int]]:
        cnt = Counter(texts)
        freq2 = [(txt, n) for txt, n in cnt.items() if n >= 2]
        freq2.sort(key=lambda x: x[1], reverse=True)
        return freq2[:top]

    latin_top = top_n(groups["Latin"])
    greek_top = top_n(groups["Greek"])
    mixed_top = top_n(groups["Mixed"])

    return latin_top, greek_top, mixed_top

def plot_horizontal_barplot(ax, data, title, wrap_width=40):
    """
    Creates a horizontal barplot for strings counts, constructed for string counts.
    
    Args:
        ax (matplotlib obj): axis
        data (List[Tuple[str:int]]): String count data
        title (str): Barplot title
        wrap_width (int): Wrap extent for y-axis string labels

    Return:
        Barplot object
    """
    texts, counts = zip(*data)
    # wrap long text
    wrapped_texts = ["\n".join(textwrap.wrap(t, wrap_width)) for t in texts]
    y_pos = range(len(wrapped_texts))

    bars = ax.barh(y_pos, counts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped_texts)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Count')

    # annotate counts to the right of each bar
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                str(count), va='center')

