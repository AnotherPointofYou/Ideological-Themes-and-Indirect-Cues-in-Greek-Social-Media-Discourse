from typing import List, Dict, Any
import copy

def unique_posts_videos(elements: List[Dict[str, Any]], id_key: str) -> Dict[str, Any]:
    """
    Removes duplicate dictionaries from a list based on a specific ID key.

    Args:
        elements (List[Dict[str, Any]]): A list of dictionaries to filter for uniqueness.
        id_key (str): The key within each dictionary to use as a unique identifier.

    Returns:
        List[Dict[str, Any]]: A list containing only unique dictionaries based on the given key.

    Raises:
        ValueError: If `elements` is not a list, if any item is not a dictionary, or if any dictionary is missing the `id_key`.
        TypeError: If `id_key` is not a string.
    """

    if not isinstance(elements, list):
        raise ValueError("`elements` must be a list of dictionaries.")
    if not all(isinstance(el, dict) for el in elements):
        raise ValueError("All elements in the list must be dictionaries.")
    if not isinstance(id_key, str):
        raise TypeError("id_key must be a string.")

    parsed = set()
    duplicates = set()
    unique_elements = []

    for element in elements:
        id_value = element[id_key]
        if id_value not in parsed:
            parsed.add(id_value)
            unique_elements.append(element)
        else:
            duplicates.add(id_value)

    return unique_elements, duplicates

def rename_dictionary_keys(dictionary_data: Dict, old_key: str, new_key: str) -> Dict[str, Any]:
    """
    It takes a dictionary, and changes a first level key name to a new name

    Args:
        dictionary_data (dict): A dictionary
        old_key (str): The old key name
        new_key (str): The new key name
    
    Returns:
        dictionary_data (dict): The updated dictionary
    """
    if old_key in dictionary_data:
        dictionary_data[new_key] = dictionary_data.pop(old_key)
    else:
        print("Old key wasn't found in the dictionary")

    return dictionary_data

### ID ASSIGNMENT

def assign_unique_author_ids(youtube_data, reddit_data, ogov_data):
    """
    Assigns a unique integer ID to every unique author across all datasets.

    Returns:
        author_to_id: dict mapping author names to unique ids
        updated_youtube, updated_reddit, updated_ogov: copies with 'author_id' fields added
    """
    author_to_id = {}
    next_id = 1

    updated_youtube = copy.deepcopy(youtube_data)
    updated_reddit = copy.deepcopy(reddit_data)
    updated_ogov = copy.deepcopy(ogov_data)

    def get_author_id(author):
        nonlocal next_id
        if not author:  # Handles None, '', etc.
            return None
        if author not in author_to_id:
            author_to_id[author] = next_id
            next_id += 1
        return author_to_id[author]

    # YouTube authors
    for video in updated_youtube:
        for comment in video.get("comments", []):
            author = comment.get("author", "")
            comment["author_id"] = get_author_id(author)

    # Reddit authors
    for post in updated_reddit:
        for comment in post.get("comments", []):
            author = comment.get("author", "")
            comment["author_id"] = get_author_id(author)

    # OpenGov authors: unique id for each comment    
    for entry in updated_ogov:
        nonlocal_id = next_id
        author = entry.get("author_name", "")
        entry["author_id"] = nonlocal_id if author else None
        next_id += 1

    return author_to_id, updated_youtube, updated_reddit, updated_ogov
