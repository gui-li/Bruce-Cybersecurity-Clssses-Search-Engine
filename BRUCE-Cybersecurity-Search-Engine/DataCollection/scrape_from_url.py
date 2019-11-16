# -*- coding: utf-8 -*-

'This module is used to scrape all texts from classified url'

from bs4 import BeautifulSoup, Comment
from fake_useragent import UserAgent
from urllib.request import urlopen
from urllib.error import URLError
from typing import List, Tuple
from nltk import sent_tokenize
from _socket import gaierror
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import urllib
import pickle
import bs4
import re
import os

UNIVERSAL_ENCODING = "utf-8"

def tag_visible(element: bs4.element.ResultSet) -> bool:
    """Filter tags in html
    
    Return False for those invisible contents' tag.
    Return True for visible contents' tag.

    Args:
        element: an bs4.element instance waiting to be filtered.
    Returns:
        False for invisible contents' tag.
        True for visible contents' tag.
    """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def crawl(URL: List[str]) -> Tuple[List[str], List[str]]:
    """Crawl corpus from classified URLs

    Args:
        URL: a list of URL string waiting to be scraped.
    Returns:
        contents: a list of string contents scrapepd from given URLs.
        valid_URL: a list of URL have been scraped and reserve for the convenience of a side-by-side annotating.
    """
    valid_URL = []
    contents = []
    for index, url in enumerate(tqdm(URL)):
        request = urllib.request.Request(url, headers={'User-Agent': UserAgent().random})
        try:
            html = urlopen(request, timeout=10).read().decode('utf-8')
        except gaierror as e:
            print(index, e, url)
            continue
        except URLError as e:
            print(index, e, url)
            continue
        except:
            print("Something else went wrong with", url, "\n")
        soup = BeautifulSoup(html, features='lxml')
        texts = soup.findAll(text=True)
        # Format and clean corpus.
        visible_texts = filter(tag_visible, texts)
        visible_texts = "".join(text for text in visible_texts)
        visible_texts = re.sub(r"(\r)+", "\r", visible_texts)
        visible_texts = re.sub(r"(\n)+", "\n", visible_texts)
        visible_texts = re.sub(r"(\r\n)+", "\n", visible_texts)
        visible_texts = re.sub(r"(\r)+", "\r", visible_texts)
        visible_texts = re.sub(r"(\n)+", "\n", visible_texts)
        visible_texts = re.sub(r"\n(\s)+", "\n", visible_texts)
        visible_texts = re.sub(r"\s\n(\s)*", "\n", visible_texts)
        visible_texts = re.sub(r"\n(\W)+\n", "\n", visible_texts)
        visible_texts = re.sub(r"^(\s)+", "", visible_texts)
        visible_texts = re.sub(r"(\s)+$", "", visible_texts)
        visible_texts = re.sub(r"\. ", ".\n", visible_texts)
        visible_texts = re.sub(r"\w(\. )\w", ".\n", visible_texts)
        sentences = sent_tokenize(visible_texts)
        visible_texts = "\n".join(sentence for sentence in sentences)
        if visible_texts:
            valid_URL.append(url)
            contents.append(visible_texts)
        del visible_texts
    assert len(contents) == len(valid_URL)
    return contents, valid_URL
    


if __name__ == '__main__':
    corpus_folder = Path("../Data/Corpus2/")
    url_folder = Path("../Course_Collected/")
    websites = pd.read_csv(url_folder / "Final.csv")
    contents, scraped_URL = crawl(websites.URL)
    pickle.dump(contents, open(corpus_folder / "content.p", "wb"))
    pickle.dump(scraped_URL, open(corpus_folder / "url.p", "wb"))
    contents = pickle.load(open(corpus_folder / "content.p", "rb"))
    scraped_URL = pickle.load(open(corpus_folder / "url.p", "rb"))
    
    # Save corpus to file
    corpus_index = 0
    for content in tqdm(contents):
        file_name = str(corpus_index) + ".txt"
        full_file_path = corpus_folder / file_name
        with open(full_file_path, "w", encoding=UNIVERSAL_ENCODING) as file:
            file.write(content)
        corpus_index += 1

    # Save url list
    url_file_name = "url.txt"
    url_full_file_path = corpus_folder / url_file_name
    file = open(url_full_file_path, "w", encoding=UNIVERSAL_ENCODING)
    for key, value in tqdm(enumerate(scraped_URL)):
        if key != len(scraped_URL) - 1:
            file.write(value + '\n')
        else:
            file.write(value)
    file.close()