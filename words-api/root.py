import random
import re
from typing import List
import gensim.downloader
from nltk import edit_distance
from flask import Blueprint, jsonify, request
import sys

ALPHABETIC = re.compile(r"[a-zA-Z]+")

model = gensim.downloader.load('word2vec-google-news-300')

# This is to select how many words we get out of the model, in terms of the count variable
# The number of words is gonna be TIMES_MORE_WORDS * count, out of which we pick count final words
TIMES_MORE_WORDS = 5

# We do not allow terms longer than this. This is directed towards multiword terms that generate big 
# word search grids
DEFAULT_WORD_MAX_LENGTH = 20

def filter_func(word: str, theme: str):
  # We do not allow any other character except letters and underscore
  if re.search('[^a-zA-z_]', word) != None:
    return False
  
  # A reasonable similarity is at least a third of the letters are different
  return edit_distance(word.lower(), theme.lower()) > len(theme) // 3

def query_model(theme: str, count: int, max_length: int):
  try:
    # Get the top count most similar words to string
    initial = list(map((lambda p: p[0]), model.most_similar(theme, topn=count)))
    # And filter out the ones which are too similar
    initial = list(filter(lambda w: filter_func(w, theme), initial))
    # We filter words that are too long out of the answer, but we keep them for feedback
    ans = list(filter(lambda w: len(w.replace(" ", "")) <= max_length, initial))

    # Calculate how many words we still have to get and take a random sample from all the results
    n = len(ans)
    diff = n - count
    if diff > 0:
      old = set(initial)
      new = set()
      for res in initial:
        l = list(filter((lambda s: s not in old, new), query_model(res, diff)))
        new.update(l)
      ans += random.sample(new, diff)
    
    return ans 
  except KeyError:
    return []

def try_formats(theme: str, no_results: int, max_length: int):
  theme = add_underscore(theme)

  # Formats that are the most likely to be contained in the model and return meaningful results
  id = lambda s: s
  lower = lambda s: s.lower()
  title = lambda s: s.title()
  formats = [id, lower, title]

  for f in formats:
    res = query_model(f(theme), no_results, max_length)
    if res != []:
      break

  return list(map(remove_underscore, res))
  
def remove_underscore(string: str):
  return string.replace('_', ' ')

def add_underscore(string: str):
  return string.replace(' ', '_')

def pick_words(theme: str, count: int, allow_multiword=True, already_used=[], max_length=DEFAULT_WORD_MAX_LENGTH) -> List[str]:
  """
  Return a list of 'count' randomly selected words for given theme 'theme'.
  If 'allow_multiword' is false, selections consisting of multiple words (e.g., space-separated or hyphen-separated) will not be included.
  """

  word_bank = try_formats(theme, TIMES_MORE_WORDS * count, max_length)
  
  # TODO: We are not guaranteed to have enough words after we do this, we have to pass
  # allow_multiword in the other functions as well
  if not allow_multiword:
    word_bank = list(filter(ALPHABETIC.fullmatch, word_bank))

  word_bank = list(filter(lambda w: w not in already_used, word_bank))

  words = random.sample(word_bank, min(count, len(word_bank)))
  return words

bp = Blueprint("root", __name__)

@bp.route("/")
def root():
  return "Words API is working!"

@bp.route("/words")
def words():

  print(request.args, file=sys.stderr)

  theme = request.args.get('theme', type=str)
  count = request.args.get('count', type=int)
  allow_multiword = request.args.get('allow_multiword', True, type=bool)
  already_used_str = request.args.get('already_used', "", type=str)
  max_length = request.args.get('max_length', DEFAULT_WORD_MAX_LENGTH, type=int)

  already_used = already_used_str.split(',')

  print(allow_multiword, file=sys.stderr)

  return jsonify(pick_words(
    theme = theme,
    count = count,
    allow_multiword = allow_multiword,
    already_used = already_used,
    max_length=max_length))
