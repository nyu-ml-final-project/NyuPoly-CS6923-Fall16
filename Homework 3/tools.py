from collections import defaultdict
import re
import math
from unidecode import unidecode

spchars = re.compile('\`|\~|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\=|\\|\||\{|\[|\]|\}|\:|\;|\'|\"|\<|\,|\>|\?|\/|\.|\-')

# Utility function that does the following to the text:
# - Convert to unicode
# - Convert to lowercase
# - Remove special chars
def make_text_parsable(text):
    # convert to unicode
    text = unidecode(text) #.decode('utf-8', 'ignore'))
    # convert text to lowercase
    text = text.lower()
    # remove special characters
    text = spchars.sub(" ", text)
    return(text)

#
# Tokenize by whitespace. Use the defaultdict(int) whichsets the default 
# factory to int which makes it  the default dict useful for counting. 
#
def count_words(text, wc=None):
    if wc == None:
        wc = defaultdict(int)
    tokens = text.split(" ")
    for t in tokens:
        wc[t] += 1  
    return(wc)

#
# Main function. Opens the file and calls helper functions to parse
# Returns the sorted word count
#
def extract_info(filename):
    import json
    wc = defaultdict(int)
    df = defaultdict(set)
    count = 0
    with open(filename) as fin:
        for line in fin:
            count += 1
            current = json.loads(line)
            text = make_text_parsable(current["abstract"] + " " + \
                current["description"] + " " + current["title"])
            wc = count_words(text, wc)
    

    sorted_wc = sorted(wc.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_wc