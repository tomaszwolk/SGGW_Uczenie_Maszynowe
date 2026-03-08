
import re
import nltk
from nltk.corpus import stopwords

# Ensure nltk data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def prepareNewsText(text_data):
    """
    Optimized version of prepareNewsText.
    - Loads stopwords set ONCE outside the loop.
    - Uses set lookup (O(1)) instead of list lookup (O(N)).
    """
    finalText = []
    # Load stopwords once and convert to set for fast lookup
    stops = set(stopwords.words('english'))
    
    for sentence in text_data:
        # Remove unwanted formatting/signs
        sentence = re.sub(r'[^\w\s]', '', sentence) 
        
        # Split, lower, check stopwords, and join
        # Note: checking token.lower() in stops set
        finalText.append(' '.join(token.lower()
                                for token in str(sentence).split()
                                if token.lower() not in stops))
    return finalText
