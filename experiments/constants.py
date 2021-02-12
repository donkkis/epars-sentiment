STOP_WORDS = [
    'i',
    'me',
    'my',
    'myself',
    'we',
    'our',
    'ours',
    'ourselves',
    'you',
    "you're",
    "you've",
    "you'll",
    "you'd",
    'your',
    'yours',
    'yourself',
    'yourselves',
    'he',
    'him',
    'his',
    'himself',
    'she',
    "she's",
    'her',
    'hers',
    'herself',
    'it',
    "it's",
    'its',
    'itself',
    'they',
    'them',
    'their',
    'theirs',
    'themselves',
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    "that'll",
    'these',
    'those',
    'am',
    'is',
    'are',
    'was',
    'were',
    'be',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'do',
    'does',
    'did',
    'doing',
    'a',
    'an',
    'the',
    'and',
    'but',
    'if',
    'or',
    'as',
    'of',
    'at',
    'by',
    'for',
    'with',
    'about',
    'into',
    'through',
    'during',
    'to',
    'from',
    'in',
    'out',
    'on',
    'off',
    'then',
    'once',
    'here',
    'there',
    'when',
    'where',
    'why',
    'how',
    'both',
    'each',
    'other',
    'such',
    'own',
    'so',
    's',
    't',
    'can',
    'will',
    'just',
    'now',
    'd',
    'll',
    'm',
    'o',
    're',
    've',
    'y',
]

FREQUENT_POS = [
    'safety',
    'data',
    'study',
    'efficacy',
    'clinical',
    'patients',
    'considered',
    'treatment',
    'profile',
    'product'
 ]

FREQUENT_NEG = [
    'safety',
    'data',
    'patients',
    'study',
    'should',
    'treatment',
    'limited',
    'further',
    'address',
    'efficacy'
]

FREQUENT_NEUTR = [
    'studies',
    'safety',
    'study',
    'ct-p10',
    'efficacy',
    'data',
    'patients',
    'dose',
    'insulin',
    'product'
]

PHRASES_POS = [
    'based on the',
    'bioequivalence',
    'bioequivalent',
    'biosimilarity',
    'accepted by the chmp',
    'comparable',
    'these objectives have been met',
    'the available safety data are considered supportive'
]

PHRASES_NEG = [
    'should be provided',
    'data are considered very limited',
    'chmp considers the following measures',
]

FREQUENT = list(set(FREQUENT_NEG + FREQUENT_POS + FREQUENT_NEUTR))
PHRASES = PHRASES_POS + PHRASES_NEG

with open('../data/negative-words.txt', 'r') as f:
    SENTILEX_NEG = [line.strip('\n') for line in f if line and not line.startswith(';')]

with open('../data/positive-words.txt', 'r') as f:
    SENTILEX_POS = [line.strip('\n') for line in f if line and not line.startswith(';')]

SENTILEX = list(set(SENTILEX_POS + SENTILEX_NEG))
