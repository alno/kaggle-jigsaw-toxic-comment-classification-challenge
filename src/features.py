import re
import string
import unicodedata

import pandas as pd
import numpy as np

from collections import Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


def count_regexp_occ(regexp, text):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))



def clean1(raw):
    def clean(line):
        line = re.sub('[\s\n\t_]+', ' ', ' ' + line.lower() + ' ')  # replace sequence of spacing symbols with single space

        line = re.sub('([0-9a-f]+:+)+[0-9a-f]+', 'iptoken', line)  # ipv6 addresses
        line = re.sub('([0-9]+\\.+)+[0-9]+', 'iptoken', line)  # ipv4 addresses

        line = re.sub('(\d)([^\d])', '\\1 \\2', line)  # split 5million
        line = re.sub('([^\d])(\d)', '\\1 \\2', line)  # split wikipedia86
        line = re.sub('\d\d+', '00', line)  # replace big numerics with 00
        return line.strip()

    return raw.applymap(clean)


def clean2(raw):
    url_pattern = re.compile(r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")

    def clean(line):
        line = re.sub(r'[\s\n\t_]+', ' ', ' ' + line.lower() + ' ')  # replace sequence of spacing symbols with single space
        line = line.replace('\xad', '')
        line = re.sub(r'\[\d+\]', '', line)  # remove wiki references
        line = unicodedata.normalize('NFKD', line)  # normalize unicode
        line = line.encode("ascii", errors="ignore").decode()  # remove non-ascii

        line = re.sub(url_pattern, ' urltoken ', line)  # urls

        line = re.sub(r'([0-9a-f]+:+)[0-9a-f]+', ' iptoken ', line)  # ipv6 addresses
        line = re.sub(r'([0-9]+\.+){2,}[0-9]+', ' iptoken ', line)  # ipv4 addresses

        line = re.sub(r'(\d)([^\d])', '\\1 \\2', line)  # split 5million
        line = re.sub(r'([^\d])(\d)', '\\1 \\2', line)  # split wikipedia86
        line = re.sub(r'\d\d+', '00', line)  # replace big numerics with 00

        line = re.sub(r'(.)\1{2,}', r'\1', line)  # replace identical consecutive characters

        return line

    return raw.applymap(clean)


def clean2_corrected_fasttext(clean2):
    alphabet = string.ascii_lowercase
    corrector_cache = {}

    fasttext_voc = set(line.split()[0] for line in open('input/crawl-300d-2M.vec'))
    clean_voc = Counter(w for line in clean2['comment_text'] for w in re.findall(r'(?u)\b\w\w+\b', line) if w in fasttext_voc)

    def edits1(word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        inserts = [a + c + b for a, b in splits for c in alphabet]
        return (deletes + transposes + replaces + inserts)

    def correct(word):
        if word in clean_voc:
            return [word]

        corrs = corrector_cache.get(word)
        if corrs is not None:
            return corrs

        corrs, corr_freq = None, 0
        for cand in edits1(word):
            cand_freq = clean_voc.get(cand, 0)
            if cand_freq > corr_freq:
                corrs = [cand]
                corr_freq = cand_freq

        if corrs is not None:
            corrector_cache[word] = corrs
            return corrs

        if len(word) > 5 and len(word) < 30:
            for sz in reversed(range(1, min(len(word) - 1, 10))):
                if clean_voc.get(word[:sz], 0) > 100:
                    corrs = [word[:sz]] + correct(word[sz:])
                    corrector_cache[word] = corrs
                    return corrs

        corrs = [word]
        corrector_cache[word] = corrs
        return corrs

    def correct_line(line):
        words = re.findall(r'(?u)\b\w\w+\b', line)
        words = [sw for w in words for sw in correct(w)]
        return ' '.join(words)

    return clean2.applymap(correct_line)


def clean2_no_punct(clean2):
    def rm_punct(x):
        return re.sub(r'[^\w\s]', ' ', x)

    return clean2.applymap(rm_punct)


def clean2_expand_no_punct(clean2):
    expand_patterns = [
        (r'US', 'United States'),
        (r'IT', 'Information Technology'),
        (r'(W|w)on\'t', 'will not'),
        (r'(C|c)an\'t', 'can not'),
        (r'(I|i)\'m', 'i am'),
        (r'(A|a)in\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would'),
    ]

    def expand_rm_punct(x):
        for pattern, repl in expand_patterns:
            x = re.sub(pattern, repl, x)
        return re.sub(r'[^\w\s]', ' ', x)

    return clean2.applymap(expand_rm_punct)


def clean2_expand_no_punct_lemmatize(clean2_expand_no_punct):
    from nltk.stem import WordNetLemmatizer

    wordnet_lemmatizer = WordNetLemmatizer()

    def lemmatize(x):
        return ' '.join(map(wordnet_lemmatizer.lemmatize, x.split()))

    return clean2_expand_no_punct.applymap(lemmatize)


def num1(raw):
    def cap_ratio(line):
        line = re.sub('\W', '', line)
        return len(re.sub('[^A-Z]', '', line)) / (len(line) + 1.0)

    def exq_ratio(line):
        line = re.sub('[^\w!?]', '', line)
        return len(re.sub('[^!?]', '', line)) / (len(line) + 1.0)

    df = pd.DataFrame(index=raw.index)
    df['cap_ratio'] = raw['comment_text'].map(cap_ratio)
    df['exq_ratio'] = raw['comment_text'].map(exq_ratio)
    return df


def num2(clean2):
    res = []
    for line in clean2['comment_text']:
        sents = tokenize.sent_tokenize(line)
        words = re.findall(r'(?u)\b\w\w+\b', line)

        res.append(dict(
            num_sents=len(sents),
            num_words=len(words),
            mean_sent_len=np.mean(list(map(len, sents))) if len(sents) > 0 else 0,
            mean_sent_len_words=np.mean([len(re.findall(r'(?u)\b\w\w+\b', s)) for s in sents]) if len(sents) > 0 else 0,
            mean_word_len=np.mean(list(map(len, words))) if len(words) > 0 else 0,
            uniq_word_ratio=len(set(words)) / (len(words) + 1e-3),
        ))

    return pd.DataFrame.from_records(res, index=clean2.index)


def sentiment1(raw):
    analyzer = SentimentIntensityAnalyzer()
    res = []
    for text in raw['comment_text']:
        res.append(analyzer.polarity_scores(text))
    return pd.DataFrame.from_records(res, index=raw.index)


def clean2_bpe50k(clean2):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("input/en.wiki.bpe.op50000.model")

    def apply_bpe(line):
        line = re.sub(r'\s+', ' ', line).lower().strip()
        return ' '.join(sp.EncodeAsPieces(line))

    return clean2.applymap(apply_bpe)


def clean2_bpe25k(clean2):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("input/en.wiki.bpe.op25000.model")

    def apply_bpe(line):
        line = re.sub(r'\s+', ' ', line).lower().strip()
        return ' '.join(sp.EncodeAsPieces(line))

    return clean2.applymap(apply_bpe)


def clean2_bpe10k(clean2):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("input/en.wiki.bpe.op10000.model")

    def apply_bpe(line):
        line = re.sub(r'\s+', ' ', line).lower().strip()
        return ' '.join(sp.EncodeAsPieces(line))

    return clean2.applymap(apply_bpe)


def ind1(raw):
    text = raw["comment_text"]
    df = pd.DataFrame(index=raw.index)

    # Count number of \n
    df["ant_slash_n"] = text.apply(lambda x: count_regexp_occ(r"\n", x))
    # Get length in words and characters
    df["raw_word_len"] = text.apply(lambda x: len(x.split()))
    df["raw_char_len"] = text.apply(lambda x: len(x))
    # Check number of upper case, if you're angry you may write in upper case
    df["nb_upper"] = text.apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    # Number of F words - f..k contains folk, fork,
    df["nb_fk"] = text.apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    # Number of S word
    df["nb_sk"] = text.apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    # Number of D words
    df["nb_dk"] = text.apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    # Number of occurence of You, insulting someone usually needs someone called : you
    df["nb_you"] = text.apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    # Just to check you really refered to my mother ;-)
    df["nb_mother"] = text.apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    # Just checking for toxic 19th century vocabulary
    df["nb_ng"] = text.apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    # Some Sentences start with a <:> so it may help
    df["start_with_columns"] = text.apply(lambda x: count_regexp_occ(r"^\:+", x))
    # Check for time stamp
    df["has_timestamp"] = text.apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    # Check for dates 18:44, 8 December 2010
    df["has_date_long"] = text.apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    # Check for date short 8 December 2010
    df["has_date_short"] = text.apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    # Check for http links
    df["has_http"] = text.apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    # check for mail
    df["has_mail"] = text.apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    # Looking for words surrounded by == word == or """" word """"
    df["has_emphasize_equal"] = text.apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = text.apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

    return df

