import re
import os
import time
import string
import threading
import unicodedata
import unidecode

import pandas as pd
import numpy as np

from collections import Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

from joblib import Parallel, delayed


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


def apply_corrections(df, vectors):
    alphabet = string.ascii_lowercase
    corrector_cache = {}

    fasttext_voc = set(line.split()[0] for line in open(vectors))
    clean_voc = Counter(w for line in df['comment_text'] for w in re.findall(r'(?u)\b\w\w+\b', line) if w in fasttext_voc)

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
        if isinstance(line, float):  # skip nan
            return line

        words = re.findall(r'(?u)\b\w\w+\b', line)
        words = [sw for w in words for sw in correct(w)]
        return ' '.join(words)

    return df.applymap(correct_line)


def clean2_corrected_fasttext(clean2):
    return apply_corrections(clean2, 'input/crawl-300d-2M.vec')


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


def translate(comment, language):
    from textblob import TextBlob
    from textblob.translate import NotTranslated

    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")

    text = TextBlob(comment)
    try:
        text = text.translate(to=language)
        text = text.translate(to="en")
    except NotTranslated:
        pass

    return str(text)


def multilang(raw):
    df = raw.copy()
    langs = ['de', 'fr', 'es']

    if False:
        parallel = Parallel(300, backend="threading", verbose=5)
        for language in langs:
            print('Translate comments using "{0}" language'.format(language))
            df['comment_text__%s' % language] = parallel(delayed(translate)(comment, language) for comment in raw['comment_text'])
    else:
        for language in langs:
            df['comment_text__%s' % language] = pd.read_csv('input/train_%s.csv' % language, index_col='id')['comment_text'].loc[df.index]

    return df


def multilang_clean3(multilang):
    url_pattern = re.compile(r"""(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")

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

    def clean(line):
        if isinstance(line, float): # skip nan
            return line

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

        for pattern, repl in expand_patterns:
            line = re.sub(pattern, repl, line)

        return line

    return multilang.applymap(clean)


def multilang_clean3_corrected_fasttext(multilang_clean3):
    return apply_corrections(multilang_clean3, 'input/crawl-300d-2M.vec')


def multilang_clean4(multilang):
    url_pattern = re.compile(r"""(?i)\b((?:[hf][\w-]{2,3}:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))""")

    expand_patterns = [
        (r'US', 'United States'),
        (r'IT', 'Information Technology'),
        (r'wasn\'?t', 'was not'),
        (r'you\'?re', 'you are'),
        (r'won\'?t', 'will not'),
        (r'can\'?t', 'can not'),
        (r'i\'?m', 'i am'),
        (r'ain\'?t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would'),
        (r'\bf[*@$]+k', 'fuck'),
        (r'\bsu?[*@$]+', 'suck'),
        (r'\bf[*@$]+\b', 'fuck'),
        (r'\bf[*@$]+i', 'fucki'),
    ]

    def clean(line):
        if isinstance(line, float):  # skip nan
            return line

        line = unicodedata.normalize('NFKD', line.lower().replace('\xad', ''))  # normalize unicode
        line = line.encode("ascii", errors="ignore").decode()  # remove non-ascii
        line = re.sub(r'\b\w([^\w])\w(\1\w)+\b', lambda m: re.sub('\W', '', m.group()), line)

        line = re.sub(r'[\s\n\t_]+', ' ', ' ' + line + ' ')  # replace sequence of spacing symbols with single space
        line = re.sub(r'\[\d+\]', '', line)  # remove wiki references
        line = re.sub(r'user:\w+', ' user ', line)  # replace user:username
        line = line.replace('[talk]', ' ').replace('(talk)', ' ')

        line = re.sub(url_pattern, ' url ', line)  # urls

        line = re.sub(r'([0-9a-f]+:+)[0-9a-f]+', ' ', line)  # ipv6 addresses
        line = re.sub(r'([0-9]+\.+){2,}[0-9]+', ' ', line)  # ipv4 addresses

        line = re.sub(r'(\d)([^\d])', '\\1 \\2', line)  # split 5million
        line = re.sub(r'([^\d])(\d)', '\\1 \\2', line)  # split wikipedia86
        line = re.sub(r'\d+(\s+\d+)*', '0', line)  # replace big numerics with 00

        line = re.sub(r'(.)\1{2,}', r'\1', line)  # replace identical consecutive characters

        for pattern, repl in expand_patterns:
            line = re.sub(pattern, repl, line)

        return line

    return multilang.applymap(clean)


def apply_corrections2(df, vectors):
    alphabet = string.ascii_lowercase
    corrector_cache = {}

    fasttext_voc = set(line.split()[0] for line in open(vectors))
    clean_voc = Counter(w for line in df['comment_text'] for w in re.findall(r'(?u)\b\w\w*\b', line) if w in fasttext_voc)

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

        if len(word) < 20:
            corrs, corr_freq = None, 0
            for cand in edits1(word):
                cand_freq = clean_voc.get(cand, 0)
                if cand_freq > corr_freq:
                    corrs = [cand]
                    corr_freq = cand_freq

            if corrs is not None:
                corrector_cache[word] = corrs
                return corrs

        if len(word) < 15:
            for precand in edits1(word):
                for cand in edits1(precand):
                    cand_freq = clean_voc.get(cand, 0)
                    if cand_freq > corr_freq:
                        corrs = [cand]
                        corr_freq = cand_freq

            if corrs is not None:
                corrector_cache[word] = corrs
                return corrs

        if len(word) > 4 and len(word) < 50:
            for sz in reversed(range(1, min(len(word) - 1, 10))):
                if clean_voc.get(word[:sz], 0) > 100:
                    corrs = [word[:sz]] + correct(word[sz:])
                    corrector_cache[word] = corrs
                    return corrs

        if len(word) > 20:
            corrector_cache[word] = []
            return []

        corrs = [word]
        corrector_cache[word] = corrs
        return corrs

    def correct_line(line):
        if isinstance(line, float):  # skip nan
            return line

        words = re.findall(r'(?u)\b\w\w*\b', line)
        words = [sw for w in words for sw in correct(w)]
        return ' '.join(words)

    return df.applymap(correct_line)


def multilang_clean4_corrected_fasttext(multilang_clean4):
    return apply_corrections2(multilang_clean4, 'input/crawl-300d-2M.vec')


def multilang_clean4_corrected_twitter(multilang_clean4):
    return apply_corrections2(multilang_clean4, 'input/glove.twitter.27B.200d.txt')


def multilang_clean4_bpe50k(multilang_clean4):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("input/en.wiki.bpe.op50000.model")

    def apply_bpe(line):
        if isinstance(line, float):  # skip nan
            return line

        line = re.sub(r'\s+', ' ', line).lower().strip()
        return ' '.join(sp.EncodeAsPieces(line))

    return multilang_clean4.applymap(apply_bpe)


def multilang_clean4_bpe25k(multilang_clean4):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("input/en.wiki.bpe.op25000.model")

    def apply_bpe(line):
        if isinstance(line, float):  # skip nan
            return line

        line = re.sub(r'\s+', ' ', line).lower().strip()
        return ' '.join(sp.EncodeAsPieces(line))

    return multilang_clean4.applymap(apply_bpe)


def multilang_clean4_bpe10k(multilang_clean4):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load("input/en.wiki.bpe.op10000.model")

    def apply_bpe(line):
        if isinstance(line, float):  # skip nan
            return line

        line = re.sub(r'\s+', ' ', line).lower().strip()
        return ' '.join(sp.EncodeAsPieces(line))

    return multilang_clean4.applymap(apply_bpe)


def atanas(raw):
    tr = pd.read_csv('input/train_atanas.csv', index_col='id')[['comment_text']]
    te = pd.read_csv('input/test_atanas.csv', index_col='id')
    return pd.concat([tr, te]).fillna('').loc[raw.index]


def api1(raw):
    data = pd.concat((pd.read_csv('input/meta_train_from_api.csv'), pd.read_csv('input/meta_test_from_api.csv')))
    data.drop(['text'], axis=1, inplace=True)

    return pd.DataFrame(data.values, columns=data.columns, index=raw.index)


def api2_raw(raw):
    from googleapiclient import discovery

    tls = threading.local()
    eos_pattern = re.compile(r"[!?.]\s+")
    eow_pattern = re.compile(r"\s+")

    def aggregate_api_responses(responses):
        if any(r is None for r in responses):
            return None

        keys = ['UNSUBSTANTIAL', 'LIKELY_TO_REJECT', 'OBSCENE', 'SEVERE_TOXICITY', 'TOXICITY', 'INFLAMMATORY', 'SPAM', 'ATTACK_ON_AUTHOR', 'INCOHERENT', 'ATTACK_ON_COMMENTER']
        res = {}

        for k in keys:
            vals = [r[k] for r in responses]
            res[k] = {
                'spanScores': sum([v['spanScores'] for v in vals], []),
                'summaryScore': {'type': 'PROBABILITY', 'value': (sum(v['summaryScore']['value'] for v in vals) / len(vals))},
                'parts': vals
            }

        return res

    def split_text(text, min_len, max_len):
        parts = []

        pos = 0
        while len(text) - pos > max_len:
            split_match = eos_pattern.search(text, pos + min_len, pos + max_len)
            if split_match is None:
                split_match = eow_pattern.search(text, pos + min_len, pos + max_len)
            if split_match is None:
                split_pos = pos + min_len
            else:
                split_pos = split_match.end()
            parts.append(text[pos:split_pos])
            pos = split_pos

        parts.append(text[pos:])
        return parts

    def get_api_response(text):
        if len(text) > 2900:
            return aggregate_api_responses([get_api_response(p) for p in split_text(text, 2000, 2900)])

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'ATTACK_ON_AUTHOR': {},
                'ATTACK_ON_COMMENTER': {},
                'INCOHERENT': {},
                'INFLAMMATORY': {},
                'LIKELY_TO_REJECT': {},
                'OBSCENE': {},
                'SPAM': {},
                'UNSUBSTANTIAL': {}
            }
        }

        try:
            if not hasattr(tls, 'service'):
                tls.service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=os.getenv('API_KEY'))

            time.sleep(1.0)
            response = tls.service.comments().analyze(body=analyze_request).execute()
            return response['attributeScores']
        except Exception as e:
            print(e)
            return None

    with Parallel(10, backend="threading", verbose=5) as parallel:
        if os.path.exists('tmp/state.pickle'):
            df = pd.read_pickle('tmp/state.pickle')
        else:
            df = pd.DataFrame(index=raw.index)
            df['api_response'] = None

        index = df[df['api_response'].isnull()].index

        batch_size = 5000
        for ofs in range(0, len(index), batch_size):
            idx = index[ofs:ofs+batch_size]
            df.loc[idx, 'api_response'] = parallel(delayed(get_api_response, check_pickle=False)(comment) for comment in raw.loc[idx, 'comment_text'])
            df.to_pickle('tmp/state.pickle')
            print("Saved state on step %d" % ofs)

    return df


def api2(api2_raw):
    def extract_features(resp):
        if resp is None:
            return {}

        res = {}
        for k in resp.keys():
            res['%s_summary' % k] = resp[k]['summaryScore']['value']
            res['%s_min' % k] = min(span['score']['value'] for span in resp[k]['spanScores'])
            res['%s_max' % k] = min(span['score']['value'] for span in resp[k]['spanScores'])
            res['%s_mean' % k] = np.mean([span['score']['value'] for span in resp[k]['spanScores']])
            res['%s_std' % k] = np.std([span['score']['value'] for span in resp[k]['spanScores']])

        return res

    records = []
    for resp in api2_raw['api_response']:
        records.append(extract_features(resp))

    return pd.DataFrame.from_records(records, index=api2_raw.index)



def api3(raw):
    def extract_features(resp):
        if isinstance(resp, str):
            return {}

        res = {}
        for k in resp.keys():
            res['%s_summary' % k] = resp[k]['summaryScore']['value']
            res['%s_min' % k] = min(span['score']['value'] for span in resp[k]['spanScores'])
            res['%s_max' % k] = min(span['score']['value'] for span in resp[k]['spanScores'])

        return res

    records = []
    for filename in ['input/new_train_api.pickle', 'input/new_test_api.pickle']:
        for line in pd.read_pickle(filename):
            records.append(extract_features(line[1]))

    return pd.DataFrame.from_records(records, index=raw.index)


def api3_2(raw):
    def extract_features(resp):
        if isinstance(resp, str):
            return {}

        res = {}
        for k in resp.keys():
            res['%s_summary' % k] = resp[k]['summaryScore']['value']
            res['%s_min' % k] = min(span['score']['value'] for span in resp[k]['spanScores'])
            res['%s_max' % k] = min(span['score']['value'] for span in resp[k]['spanScores'])
            res['%s_mean' % k] = np.mean([span['score']['value'] for span in resp[k]['spanScores']])
            res['%s_std' % k] = np.std([span['score']['value'] for span in resp[k]['spanScores']])

        return res

    records = []
    for filename in ['input/new_train_api.pickle', 'input/new_test_api.pickle']:
        for line in pd.read_pickle(filename):
            records.append(extract_features(line[1]))

    return pd.DataFrame.from_records(records, index=raw.index)
