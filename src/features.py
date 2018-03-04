import re
import string
import unicodedata

import pandas as pd

from collections import Counter


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
