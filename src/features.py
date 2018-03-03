import re
import pandas as pd


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
