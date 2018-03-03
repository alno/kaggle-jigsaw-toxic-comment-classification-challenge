import re


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
