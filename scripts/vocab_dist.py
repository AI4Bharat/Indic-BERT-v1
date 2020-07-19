
import sys
import unicodedata as ud
import collections


def get_lang(w):
    try:
        if w[0] == '▁':
            lang = ud.name(w[1]).split()[0]
        else:
            lang = ud.name(w[0]).split()[0]
        return lang
    except:
        return 'unk'


fname = sys.argv[1]
words = open(fname).read().split('\n')
words = map(lambda w: w.split()[0] if w != '' else '', words)
words = filter(lambda w: '[' not in w, words)
words = map(lambda w: w.replace('#', ''), words)

langs = map(lambda w: get_lang(w), words)
counter = collections.Counter(langs)
counter = sorted(counter.items(), key=lambda k: -k[1])
counter = list(filter(lambda item: item[1] > 10, counter))

for k, v in counter:
    print(k, ": ", v)
