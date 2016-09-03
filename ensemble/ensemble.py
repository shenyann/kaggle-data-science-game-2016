import sys
import numpy as np

print 'Id,label'
for i in sys.stdin:
    terms = i.split(' ')
    score = np.zeros(5)
    counts = [ int(terms[1]), int(terms[4]), int(terms[7]), int(terms[10])]
    score[int(terms[1])] += float(terms[2])
    score[int(terms[4])] += float(terms[5])
    score[int(terms[7])] += float(terms[8])
    score[int(terms[10])] += 0.81729
    am = score.argmax()
    print "%s,%s"%(terms[0], am)

