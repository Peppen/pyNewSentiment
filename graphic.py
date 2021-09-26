import matplotlib.pyplot as plt
import numpy as np


def neg_hate_graph():
    newspaper = 'CentreDaily'
    categories = ['Cronaca Rosa', 'Politica', 'Tecnologia', 'Cronaca Nera', 'Religione']
    pos = [212, 327, 99, 356, 151]
    neg = [71, 129, 15, 79, 107]
    total_neg = 0
    for i in range(len(neg)):
        total_neg += neg[i]
    hate = [17, 48, 2, 8, 22]
    x = np.arange(len(categories))
    plt.bar(x, pos, width=0.25, color='darkorchid', label='Positive Comments')
    plt.bar(x + 0.25, neg, width=0.25, color='red', label='Negative Comments')
    plt.bar(x + 0.25, hate, width=0.25, color='darkred', label='Hate Comments')
    for i in range(len(categories)):
        plt.annotate(pos[i], (i-0.13, pos[i]))
        plt.annotate(neg[i], (i+0.15, neg[i]))
        plt.annotate(hate[i], (i+0.16, hate[i]))
    plt.legend()
    plt.title('CentreDaily Hate Rate')
    plt.ylabel('Tweets')
    plt.xlabel('Categories')
    plt.savefig('graphs/' + newspaper + '.png')
    plt.show()


def negative():

    categories = ['TIME', 'BBCWorld', 'NYTimes', 'BklynEagle', 'BostonGlobe', 'CentreDaily', 'CitizensVoice', 'KCStar', 'LADailyNews', 'NewsDay', ' NorthBayNews', 'NYPost', 'SeaTimes', 'PhillyNews', 'Tennessean', 'TheSun', 'ThePlainDealer', 'USAToday', 'WashTimes', 'WashingtonPost', 'WSJ']
    comments = [1637, 1444, 1558, 1756, 1811, 1546, 1652, 1598, 1499, 1670, 1543, 1765, 1801, 1698, 1714, 1784, 1652, 1871, 1709, 1552, 1693]
    neg =      [381,  309,  343,  287,  395,  401,  321,  342,  355,  361,  425,  361,  384,  375,  393,  278,  299,  305,  322,  380,  396]
    pos =      [1256, 1135, 1215, 1469, 1416, 1145, 1331, 1256, 1144, 1309, 1118, 1404, 1417, 1323, 1321, 1506, 1353, 1566, 1387, 1172, 1397]
    x = np.arange(len(comments))
    plt.bar(x, comments, width=0.25, color='darkblue', label='Average Comments')
    plt.bar(x + 0.25, pos, width=0.25, color='darkorchid', label='Average Positive Comments')
    plt.bar(x + 0.5, neg, width=0.25, color='r', label='Average Negative Comments')
    for i in range(len(comments)):
        plt.annotate(comments[i], (i - 0.1, comments[i]))
        plt.annotate(pos[i], (i + 0.14, pos[i]))
        plt.annotate(neg[i], (i + 0.4, neg[i]))
    plt.title('Positive-Negative Rate')
    plt.ylabel('Tweets')
    plt.xlabel('Newspapers')
    plt.xticks([i + 0.25 for i in range(21)], categories)
    plt.legend()
    plt.savefig('graphs/Positive-Negative.png')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    neg_hate_graph()
