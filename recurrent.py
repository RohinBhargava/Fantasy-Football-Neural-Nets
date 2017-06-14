import tensorflow as tf, pandas as pd, numpy as np, math, random
from tensorflow.contrib import rnn

team_dict = {'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GNB': 'Green Bay Packers', 'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KAN': 'Kansas City Chiefs', 'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings', 'NOR': 'New Orleans Saints', 'NWE': 'New England Patriots', 'NYG': 'New York Giants', 'NYJ': 'New York Jets', 'OAK': 'Oakland Raiders', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SDG': 'San Diego Chargers', 'SEA': 'Seattle Seahawks', 'SFO': 'San Francisco 49ers', 'STL': 'St. Louis Rams', 'TAM': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Redskins'}
pos_dict = {'QB': 2, 'RB': 4, 'WR': 3, 'TE': 1}

def rosterize(year1, year2):
    y1 = pd.read_csv('Data/' + str(year1) + '.csv')
    y2 = pd.read_csv('Data/' + str(year2) + '.csv')
    roster_data = dict()
    ranks = []
    y1codes = [i.split('\\')[1] for i in y1['Name']]
    y2codes = [i.split('\\')[1] for i in y2['Name']]
    table = y1[['Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att', 'Yds', 'Y/A', 'TD', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', 'FantPt', 'DKPt', 'FDPt', 'PosRank']]
    ctr = 0
    for i in y1codes:
        if i in y2codes:
            a = list(table.ix[ctr]) + [0]
            for b in range(len(a)):
                if type(a[b]) == float and math.isnan(a[b]):
                    a[b] = 0.0
            roster_data[i] = a
            val = y2['FantPt'][y2codes.index(i)]
            if math.isnan(val):
                val = 0.0
            ranks.append((i, val))
        ctr += 1
    raw_ranks = [i for i in sorted(ranks, key=lambda x: x[1], reverse=True)][:425]
    divisor = sum([x[1] for x in raw_ranks])
    return roster_data, [(j[0], j[1]/divisor) for j in raw_ranks], divisor

def rank(year):
    nfc = pd.read_csv('Data/NFC' + str(year) + '.csv')
    afc = pd.read_csv('Data/AFC' + str(year) + '.csv')
    ranks = []
    i = 0
    j = 0
    while i < len(nfc['Tm']) or j < len(afc['Tm']):
        if i == len(nfc['Tm']):
            ranks.append(afc['Tm'][j])
            j += 1
        elif j == len(afc['Tm']):
            ranks.append(nfc['Tm'][i])
            i += 1
        elif afc['W-L%'][j] > nfc['W-L%'][i]:
            ranks.append(afc['Tm'][j])
            j += 1
        elif afc['W-L%'][j] == nfc['W-L%'][i]:
            if afc['SRS'][j] > nfc['SRS'][i]:
                ranks.append(afc['Tm'][j])
                j += 1
            else:
                ranks.append(nfc['Tm'][i])
                i += 1
        else:
            ranks.append(nfc['Tm'][i])
            i += 1
    return ranks

sess = tf.InteractiveSession()
roster_data = []
rankings = []
totals = []
for year in range(2012, 2016):
    roster_datum, ranking, total = rosterize(year, year + 1)
    team_rankings = rank(year)
    for i in roster_datum:
        if roster_datum[i][0] in team_dict:
            roster_datum[i][0] = team_rankings.index(team_dict[roster_datum[i][0]])
        else:
            roster_datum[i][0] = 0
        if roster_datum[i][1] in pos_dict:
            roster_datum[i][1] = pos_dict[roster_datum[i][1]]
        else:
            roster_datum[i][1] = 0
    roster_data.append(roster_datum)
    rankings.append(ranking)
    totals.append(total)

x = tf.placeholder(tf.float32, shape=[None, 12, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def RNN(x, weights, biases):
    x = tf.unstack(x, 12, 1)
    lstm_cell = rnn.BasicLSTMCell(69, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights) + biases

W_out = weight_variable([69, 1])
b_out = bias_variable([1])

y = RNN(x, W_out, b_out)
cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

sess.run(tf.global_variables_initializer())

def is_list(x, y):
    j = 0
    for i in y:
        if x == y:
            return j
        j += 1
    return 0

accuracyl = []
for b in range(100):
    for i in range(len(rankings) - 1):
        random.shuffle(rankings[i])
        batches = [rankings[i][25 * j: 25 * (j + 1)] for j in range(17)]
        for l in batches:
            train_step.run(feed_dict={x: np.array([roster_data[i][j[0]] for j in l]).reshape(25, 12, 2), y_: np.array([[j[1]] for j in l])})
    accuracy = tf.reduce_mean(tf.cast(abs(y - y_) * np.mean(totals) < 30, tf.float32))
    accuracyl.append(accuracy.eval(feed_dict={x: np.array([roster_data[-1][i[0]] for i in rankings[-1]]).reshape(len(rankings[-1]), 12, 2), y_: np.array([[i[1]] for i in rankings[-1]])}))
    print "Epoch:", b, "Accuracy:", accuracyl[-1]
print "Max Accuracy:", max(accuracyl), "Epoch:", accuracyl.index(max(accuracyl))
