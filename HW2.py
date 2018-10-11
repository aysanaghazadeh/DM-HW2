import pandas as pd
import numpy as np
from sklearn import tree
import random
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


test = pd.read_csv("test.csv")
test.columns = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper-rolling', 'condition', 'formability', 'strength', 'non-aging', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phose', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/brigh/varn/clean', 'lustre', 'jurofm','s', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing']
train = pd.read_csv("train.csv")
test.replace({"?":None},inplace=True)
train.columns = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper-rolling', 'condition', 'formability', 'strength', 'non-aging', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phose', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/brigh/varn/clean', 'lustre', 'jurofm','s', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing','class']
train.replace({"?":None},inplace=True)
# print(train.count())


for i in train:
    if(train.count()[i])<100:
        del train[i]
        del test[i]



train['family'] = train['family'].fillna(pd.Series(np.random.choice(['TN', 'ZS'], p=[60/111, 51/111], size=len(train))))
test['family'] = test['family'].fillna(pd.Series(np.random.choice(['TN', 'ZS'], p=[60/111, 51/111], size=len(test))))
# print(train['family'])

count = {}
c = 0
for each in train["family"]:
    try:
        count[str(each)] = count[str(each)] + 1
    except:
        count[str(each)] = 1
        c+= 1
# print(count)


count = {}
c = 0
for each in train["product-type"]:
    try:
        count[str(each)] = count[str(each)] + 1
    except:
        count[str(each)] = 1
        c+= 1
# print(count)
list = []
for i in count:
    if not(str(i) == str(None)):
        list.append(i)

# print(list)

del train['product-type']
del test['product-type']

count = {}
c = 0
for each in train["steel"]:
    try:
        count[str(each)] = count[str(each)] + 1
    except:
        count[str(each)] = 1
        c+= 1
# print(count)
list = []
for i in count:
    if not(str(i) == str(None)):
        list.append(i)

train['steel'] = train['steel'].fillna(pd.Series(np.random.choice(['R', 'A', 'K', 'S', 'W', 'M', 'V'], p=[231/727, 396/727, 44/727, 9/727, 17/727, 17/727, 13/727], size=len(train))))
test['steel'] = test['steel'].fillna(pd.Series(np.random.choice(['R', 'A', 'K', 'S', 'W', 'M', 'V'], p=[231/727, 396/727, 44/727, 9/727, 17/727, 17/727, 13/727], size=len(test))))

del train['temper-rolling']
del test['temper-rolling']

train['condition'] = train['condition'].fillna(pd.Series(np.random.choice(['S', 'A'], p=[494/526, 32/526], size=len(train))))
test['condition'] = test['condition'].fillna(pd.Series(np.random.choice(['S', 'A'], p=[494/526, 32/526], size=len(test))))

train['formability'] = train['formability'].fillna(pd.Series(np.random.choice(['2', '3', '1', '5'], p=[338/515, 128/515, 40/515,9/515], size=len(train))))
test['formability'] = test['formability'].fillna(pd.Series(np.random.choice(['2', '3', '1', '5'], p=[338/515, 128/515, 40/515,9/515], size=len(test))))

train['surface-quality'] = train['surface-quality'].fillna(pd.Series(np.random.choice(['E', 'G', 'D', 'F'], p=[278/580, 199/580, 50/580, 53/580], size=len(train))))
test['surface-quality'] = test['surface-quality'].fillna(pd.Series(np.random.choice(['E', 'G', 'D', 'F'], p=[278/580, 199/580, 50/580, 53/580], size=len(test))))

del train['bf']
del test['bf']

train['bw/me'] = train['bw/me'].fillna(pd.Series(np.random.choice(['B', 'M'], p=[146/189, 43/189], size=len(train))))
test['bw/me'] = test['bw/me'].fillna(pd.Series(np.random.choice(['B', 'M'], p=[146/189, 43/189], size=len(test))))

del train['bl']
del test['bl']

family = pd.get_dummies(train["family"])
for key in family:
    train["family:"+key] = family[key]
del (train["family"])

family = pd.get_dummies(test["family"])
for key in family:
    test["family:"+key] = family[key]
del (test["family"])

steel = pd.get_dummies(train["steel"])
for key in steel:
    train["steel:"+key] = steel[key]
del (train["steel"])

steel = pd.get_dummies(test["steel"])
for key in steel:
    test["steel:"+key] = steel[key]
del (test["steel"])

condition = pd.get_dummies(train["condition"])
for key in condition:
    train["condition:"+key] = condition[key]
del (train["condition"])

condition = pd.get_dummies(test["condition"])
for key in condition:
    test["condition:"+key] = condition[key]
del (test["condition"])

formability = pd.get_dummies(train["formability"])
for key in formability:
    train["formability:"+key] = formability[key]
del (train["formability"])

formability = pd.get_dummies(test["formability"])
for key in formability:
    test["formability:"+key] = formability[key]
del (test["formability"])

surfaceQuality = pd.get_dummies(train["surface-quality"])
for key in surfaceQuality:
    train["surface-quality:"+key] = surfaceQuality[key]
del (train["surface-quality"])

surfaceQuality = pd.get_dummies(test["surface-quality"])
for key in surfaceQuality:
    test["surface-quality:"+key] = surfaceQuality[key]
del (test["surface-quality"])

bwme = pd.get_dummies(train["bw/me"])
for key in bwme:
    train["bw/me:"+key] = bwme[key]
del (train["bw/me"])

bwme = pd.get_dummies(test["bw/me"])
for key in bwme:
    test["bw/me:"+key] = bwme[key]
del (test["bw/me"])

shape = pd.get_dummies(train["shape"])
for key in shape:
    train["shape:"+key] = shape[key]
del (train["shape"])

shape = pd.get_dummies(test["shape"])
for key in shape:
    test["shape:"+key] = shape[key]
del (test["shape"])

output = train["class"]
input = train
del input["class"]
clf = tree.DecisionTreeClassifier()
clf.fit(input[0:683],output[0:683])
# print(clf.get_params())
# print(clf.score(input[684:796],output[684:796]))
# max_depth={}
# for i in range(1,100,3):
#     clf.set_params(max_depth = i)
#     clf.fit(input[0:683],output[0:683])
#     max_depth[i]=clf.score(input[684:796],output[684:796])
# print(max_depth)
# train.to_csv('/Users/aisanaghazade/DM-HW2/result.csv')
# print(train.count())

# min_samples_split={}
# for i in range(2,100,3):
#     clf.set_params(min_samples_split = i)
#     clf.fit(input[0:683],output[0:683])
#     min_samples_split[i]=clf.score(input[684:796],output[684:796])
# print(min_samples_split)
#
# criterion={}
# for i in ['gini' , 'entropy']:
#     clf.set_params(criterion = i)
#     clf.fit(input[0:683],output[0:683])
#     criterion[i]=clf.score(input[684:796],output[684:796])
# print(criterion)
#
#
# max_features={}
# for i in range(0,30):
#     clf.set_params(max_features = i+1)
#     clf.fit(input[0:683],output[0:683])
#     max_features[i+1]=clf.score(input[684:796],output[684:796])
# print(max_features)
#
# max_leaf_nodes={}
# for i in range(2,100,3):
#     clf.set_params(max_leaf_nodes = i)
#     clf.fit(input[0:683],output[0:683])
#     max_leaf_nodes[i]=clf.score(input[684:796],output[684:796])
# print(max_leaf_nodes)
#
# splitter={}
# for i in ['random','best']:
#     clf.set_params(splitter = i)
#     clf.fit(input[0:683],output[0:683])
#     splitter[i]=clf.score(input[684:796],output[684:796])
# print(splitter)

importance = {}
count = 0
importances = clf.feature_importances_
max = 0
min = 10
for i in train.columns.values:
    importance[i] = importances[count]
    if max <= importance[i]:
        max = importance[i]
        key = i
    if min >= importance[i] and importance[i] != 0:
        min = importance[i]
        key1 = i
    count += 1
print(key+":"+str(max))
print(key1+":"+str(min))
print(importance)



clf.set_params(class_weight = {'carbon': 0.075188922146897305, 'hardness': 0.1980617720548529, 'strength': 0.053046715496272032, 'thick': 0.211359886965322, 'width': 0.071308823735475688, 'len': 0.017466777928670223, 'bore': 0.0, 'family:TN': 0.068611371227633056, 'family:ZS': 0.0036277878992292943, 'steel:A': 0.054227687686137956, 'steel:K': 0.0, 'steel:M': 0.0041342312241929241, 'steel:R': 0.0048370505323057254, 'steel:S': 0.022062951857985163, 'steel:V': 0.012752049356432378, 'steel:W': 0.020384085909281119, 'condition:A': 0.013024163574426057, 'condition:S': 0.0030231565826910806, 'formability:1': 0.0066383240939091427, 'formability:2': 0.064305148431232334, 'formability:3': 0.0, 'formability:5': 0.0, 'surface-quality:D': 0.0070627005756202345, 'surface-quality:E': 0.028063623058253728, 'surface-quality:F': 0.0, 'surface-quality:G': 0.0028146630252641031, 'bw/me:B': 0.0096741010646114525, 'bw/me:M': 0.0096741010646114525, 'shape:COIL': 0.012334478857379601, 'shape:SHEET': 0.026315425651313042})
# print(clf.get_params())

# CV = StratifiedKFold(output, n_folds=6)
# gs = GridSearchCV(clf,scoring='accuracy',param_grid={'max_depth':[10,30,50,60],'min_samples_split':[5,10,15], 'max_features':[12,15,20,23], 'max_leaf_nodes':[40,50,70]},cv=CV)
# gs.fit(input, output)
# best = gs.best_params_
# print(best)
#
# clf.set_params(max_depth = 50, max_features = 20, max_leaf_nodes = 50, min_samples_split = 5)
# clf.fit(input[0:683],output[0:683])
print(clf.score(input[684:796],output[684:796]))

test["class"] = clf.predict(test)
test["class"].to_csv("result.csv")

