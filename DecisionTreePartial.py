import numpy as np
import pandas as pd
from pprint import pprint


dataset_normal = pd.read_csv("antiques.csv",
                      names=['Pre1800','RareMaker','RareType','GoodCond','Cheap','Buy',])
#print(type(dataset)) - is a dataframe

dataset_normal.columns = dataset_normal.columns.str.strip()

dataset_normal = dataset_normal*1

dataset = dataset_normal.sample(frac=1)

#print(dataset)

def entropy(target_col):

    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data,split_attribute_name,target_name="class"):

    total_entropy = entropy(data[target_name])

    vals,counts = np.unique(data[split_attribute_name],return_counts=True)

    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def tree_gen(data, originaldata, features, target_attribute_name="Buy", parent_node_class = None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]

    elif len(features) ==0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]

        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature:{}}

        features = [i for i in features if i != best_feature]


        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = tree_gen(sub_data, dataset, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return(tree)


def predict(query,tree,default = 1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


def train_test_split1(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split2(dataset):
    training_data = dataset.iloc[:50].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split3(dataset):
    training_data = dataset.iloc[:20].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split_one(dataset):
    training_data = dataset.iloc[:10].reset_index(drop=True)
    testing_data = dataset.iloc[150:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split_two(dataset):
    training_data = dataset.iloc[:10].reset_index(drop=True)
    testing_data = dataset.iloc[300:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split_three(dataset):
    training_data = dataset.iloc[:10].reset_index(drop=True)
    testing_data = dataset.iloc[450:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split_smallData(dataset):
    training_data = dataset.iloc[:5].reset_index(drop=True)
    testing_data = dataset.iloc[5:].reset_index(drop=True)
    return training_data,testing_data

def train_test_split_fullData(dataset):
    training_data = dataset.iloc[:500].reset_index(drop=True)
    testing_data = dataset.iloc[400:].reset_index(drop=True)
    return training_data,testing_data



training_data1 = train_test_split1(dataset)[0]
testing_data1 = train_test_split1(dataset)[1]
training_data2 = train_test_split2(dataset)[0]
testing_data2 = train_test_split2(dataset)[1]
training_data3 = train_test_split3(dataset)[0]
testing_data3 = train_test_split3(dataset)[1]
training_data_one = train_test_split_one(dataset)[0]
testing_data_one = train_test_split_one(dataset)[1]
training_data_two = train_test_split_two(dataset)[0]
testing_data_two = train_test_split_two(dataset)[1]
training_data_three = train_test_split_three(dataset)[0]
testing_data_three = train_test_split_three(dataset)[1]
training_data_small = train_test_split_smallData(dataset)[0]
testing_data_small = train_test_split_smallData(dataset)[1]
training_data_full = train_test_split_fullData(dataset)[0]
testing_data_full = train_test_split_fullData(dataset)[1]


def test(data,tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    data.reset_index(drop=True)
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0)
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["Buy"])/len(data))*100,'%')

tree1 = tree_gen(training_data1, training_data1, training_data1.columns[:-1])
tree2 = tree_gen(training_data2, training_data2, training_data2.columns[:-1])
tree3 = tree_gen(training_data3, training_data3, training_data3.columns[:-1])
tree_one = tree_gen(training_data_one, training_data_one, training_data_one.columns[:-1])
tree_two = tree_gen(training_data_two, training_data_two, training_data_two.columns[:-1])
tree_three = tree_gen(training_data_three, training_data_three, training_data_three.columns[:-1])
tree_small_data = tree_gen(training_data_small, training_data_small, training_data_small.columns[:-1])
tree_full_data = tree_gen(training_data_full, training_data_full, training_data_full.columns[:-1])

pprint(tree1)

test(testing_data1, tree1)
test(testing_data2, tree2)
test(testing_data3, tree3)
test(testing_data_one, tree_one)
test(testing_data_two, tree_two)
test(testing_data_three, tree_three)
test(testing_data_small, tree_small_data)
test(testing_data_full, tree_full_data)
