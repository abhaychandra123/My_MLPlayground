import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

def compute_entropy(y):
    entropy = 0.
    p=0
    for i in y:
         if int(i)==1:
            p+=1
    p=p/len(y)
    if p==0 or p==1:
        entropy=0
        return entropy
    
    entropy= -p*np.log2(p)-(1-p)*np.log2(1-p)
    return entropy

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if X[i][feature]==1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices
    

def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0
    if len(X_left)==0:
        information_gain=compute_entropy(y_node)-((len(X_right)/len(X_node))*compute_entropy(y_right))
    elif len(X_right)==0:
        information_gain=compute_entropy(y_node)-((len(X_left)/len(X_node))*compute_entropy(y_left))
    
    else:
        information_gain=compute_entropy(y_node)-((len(X_left)/len(X_node))*compute_entropy(y_left)+(len(X_right)/len(X_node))*compute_entropy(y_right))
    
    return information_gain
    
def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]

    best_feature = -1
    max_ig=0
    for i in range(num_features):
        ig=compute_information_gain(X, y, node_indices, feature=i)
        if ig>max_ig:
            max_ig=ig
            best_feature=i
#     if best_feature==0:
#         best_feature=-1

    return best_feature
#chking some thingss
print("Entropy at root node: ", compute_entropy(y_train)) 

best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

#finally using all above thingss to build our tree
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    # Agar Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
        
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)


#Apply the main funtion!
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
#DONE



