import os 
import pandas as pd 
import json 

class_map_path = 'class_map.txt'

df = pd.read_csv(class_map_path,sep=' ',index_col='class')
# print(df)

# # with open(class_map_path,'r') as f:
# #     f.iter()
data_path = "./data/train"

for (root,dir,files) in os.walk(data_path):
    if len(dir) == 100:
        df_100 = df.loc[dir]


# df_100.to_csv("mini_imagenet_classes.csv") 

with open("imagenet-data.json",'r') as f:
 data = json.load(f)

data['mark'] = True
# print(data['children'])
class_list = df_100.index.tolist()

def dfs(curr,chk):
    if 'mark' not in curr.keys():
        curr['mark'] = False

    if curr['id'] == chk:
        curr['mark'] = True
        print("got hit!",chk, curr['name'])
        return True
    
    if 'children' in curr.keys():
        for x in curr['children']:
            flag = dfs(x,chk)
            if flag:
                curr['mark'] = True
                return True
    return False

for x in class_list:
    dfs(data,x)

print('marking done!')
def update_dfs(curr,new_curr):
    if 'mark' in curr.keys() and curr['mark']:
        print("adding ",curr['name'])
        new_curr = curr.copy()
        if 'children' in curr.keys():
            new_curr['children'] = []
            for x in curr['children']:
                obj = {}
                obj = update_dfs(x,obj)
                if len(obj.keys()) > 0:
                    new_curr['children'].append(obj)

        
    return new_curr
    
   

new_data = {}  
new_data = update_dfs(data,new_data)
print("updating done!")
with open('imagenet-100-data.json', 'w') as f:
    json.dump(new_data, f)

   