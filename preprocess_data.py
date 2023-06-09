import pandas as pd
 
 # Write list to csv according to year
def write_csv(new_data):
    if new_data['created_utc'] > 1483225200:
        filename = "data_2017.csv"
    elif new_data['created_utc'] > 1451602800:
        filename = "data_2016.csv"
    elif new_data['created_utc'] > 1420076400:
        filename = "data_2015.csv"
    elif new_data['created_utc'] > 1388449200:
        filename = "data_2014.csv"
    elif new_data['created_utc'] > 1356914400:
        filename = "data_2013.csv"
    elif new_data['created_utc'] > 1325287200:
        filename = "data_2012.csv"
    elif new_data['created_utc'] > 1293760800:
        filename = "data_2011.csv"
    elif new_data['created_utc'] > 1262233200:
        filename = "data_2010.csv"
    elif new_data['created_utc'] > 1230606000:
        filename = "data_2009.csv"
    elif new_data['created_utc'] > 1199071200:
        filename = "data_2008.csv"
    elif new_data['created_utc'] > 1167543600:
        filename = "data_2007.csv"

    pd.DataFrame(new_data, index=[0]).to_csv(filename, sep=';', mode='a', header=False, index=False)

# Read comments recursively
def read_comments(comments):

    for idx_2, comment in enumerate(comments):
        global total_comments, controversial_comments
        total_comments += 1
        item = {'id': "", 'text': "", 'controversiality': 0, 'created_utc': 0, 'author': ''}

        if 'id' in comment:
            item['id'] = comment['id']
        if 'body' in comment:
            item['text'] = comment['body']
        
        if 'controversiality' in comment:
            item['controversiality'] = comment['controversiality']
            controversial_comments += comment['controversiality']

        if 'created_utc' in comment:
            item['created_utc'] = int(comment['created_utc'])

        if 'author' in comment:
            item['author'] = comment['author']

        write_csv(item)

        if 'children' in comment:
            read_comments(comment['children'])
            

def read_post(post):
    if 'comments' in post:
        for idx_1, comment_list in enumerate(post['comments']):
            read_comments(comment_list)

        
total_comments = 0
controversial_comments = 0

# Chunkwise read jsonl file
data = pd.read_json('threads.jsonl', orient="records", lines=True, chunksize=1000)

for idx, post in enumerate(data):
    print(idx)
    read_post(post)

print(f"Total: {total_comments}")
print(f"Controversial: {controversial_comments}")