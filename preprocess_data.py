import pandas as pd
import csv
 



def read_comments(comments):

    for idx_2, comment in enumerate(comments):
        global total_comments, controversial_comments
        total_comments += 1
        item = {'text': "", 'ups': 0, 'downs': 0, 'score': 0, 'level': 0, 'controversy': 0, 'controversiality': 0, 'created_utc': 0, 'author': ''}
        # print(f"Comment index {idx_1}-{idx_2}: {comment}")
        if 'body' in comment:
            item['text'] = comment['body']
            # print(f"Body of comment: {comment['body']}")
        if 'ups' in comment:
            item['ups'] = comment['ups']
            # print(f"Ups in comment: {comment['ups']}")
        if 'downs' in comment:
            item['downs'] = comment['downs']
            # print(f"Downs in comment: {comment['downs']}")
        # print("\n\n\n")
        
        if item['ups'] > 0 and item['downs'] > 0:
            if item['ups'] > item['downs']:
                item['controversy'] = item['downs'] / item['ups']
            else:
                item['controversy'] = item['ups'] / item['downs']
        
        if 'controversiality' in comment:
            item['controversiality'] = comment['controversiality']
            controversial_comments += comment['controversiality']
            # print(f"Controversiality in comment: {comment['controversiality']}")

        if 'score' in comment:
            item['score'] = comment['score']
            # print(f"Score in comment: {comment['score']}")

        if 'created_utc' in comment:
            item['created_utc'] = comment['created_utc']
            # print(f"Created_utc in comment: {comment['created_utc']}")

        if 'author' in comment:
            item['author'] = comment['author']
            # print(f"Author in comment: {comment['author']}")

        if 'level' in comment:
            item['level'] = comment['level']
            # print(f"Level in comment: {comment['level']}")

        pd.DataFrame(item, index=[0]).to_csv('data.csv', sep=';', mode='a', header=False, index=False)

        if 'children' in comment:
            read_comments(comment['children'])
            # print(f"Children in comment: {comment['children']}")
            

def read_post(post):
    if 'comments' in post:
        for idx_1, comment_list in enumerate(post['comments']):
            read_comments(comment_list)

        
total_comments = 0
controversial_comments = 0

data = pd.read_json('threads.jsonl', orient="records", lines=True, chunksize=1000)
# print(data.shape)
for idx, post in enumerate(data):
    print(idx)
    read_post(post)

print(f"Total: {total_comments}")
print(f"Controversial: {controversial_comments}")