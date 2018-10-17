import json
from pprint import pprint



data1 = {
    'name' : 'ACME',
    'shares' : 100,
    'price' : 542.23,
}


data = {
    'name' : 'ACME',
    'shares' : 100,
    'price' : 542.23,
    'data': data1,
}




# Writing JSON data
with open('data.json', 'w') as f:
    json.dump(data, f)

# Reading data back
with open('data.json', 'r') as f:
    data = json.load(f)

with open('print.txt', 'w') as f:
	json.dump(data, f)



