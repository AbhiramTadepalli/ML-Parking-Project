import boto3  
import json  

dynamodb = boto3.resource('dynamodb')  
table = dynamodb.Table('UTDParkingMonitorTable')  
response = table.scan()  
data = response['Items']  
print(f"Fetched {len(data)} items so far... {response['LastEvaluatedKey']}")

while 'LastEvaluatedKey' in response:  
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])  
    data.extend(response['Items'])
    print(f"Fetched {len(data)} items so far... {response.get('LastEvaluatedKey', 'No more items')}")
print(f"Total items fetched: {len(data)}")
with open('export.json', 'w') as f:  
    json.dump(data, f)  
