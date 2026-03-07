#accsept imnput from user : 
# 1 studdy hours 
# 2 attendence
#apply knn algo 
# predict whether the student passes or fail

import math

def eucdistance(p1,p2):
    ans=math.sqrt((p1['studyhour'] - p2['studyhour'])**2 + (p1['attendance'] - p2['attendance'])**2)
    return ans

def KNNclassifier():

    border="-"*40

    data=[
        {'studyhour':2,'attendance':60,'label':'fail'},
        {'studyhour':5,'attendance':80,'label':'pass'},
        {'studyhour':6,'attendance':85,'label':'pass'},
        {'studyhour':1,'attendance':50,'label':'fail'},
    ]

    print(border)
    print("Training Dataset")
    print(border)

    for i in data:
        print(i)

    print(border)

    # Accept input
    studyhour=int(input("Enter Study Hours: "))
    attendance=int(input("Enter Attendance: "))

    new_point={'studyhour':studyhour,'attendance':attendance}

    # Calculate distance
    for d in data:
        d['distance']=eucdistance(d,new_point)

    # Sort distances
    sorted_data=sorted(data,key=lambda item:item['distance'])

    # Select K neighbors
    k=3
    nearest=sorted_data[:k]

    print(border)
    print("Nearest 3 elements are:")
    print(border)

    for d in nearest:
        print(d)

    # voting
    votes={}

    for neighbour in nearest:
        label=neighbour['label']
        votes[label]=votes.get(label,0)+1

    print(border)
    print("Voting result is:")
    print(border)

    for v in votes:
        print("name :",v,"number of votes :",votes[v])

    print(border)

    predicted_class=max(votes,key=votes.get)

    print("Predicted result is :",predicted_class)

    print(border)


def main():
    KNNclassifier()


if __name__=="__main__":
    main()


    