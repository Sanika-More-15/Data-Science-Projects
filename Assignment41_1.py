import numpy as np
import math

def EucDistance(P1,P2):
    Ans = math.sqrt((P1['X'] - P2['X']) ** 2 + (P1['Y'] - P2['Y']) ** 2)
    return Ans

def KNNClassifire():
    border="-"*60

    data=[
        {'Point':'A', 'X':1, 'Y':2, 'Label':'Red'},
        {'Point':'B', 'X':2, 'Y':3, 'Label':'Red'},
        {'Point':'C', 'X':3, 'Y':1, 'Label':'Blue'},
        {'Point':'D', 'X':6, 'Y':5, 'Label':'Blue'}
    ]

    print(border)
    print("Marvellous Use Defined KNN")
    print(border)

    print(border)
    print("Traning Dataset")
    print(border)

    for i in data:
        print(i)

    print(border)

    new_point = {'X':2, 'Y':2}

    print(data[0])
    print(new_point)

    Result=EucDistance(data[0],new_point)
    print(Result)

    #Calculated all distances
    for d in data:
        d['distance'] = EucDistance(d,new_point)

    print(border)
    print("Calculated distance are :")
    print(border)

    for d in data:
        print(d)

    sorted_data = sorted(data, key=lambda item : item['distance'])

    print(border)
    print("Sorted data is :")
    print(border)

    for d in sorted_data:
        print(d)

    k = 3
    nearest = sorted_data[:k]

    print(border)
    print("Nearest Three elements are :")
    print(border)

    for d  in nearest:
        print(d)

    # Voting

    Votes = {}
    for neighbour in nearest:
        label = neighbour['Label']
        Votes[label] = Votes.get(label,0) + 1

    print(border)
    print("Voting result is :")
    print(border)

    for d in Votes:
        print("Name :",d ,"No. of votes :",Votes[d])

    print(border)

    predicted_class = max(Votes, key=Votes.get)

    print("Predicted class of (2,2) is :",predicted_class)


def main():
    KNNClassifire()

if __name__=="__main__":
    main()