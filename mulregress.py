import  csv,numpy
w=numpy.zeros(6,dtype=float)
b=0
rows=[]
file="realestate.csv"
with open(file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
 
    # extracting field names through first row
    fields = next(csvreader)
    
    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

data = numpy.array(rows,dtype=float)

pred=[]
for i,j in zip(data,range(len(data)-1)):
    pred[j]=numpy.dot(i[1:7],w)+b

print(pred)