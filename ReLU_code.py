import numpy as np

inputs =[0,2,-1,3.3,-2.7,1.1,2.2,-100]

output =[]


#forma  1 para escribir ReLU 
for i in inputs:
    if i >0:
       output.append(i)
    else:
        output.append(0)
print(output)

#forma 2 para escribir ReLU
for i in inputs:
    output.append(max(0,i))

print(output)

#escribir ReLU con numpy

output = np.maximum(0,inputs)

print(output)