#!/bin/bash

# Loop over the batches of training data
#rm sge_output/*


for i in {0,1,2,3,4,5,6}; 
do	
    for j in {0.01,0.025,0.05,0.075,0.1};
    do
    	qsu -v qubit=8,layer_tot=25,name_data='mnist',sample=40,batch=$i,noise=$j rt_pqk_arb.sh
    	echo "submitting batch=$i noise=$j"
    done
done	
