#!/bin/bash

# Loop over the batches of training data
#rm sge_output/*


for i in {0,1,2,3,4,5,6}; 
do	
	qsub -v qubit=8,layer_tot=25,name_data='mnist',sample=40,batch=$i,noise=0.1 rt_pqk_arb.sh
	echo "submitting batch=$i"
done	

