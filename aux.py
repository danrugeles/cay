#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is available under the MIT License.
# (c)2015 Daniel Rugeles 
# daniel007[at]e.ntu.edu.sg / Nanyang Technological University

import math
import numpy as np
from random import randint,seed


__author__ = "Daniel Rugeles"
__copyright__ = "Copyright 2015, Auxiliary"
__credits__ = ["Daniel Rugeles"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Daniel Rugeles"
__email__ = "daniel007@e.ntu.edu.sg"
__status__ = "Released"


#01-06-2015 Added combine

# Compose two dictionaries
def combine(dict1,dict2):
	dict3={}
	for k1 in dict1:
		dict3[k1]=dict2[dict1[k1]]
	return dict3
	
#Makes sure all columns go from 0-n
def dictionate(alldata):
	alldata=np.array(alldata)
	alldata_T=alldata.transpose()	
	val2idxs=[]
	idx2vals=[]
	counts=[]

	#print alldata_T
	for row in alldata_T:
		sortedvalues= np.sort(row)
		idx=-1
		
		prevval=-1
		val2idx={}
		idx2val={}
		count=[]
		c=0
		for value in sortedvalues:
			c+=1
			if value!=prevval:
				count.append(c)
				c=0
				prevval=value
				idx+=1
				val2idx[value]=idx
				idx2val[idx]=value
		count.append(c+1)
		counts.append(count[1:])
		val2idxs.append(val2idx)
		idx2vals.append(idx2val)

	
	alldata_T_hashed=alldata_T.copy()

	for idx,col in enumerate(alldata_T):
		alldata_T_hashed[idx]=map(lambda x: val2idxs[idx][x],alldata_T[idx]) 
		#print alldata_T_hashed[idx]
	
	return alldata_T_hashed.transpose(),idx2vals,val2idxs,counts


# Data must be dictionated!
def join2(data):
	
	max_=np.max(data,axis=0)
	D=np.zeros(max_+1)	

	for row in data:
		D[tuple(row)]+=1

	return D

	
def addrandomtopic(data,k,col):

	k=round(k)
	for idx in range(len(data)):
		data[idx][col]=float(randint(0,k-1))
	return data

