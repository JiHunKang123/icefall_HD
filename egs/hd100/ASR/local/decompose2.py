#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
import hgtk

if len(sys.argv) != 3:
    print("Insufficient arguments")
    sys.exit()

input_txt=sys.argv[1]
output_txt=sys.argv[2]

with open(input_txt, 'r') as f:
  lines = f.readlines()

decomposed_lines=[]
for line in lines:
  line = line.replace('\n','')
  decomposed_line = hgtk.text.decompose(line).replace('ᴥ','ㄿ')
  decomposed_lines.append(decomposed_line+'\n')

with open(output_txt, 'w') as f:
  f.writelines(decomposed_lines)
