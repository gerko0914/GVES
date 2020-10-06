GVES
GVES: Gene Vector for Each Sample

Data formats
gene expression(example_expression.pkl) : a DataFrame file n+1 rows(index) and s columns. n is the number of genes and s is the number of samples. The additional row at first is the 0/1 binary prognosis labels. 
gene network(example_network.txt) : a text file with the number of edges. It consists of two genes per row.

Run GVES
In the terminal, 
        python GVES.py "example_expression.pkl" "example_network.txt" "Result_folder"
where Result_folder is the path where the results will be saved.
