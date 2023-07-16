import subprocess
import pandas as pd

file_path_geom = r"../data/geom_size.csv"
geom_size = pd.read_csv(file_path_geom).to_numpy()



# loop for compute
for num, row in enumerate(geom_size):
    if num >= 340 and num < 389:
        row_str = ' '.join(map(str, row))
        
        # run freecad2gmsh.py
        subprocess.run(['python3', "freecad2gmsh.py", row_str, str(num)])

        # run fluent.py
        subprocess.run(['python3', "fluent.py", row_str, str(num)])


