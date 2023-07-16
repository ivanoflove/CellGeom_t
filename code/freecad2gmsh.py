# path to your FreeCAD.so or FreeCAD.dll file
FREECADPATH = r"/usr/lib64/FreeCAD/lib"
import sys
sys.path.append(FREECADPATH)
# import FreeCAD module

import FreeCAD as app
import Part
import BOPTools.SplitFeatures
import pandas as pd
import numpy as np
import gmsh
import os


num = int(sys.argv[2]) + 1
row_str = sys.argv[1]
row = list(map(float, row_str.split()))
width_pitch, width_rib_a, width_rib_c, height_anode, height_cathode, height_electrolyte = row

# file_path_geom = r"D:\document\Code\python\CellGeom\data\geom_size.csv"
file_brep = f"../cad/cell{num}.brep".format(num)
file_mesh = f"../mesh/cell{num}.unv".format(num)
# geom_size = pd.read_csv(file_path_geom)
# geom_size = geom_size.to_numpy()


def create_rec(x1, y1, x2, y2, x3, y3, x4, y4, L, str):
    point1 = app.Vector(x1, y1, 0)
    point2 = app.Vector(x2, y2, 0)
    point3 = app.Vector(x3, y3, 0)
    point4 = app.Vector(x4, y4, 0)
    
    
    split_line1 = Part.LineSegment(point1, point2).toShape()
    split_line2 = Part.LineSegment(point2, point3).toShape()
    split_line3 = Part.LineSegment(point3, point4).toShape()
    split_line4 = Part.LineSegment(point4, point1).toShape()
    solid_face = Part.Face(Part.Wire([split_line1, split_line2, split_line3, split_line4]))
    doc.addObject("Part::Feature", str).Shape = solid_face.extrude(app.Vector(0, 0, L))


doc = app.newDocument("soec")

if width_rib_a == width_rib_c:
    x_divisions = [-width_pitch/2, -(width_pitch-width_rib_c)/2, (width_pitch-width_rib_c)/2, width_pitch/2]
elif width_rib_a > width_rib_c:
    x_divisions = [-width_pitch/2, -(width_pitch-width_rib_c)/2, -(width_pitch-width_rib_a)/2, 
                    (width_pitch-width_rib_a)/2, (width_pitch-width_rib_c)/2, width_pitch/2]
else:
    x_divisions = [-width_pitch/2, -(width_pitch-width_rib_a)/2, -(width_pitch-width_rib_c)/2, 
                    (width_pitch-width_rib_c)/2, (width_pitch-width_rib_a)/2, width_pitch/2]
    
y_divisions = [0, 0.5, 1.0, 1.0+height_cathode*1000/1000, 1.0+(height_cathode+height_anode)*1000/1000, 
            1.0+(height_cathode+height_anode)*1000/1000+0.5, 2.0+(height_cathode+height_anode)*1000/1000]

# 计算每个切割点的坐标
points = []
for y in y_divisions:
    for x in x_divisions:
        points.append([x, y])


draw_points = []
epoch = len(x_divisions) * (len(y_divisions) - 1)
for i in range(0, epoch-1):
    points_add = points[i] + points[i+1] + points[i+len(x_divisions)+1] + points[i+len(x_divisions)]
    if ( i + 1 ) % len(x_divisions) == 0:
        continue
    draw_points.append(points_add)

for row in draw_points:
    i = 0
    L = 89
    str = "solid{}".format(i)
    x1, y1, x2, y2, x3, y3, x4, y4 = row
    create_rec(x1, y1, x2, y2, x3, y3, x4, y4, L, str)
    i = i + 1


doc.recompute()
obj_list = [i for i in doc.Objects]

j = BOPTools.SplitFeatures.makeBooleanFragments(name="BooleanFragments")
j.Objects = obj_list
doc.recompute()


Part.export([doc.getObject("BooleanFragments")], file_brep)


gmsh.initialize()

gmsh.merge(file_brep)

if width_rib_a == width_rib_c:
    curves = [54, 49, 57, 62, 61, 56, 48, 53]
    # Define physical group information as a list of tuples
    physical_groups = [
        (2, [64], 119, "inlet_a"),
        (2, [25], 120, "inlet_c"),
        (2, [63], 121, "outlet_a"),
        (2, [24], 122, "outlet_c"),
        (2, [70, 75, 79], 123, "voltage"),
        (2, [1, 7, 12], 124, "current"),
        (2, [31, 36, 40], 125, "electrolyte"),
        (2, [53, 44], 126, "wal_rib_a"),
        (2, [27, 18], 127, "wall_rib_c"),
        (2, [4, 19, 32, 45, 58, 71], 128, "periodic_l"),
        (2, [13, 26, 39, 52, 65, 78], 129, "periodic_r"),
        (3, [10, 11, 12], 130, "anode"),
        (3, [7, 8, 9], 131, "cathode"),
        (3, [14], 132, "channel_a"),
        (3, [5], 133, "channel_c"),
        (3, [16, 17, 18, 15, 13], 134, "connect_a"),
        (3, [4, 1, 2, 3, 6], 135, "connect_c")
    ]

elif width_rib_a > width_rib_c:
    curves = [80, 75, 83, 88, 93, 98, 97, 92, 87, 82, 74, 79]
    physical_groups = [
        (2, [102], 185, "inlet_a"),
        (2, [39, 35, 43], 186, "inlet_c"),
        (2, [101], 187, "outlet_a"),
        (2, [42, 38, 34], 188, "outlet_c"),
        (2, [112, 117, 121, 125, 129], 189, "voltage"),
        (2, [1, 7, 12, 17, 22], 190, "current"),
        (2, [49, 54, 58, 62, 66], 191, "electrolyte"),
        (2, [70, 75, 83, 87], 192, "wal_rib_a"),
        (2, [28, 45], 193, "wall_rib_c"),
        (2, [4, 29, 50, 71, 92, 113], 194, "periodic_l"),
        (2, [23, 44, 65, 86, 107, 128], 195, "periodic_r"),
        (3, [16, 17, 18, 19, 20], 196, "anode"),
        (3, [11, 12, 13, 14, 15], 197, "cathode"),
        (3, [23], 198, "channel_a"),
        (3, [7, 8, 9], 199, "channel_c"),
        (3, [21, 26, 27, 28, 29, 22, 30, 25, 24], 200, "connect_a"),
        (3, [6, 1, 2, 3, 4, 5, 10], 201, "connect_c")
    ]
else:
    curves = [80, 75, 83, 88, 93, 98, 97, 92, 87, 82, 74, 79]
    physical_groups = [
        (2, [102, 98, 106], 185, "inlet_a"),
        (2, [39], 186, "inlet_c"),
        (2, [101, 97, 105], 187, "outlet_a"),
        (2, [38], 188, "outlet_c"),
        (2, [112, 117, 121, 125, 129], 189, "voltage"),
        (2, [1, 7, 12, 17, 22], 190, "current"),
        (2, [49, 54, 58, 62, 66], 191, "electrolyte"),
        (2, [70, 87], 192, "wal_rib_a"),
        (2, [28, 33, 41, 45], 193, "wall_rib_c"),
        (2, [4, 29, 50, 71, 92, 113], 194, "periodic_l"),
        (2, [23, 44, 65, 86, 107, 128], 195, "periodic_r"),
        (3, [16, 17, 18, 19, 20], 196, "anode"),
        (3, [11, 12, 13, 14, 15], 197, "cathode"),
        (3, [22, 23, 24], 198, "channel_a"),
        (3, [8], 199, "channel_c"),
        (3, [21, 26, 27, 28, 29, 30, 25], 200, "connect_a"),
        (3, [6, 1, 7, 2, 3, 4, 5, 10, 9], 201, "connect_c")
    ]

for curve in curves:
    gmsh.model.mesh.setTransfiniteCurve(curve, 3, meshType="Progression", coef=1.)
    
if width_rib_a != width_rib_c:
    for curve in [15, 20, 57, 85, 113, 141, 169, 31, 36, 67, 95, 123, 151, 179, 
                    30, 35, 66, 94, 122, 150, 178, 14, 19, 56, 84, 112, 140, 168]:
        gmsh.model.mesh.setTransfiniteCurve(curve, 2, meshType="Progression", coef=1.)

# Set transfinite meshing for all surfaces
surfaces = gmsh.model.get_entities(2)
for surface in surfaces:
    gmsh.model.mesh.setTransfiniteSurface(surface[1])
    # gmsh.model.mesh.setRecombine(surface[0], surface[1])

volumes = gmsh.model.get_entities(3)
for volume in volumes:
    gmsh.model.mesh.setTransfiniteVolume(volume[1])

# set boundary conditions
for group in physical_groups:
    gmsh.model.addPhysicalGroup(*group)

# set mesh options
gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
# gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
# gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.45) 
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.45)  
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
# gmsh.option.setNumber("Mesh.ElementOrder", 2)
gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)

# generate a 3D mesh.
gmsh.model.mesh.generate(3)

# save it to disk
gmsh.write(file_mesh)

gmsh.finalize()
os.remove(file_brep)
    
    
   