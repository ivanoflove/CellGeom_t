import ansys.fluent.core as pyfluent
import pandas as pd
import sys
import os


num = int(sys.argv[2]) + 1
row_str = sys.argv[1]
row = list(map(float, row_str.split()))
width_pitch, width_rib_a, width_rib_c, height_anode, height_cathode, height_electrolyte = row
# change units
height_electrolyte = height_electrolyte / 1000

# read data
soec_data = pd.read_csv(r'../data/soec.csv')
materials_data = pd.read_csv(r'../data/materials.csv')
boundary_conditions_data = pd.read_csv(r'../data/boundary_conditions.csv')
solution_data = pd.read_csv(r'../data/solution.csv')

import_filename = f"../mesh/cell{num}.unv".format(num)
file_path_udf = r"../mesh/constit.c"
file_export = f'../data/out/cell{num}.out'.format(num)
file_resault = r"../data/resault.csv"
# set the global parameters
anode_interface = 20
cathode_interface = 43
voltage_tap = 37
current_tap = 7
wall_rib_a = 25
wall_rib_c = 13
iterate_first = 200
iterate_second = 2 * iterate_first
conv_fist = 1e-6
conv_second = 9.9e-7


for index, row in soec_data.iterrows():
    value = row[1]
    name = row[2]
    globals()[name] = value

for index, row in materials_data.iterrows():
    value = row[1]
    name = row[2]
    globals()[name] = value

for index, row in boundary_conditions_data.iterrows():
    value = row[1]
    name = row[2]
    globals()[name] = value


for index, row in solution_data.iterrows():
    value = row[1]
    name = row[2]
    globals()[name] = value



# import_filename = "CrossII.cas"
solver = pyfluent.launch_fluent(
    version="3d", precision="double", processor_count=int(count), show_gui=False, mode="solver"
)
# solver.file.read(file_type="case", file_name=import_filename)

solver.tui.file.import_.ideas_universal(import_filename)

# scale mesh 
solver.tui.mesh.scale('0.001', '0.001', '0.001')
# check mesh
solver.tui.mesh.check() 

# change interconnect to solid
solver.tui.define.boundary_conditions.zone_type("connect_a", "solid")
solver.tui.define.boundary_conditions.zone_type("connect_c", "solid")

# change electrolyte to wall
solver.tui.define.boundary_conditions.zone_type("interface-electrolyte", "wall")

# set periodic boundary conditions
# solver.tui.define.boundary_conditions.modify_zones.make_periodic(
#     "periodic_l-anode",
#     "periodic_r-anode",
#     "no",
#     "yes",
#     "yes"
# )
# solver.tui.define.boundary_conditions.modify_zones.make_periodic(
#     "periodic_l-cathode",
#     "periodic_r-cathode",
#     "no",
#     "yes",
#     "yes"
# )
# solver.tui.define.boundary_conditions.modify_zones.make_periodic(
#     "periodic_l-connect_a",
#     "periodic_r-connect_a",
#     "no",
#     "yes",
#     "yes"
# )
# solver.tui.define.boundary_conditions.modify_zones.make_periodic(
#     "periodic_l-connect_c",
#     "periodic_r-connect_c",
#     "no",
#     "yes",
#     "yes"
# )


# open sofc-module
solver.tui.define.models.addon_module("4")
solver.tui.define.models.sofc_model.enable_sofc_model("yes")

# solver.tui.define.user_defined.compile_customized_addon_module("yes", r"D:\document\Code\python\CellGeom\mesh\constit.c")


solver.tui.define.models.sofc_model.model_parameters(
    "yes",  # Enable Electrolyte Conductivity Submodel
    "no",  # Enable Volumetric Energy Source
    "yes",  # Enable Surface Energy Source
    "yes",  # Enable Species Sources
    "no",  # Disable CO Electrochemistry
    "yes",  # Enable Electrolysis Mode
    "yes",  # Enable Knudsen Diffusion
    "no",  # Set Electrical Boundary Condition in Boundary Conditions Task Page
    "yes",  # Converge to Specified System Voltage
    voltage,  # Total System Voltage
    "0",  # Leakage Current Density
    "0.3",  # Current Under-Relaxation Factor
    height_electrolyte,  # Electrolyte Thickness
    resis,  # Electrolyte Resistivity
    "yes"  # Apply F-Cycle for All Equations
)

solver.tui.define.models.sofc_model.electrochemistry(
    cur_a,  # Anode Exchange Current Density
    cur_c,  # Cathode Exchange Current Density
    "yes",  # Enable Cathode Temperature Dependent I_0
    A_c,  # Temperature Dependent Coefficient A
    B_c,  # Temperature Dependent Coefficient B
    "yes",  # Enable Anode Temperature Dependent I_0
    A_a,  # Temperature Dependent Coefficient A
    B_a,  # Temperature Dependent Coefficient B
    alpha_a,  # Anode Anodic Transfer Coefficient
    beta_a,  # Anode Cathodic Transfer Coefficient
    alpha_c,  # Cathode Anodic Transfer Coefficient
    beta_c,  # Cathode Cathodic Transfer Coefficient
    mf_h2,  # H2 Reference Value
    mf_o2,  # O2 Reference Value
    mf_h2o,  # H2O Reference Value
    ex_h2,  # H2 Exponent
    ex_o2,  # O2 Exponent
    ex_h2o,  # H2O Exponent
)


# set anode and electrolyte interface
solver.tui.define.models.sofc_model.anode_interface(
    anode_interface,  
    "()"
)

# set cathode and electrolyte interface
solver.tui.define.models.sofc_model.cathode_interface(
    cathode_interface,  
    "()"
)

# set size of pore to compute Knudsen Diffusion
solver.tui.define.models.sofc_model.pore_size_interface(
    "(anode . %.10f)" % pore_a,  
    "(cathode . %.10f)" % pore_c  
)

# set tortuosity of electrode
solver.tui.define.models.sofc_model.tortuosity_interface(
    "(anode . %d)" % tor_a,  
    "(cathode . %d)" % tor_c  
)

# set the conductivity
solver.tui.define.models.sofc_model.electric_field_model.conductive_regions(
    "(anode . %.2f)" % cond_a,  
    "(cathode . %.2f)" % cond_c,  
    "(connect_a . %.2f)" % cond_ac, 
    "(connect_c . %.2f)" % cond_cc  
)

# set contact resistance of the interconnector to the electrode
solver.tui.define.models.sofc_model.electric_field_model.contact_resistance_regions(
    "(%d . %.10f)" % (wall_rib_a, resis_ac),  
    "(%d . %.10f)" % (wall_rib_c, resis_cc)  
)

# define voltage and current
solver.tui.define.models.sofc_model.electric_field_model.voltage_tap(voltage_tap, "()")
solver.tui.define.models.sofc_model.electric_field_model.current_tap(current_tap, "()")



# set materials parameters
solver.setup.materials.mixture['sofc-mixture'].density.option = 'ideal-gas'
solver.tui.define.materials.change_create(
    "anode-default", "anode-default",
    "yes", "constant", den_a,  # density
    "yes", "constant", cp_a,  # Specific heat
    "yes", "constant", tc_a,  # Thermal Conductivity
    "no"  # UDS Diffusivity
)

solver.tui.define.materials.change_create(
    "cathode-default", "cathode-default",
    "yes", "constant", den_c,  # density
    "yes", "constant", cp_c,  # Specific heat
    "yes", "constant", tc_c,  # Thermal Conductivity
    "no"  # UDS Diffusivity
)

solver.tui.define.materials.change_create(
    "cathode-collector-default", "cathode-collector-default",
    "yes", "constant", den_ac,  # density
    "yes", "constant", cp_ac,  # Specific heat
    "yes", "constant", tc_ac,  # Thermal Conductivity
    "no"  # UDS Diffusivity
)

solver.tui.define.materials.change_create(
    "anode-collector-default", "anode-collector-default",
    "yes", "constant", den_cc,  # density
    "yes", "constant", cp_cc,  # Specific heat
    "yes", "constant", tc_cc,  # Thermal Conductivity
    "no"  # UDS Diffusivity
)


# 定义操作压力
solver.tui.define.operating_conditions.operating_pressure(pressure)

# set cell zone conditions
solver.setup.cell_zone_conditions.fluid["anode"].porous_r_1.value = float(per_a)
solver.setup.cell_zone_conditions.fluid["anode"].porous_r_2.value = float(per_a)
solver.setup.cell_zone_conditions.fluid["anode"].porous_r_3.value = float(per_a)
solver.setup.cell_zone_conditions.fluid["anode"].porosity.value = float(por_a)
solver.setup.cell_zone_conditions.fluid["anode"].solid_material = 'anode-default'
solver.setup.cell_zone_conditions.fluid["anode"].surface_volume_ratio = float(sv_a)

solver.setup.cell_zone_conditions.fluid["cathode"].porous_r_1.value = float(per_c)
solver.setup.cell_zone_conditions.fluid["cathode"].porous_r_2.value = float(per_c)
solver.setup.cell_zone_conditions.fluid["cathode"].porous_r_3.value = float(per_c)
solver.setup.cell_zone_conditions.fluid["cathode"].porosity.value = float(por_c)
solver.setup.cell_zone_conditions.fluid["cathode"].solid_material = 'cathode-default'
solver.setup.cell_zone_conditions.fluid["cathode"].surface_volume_ratio = float(sv_c)

solver.setup.cell_zone_conditions.solid["connect_a"].material = 'anode-collector-default'
solver.setup.cell_zone_conditions.solid["connect_c"].material = 'cathode-collector-default'
solver.tui.define.boundary_conditions.set.fluid(
    "anode",
    "cathode",
    "()",
    "viscosity-ratio",
    "brinkman",
    "()"
)

# set pressure outlet
# solver.tui.define.boundary_conditions.zone_type("outlet_a", "pressure-outlet")
# solver.tui.define.boundary_conditions.zone_type("outlet_c", "pressure-outlet")
# set mass flow inlet
solver.tui.define.boundary_conditions.zone_type("inlet_a", "mass-flow-inlet")
solver.tui.define.boundary_conditions.zone_type("inlet_c", "mass-flow-inlet")

# set boundary conditions
solver.setup.boundary_conditions.mass_flow_inlet["inlet_a"].mass_flow.value = float(MF_a)
solver.setup.boundary_conditions.mass_flow_inlet["inlet_a"].t0.value = float(T_a)
solver.setup.boundary_conditions.mass_flow_inlet["inlet_a"].species_in_mole_fractions = True
solver.setup.boundary_conditions.mass_flow_inlet["inlet_a"].mf["h2"].value = float(mf_h2)
solver.setup.boundary_conditions.mass_flow_inlet["inlet_a"].mf["h2o"].value = float(mf_h2o)

solver.setup.boundary_conditions.mass_flow_inlet["inlet_c"].mass_flow.value = float(MF_c)
solver.setup.boundary_conditions.mass_flow_inlet["inlet_c"].t0.value = float(T_c)
solver.setup.boundary_conditions.mass_flow_inlet["inlet_c"].species_in_mole_fractions = True
solver.setup.boundary_conditions.mass_flow_inlet["inlet_c"].mf["o2"].value = float(mf_o2)

# set the solution methode
solver.tui.solve.set.p_v_coupling(SIMPLEC)  # set the SIMPLEC method
solver.tui.solve.set.gradient_scheme("no", "no")  # use Green-Gauss Cell-Based

solver.solution.methods.discretization_scheme["pressure"] = 'standard'
solver.solution.methods.discretization_scheme["density"] = 'second-order-upwind'
solver.solution.methods.discretization_scheme["mom"] = 'second-order-upwind'
solver.solution.methods.discretization_scheme["species-0"] = 'second-order-upwind'
solver.solution.methods.discretization_scheme["species-1"] = 'second-order-upwind'
solver.solution.methods.discretization_scheme["species-2"] = 'second-order-upwind'
solver.solution.methods.discretization_scheme["temperature"] = 'second-order-upwind'
solver.solution.methods.discretization_scheme["uds-0"] = 'second-order-upwind'

solver.tui.solve.set.under_relaxation(
    "pressure", ur_p,
    "density", ur_d,
    "body-force", ur_bf,
    "mom", ur_mom,
    "species-0", ur_s0,
    "species-1", ur_s1,
    "species-2", ur_s2,
    "temperature", ur_T,
    "uds-0", ur_e
)

solver.tui.solve.report_definitions.add(
    "current",
    "surface-areaavg",
    "field",
    "magnitude-of-current-density",
    "surface-names",
    "current",
    "()",
    "quit",
)

solver.tui.solve.report_definitions.add(
    "max_temperature",
    "volume-max",
    "field",
    "temperature",
    "zone-names",
    "anode",
    "cathode",
    "channel_a",
    "channel_c",
    "connect_a",
    "connect_c"
    "()",
    "quit",
)
solver.tui.solve.report_definitions.add(
    "min_temperature",
    "volume-min",
    "field",
    "temperature",
    "zone-names",
    "anode",
    "cathode",
    "channel_a",
    "channel_c",
    "connect_a",
    "connect_c"
    "()",
    "quit",
)
solver.tui.solve.report_files.add(
    "current_file",
    "report-defs",
    "current",
    "max_temperature",
    "min_temperature",
    "()",
    "print?",
    "no",
    "file-name",
    file_export,
    "quit"
)

# solver.tui.solve.report_plots.add(
#     "current_plot",
#     "report-defs",
#     "current",
#     "()",
#     "print?",
#     "yes",
#     "quit"
# )

# 定义残差检测
solver.tui.solve.monitors.residual.check_convergence("yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "yes")
solver.tui.solve.monitors.residual.criterion_type("0")
solver.tui.solve.monitors.residual.convergence_criteria(
    "%.1e" % conv_fist, "%.1e" % conv_fist, "%.1e" % conv_fist, "%.1e" % conv_fist, "%.1e" % conv_fist,
    "%.1e" % conv_fist, "%.1e" % conv_fist, "%.1e" % conv_fist, "%.1e" % conv_fist
)
# solver.tui.solve.monitors.residual.monitor("no", "no", "no", "no", "no", "no", "no", "no", "no")
solver.tui.solve.monitors.residual.plot("no")
solver.tui.solve.monitors.residual.print("no")

solver.tui.solve.initialize.set_defaults("temperature", T_a)
solver.tui.solve.initialize.initialize_flow()
solver.tui.solve.set.reporting_interval('10')
print("初始化完成")

solver.tui.solve.iterate(int(iterate_first))

print("初步计算完成，添加热源")

solver.tui.define.models.sofc_model.model_parameters(
    "yes",  # Enable Electrolyte Conductivity Submodel
    "yes",  # Enable Volumetric Energy Source
    "yes",  # Enable Surface Energy Source
    "yes",  # Enable Species Sources
    "no",  # Disable CO Electrochemistry
    "yes",  # Enable Electrolysis Mode
    "yes",  # Enable Knudsen Diffusion
    "no",  # Set Electrical Boundary Condition in Boundary Conditions Task Page
    "yes",  # Converge to Specified System Voltage
    voltage,  # Total System Voltage
    "0",  # Leakage Current Density
    "0.3",  # Current Under-Relaxation Factor
    height_electrolyte,  # Electrolyte Thickness
    resis,  # Electrolyte Resistivity
    "yes"  # Apply F-Cycle for All Equations
)

# 定义残差检测
solver.tui.solve.monitors.residual.check_convergence('yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes')
solver.tui.solve.monitors.residual.criterion_type("0")
solver.tui.solve.monitors.residual.convergence_criteria(
    "%.1e" % conv_second, "%.1e" % conv_second, "%.1e" % conv_second, "%.1e" % conv_second, "%.1e" % conv_second, 
    "%.1e" % conv_second, "%.1e" % conv_second, "%.1e" % conv_second, "%.1e" % conv_second
)

solver.tui.solve.iterate(int(iterate_second))
# 退出fluent求解器
solver.exit()

# # 读取文本文件为DataFrame
df_out = pd.read_csv(file_export, delimiter=' ', skiprows=2)
last_row = df_out.iloc[-1, 1:]
df_csv = pd.read_csv(file_resault)
cols_to_fill = df_csv.columns[6:]
df_csv.loc[num - 1, cols_to_fill] = last_row.values
df_csv.to_csv(file_resault, index=False)
os.remove(import_filename)
