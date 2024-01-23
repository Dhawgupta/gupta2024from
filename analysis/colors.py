agent_colors = {
   'ForwardTD' : 'blue',
   'TDLambda' : 'green',
   'BiTD' : 'red',
   'TDLambdaOnline' : 'purple',
   'BiTD2' : 'orange',
   'SARSALambda' : 'green',
   'BiSARSA' : 'red',
   'BiSARSA2' : 'orange',
    'MultiBiTD' : 'red',
    'MultiBiSARSA' : 'red',
    'MultiBiSARSA2' : 'orange',
    'MultiBiTD2' : 'orange',
    'MultiBiTD3' : 'blue',
    'MultiBiSARSA3' : 'blue'
}


line_stype = {
    1 : '-',
    2 : '--',
    3 : ':'
}

line_type_nodes = {
    1 : '-',
    4 : '--',
    16 : ':',
    32: '--',
    63: ':',
    64 : '-.'
}

colors_nodes ={
    1: 'blue',
    4: 'green',
    16: 'red',
    32: 'orange',
    63: 'violet',
    64: 'black'

}

line_node = {
    'continuous' : '-',
    'discrete' : ':'
}

critic_model = {
    'linear' : '-',
    'network' : ':'
}

line_stype_pretrain = {
    'pretrain' : '-',
    'pretrain_layer' : '--',
    'pretrain_node' : ':',
    'pretrain_none' : '-*-',
    'pretrain_node_deterministic_node' : '-.'
}

line_type_pretrain = {
    True : '-',
    False : ':'
}

colors_partition = {
    's' : 'orange',
    'd' : 'blue',
    'ddd': 'blue',
    'sss' : 'orange',
    'sdd' : 'red',
    'dsd' : 'green',
    'dds' : 'magenta',

    'dd' : 'purple',
    'ss' : 'orange',
    'sd' : 'red',
    'ds' : 'green',
}
colors_partition_pretrain2 = {
    's' : 'red',
    'sd': 'blue',
    'ss' : 'green',
    'ds' : 'brown'
}

line_type_modelpart = {
    'backprop' : '-',
    'coagent' : ':'
}

colors_averaging ={
    1: 'blue',
    5: 'green',
    10: 'red',
    25: 'orange',
    50: 'purple',
    100: 'black'

}


# Off Pac stuff
colors_offpac = {

    'coagent' : 'orange',
    'coagent_global_baseline' : 'green',
    'coagent_state_layer_baseline' : 'purple',

    # Off policy colors
    'coagent_offpac' : 'orange',
    'coagent_global_baseline_offpac' : 'green',
    'coagent_state_global_baseline_offpac' : 'purple'


}


'''
color
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
'''


'''

linestyle	description
'-' or 'solid'	solid line
'--' or 'dashed'	dashed line
'-.' or 'dashdot'	dash-dotted line
':' or 'dotted'	dotted line
'None'	: draw nothing
' '	draw nothing
''	draw nothing

'''
