import numpy as np
from copy import deepcopy

def generate_value_diff_tree(qval_tree1, qval_tree2, file_path):

    y_max = 9
    x_max = 10

    h = len(qval_tree1.keys())
    diff_tree = deepcopy(qval_tree1)
    for hi in range(h):
        for idx, vals in qval_tree1[hi].items():
            diff_tree[hi][idx] = qval_tree1[hi][idx] - qval_tree2[hi][idx]

    with open(file_path, 'w') as f:

        f.write(r'\documentclass[tikz, border=1mm]{standalone}' + '\n')
        f.write(r'\begin{document}' + '\n')

        f.write(r'\begin{minipage}{\textwidth}' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{between} = [rectangle, draw=none]' + '\n')
        f.write(r'\tikzstyle{qval}    = [rectangle, text centered, text width=2cm]' + '\n')
        
        # generate nodes towards which arrows will be drawn
        # (these are in between the resulting belief states)
        between_nodes = {hi:[] for hi in range(h)}
        for hi in range(1, h):

            if hi == 1:
                num_nodes = 2
            else:
                num_nodes = 2*4**(hi-1)
            x_node    = -x_max + hi*2

            for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/h, y_max*(hi+1)/h, num_nodes))):
                node_name = str(hi) + '_b_' + str(idx)
                f.write(r'\node[between] at (%.2f, %.2f) (%s){};'%(x_node, y_node, node_name) + '\n')
                between_nodes[hi].append(node_name)

        # now generate the individual belief state nodes
        state_nodes = {hi:[] for hi in range(h)}
        for hi in range(h):
            
            if hi == 1:
                num_nodes = 2
            else:
                num_nodes = 2*4**(hi-1)

            x_node    = -x_max + hi*2

            if hi == 0:
                node_name = str(hi) + '_s_' + str(0)
                y_node    = 0
                f.write(r'\node[rectangle, text centered, draw=black, minimum height=1mm, text width=3mm, inner sep=0pt, fill=white, draw opacity=1] at (%.2f, %.2f) (%s){};'%(x_node, y_node, node_name) + '\n')
                state_nodes[hi].append(node_name)
            else:
                for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/h, y_max*(hi+1)/h, num_nodes))):
                    node_name = str(hi) + '_s_' + str(idx*2)
                    f.write(r'\node[rectangle, text centered, draw=black, minimum height=1mm, text width=3mm, inner sep=0pt, fill=white, draw opacity=1] at (%.2f, %.2f) (%s) {};'%(x_node, y_node+0.08, node_name) + '\n')
                    if hi == h-1:
                        qvals  = diff_tree[hi][idx*2]
                        probas = np.exp(qvals)/np.sum(np.exp(qvals))
                        val    = np.dot(probas, qvals) 
                        f.write(r'\node[qval] at (%.2f, %.2f) () {\tiny \textcolor{blue}{%.2f}};'%(x_node+0.45, y_node+0.08, val) + '\n')
                    state_nodes[hi].append(node_name)
                    node_name = str(hi) + '_s_' + str(idx*2+1)
                    
                    f.write(r'\node[rectangle, text centered, draw=black, minimum height=1mm, text width=3mm, inner sep=0pt, fill=white, draw opacity=1] at (%.2f, %.2f) (%s) {};'%(x_node, y_node-0.08, node_name) + '\n')
                    if hi == h-1:
                        qvals  = diff_tree[hi][idx*2+1]
                        probas = np.exp(qvals)/np.sum(np.exp(qvals))
                        val    = np.dot(probas, qvals) 
                        f.write(r'\node[qval] at (%.2f, %.2f) () {\tiny \textcolor{blue}{%.2f}};'%(x_node+0.45, y_node-0.08, val) + '\n')
                    state_nodes[hi].append(node_name)

        for hi in range(h-1):
            for k in state_nodes[hi]:
                idx1 = int(k.split('_')[-1])
                for k1 in between_nodes[hi+1]:
                    idx2 = int(k1.split('_')[-1])

                    cond = (idx1*2 == idx2) or (idx1*2+1 == idx2)
                    
                    if cond:
                        colour  = 'black'

                        for kq, vq in diff_tree[hi].items():
                            if kq == idx1:
                                if (kq*2 == idx2): 
                                    q_val = vq[idx2%2]
                                    break
                                if (kq*2+1 == idx2):
                                    q_val = vq[idx2%2]
                                    break
                        
                        if q_val >= 0:
                            qcolour = 'red'
                        else:
                            qcolour = 'blue'
                        f.write(r'\draw[->, thick, %s] (%s) -- (%s) node [pos=0.80, above=-0.2em, sloped, font=\tiny] () {\textcolor{%s}{%.2f}};'%(colour, k, k1, qcolour, q_val) + '\n')

        f.write(r'\end{tikzpicture}' + '\n')
        f.write(r'\end{minipage}' + '\n')
        f.write(r'\end{document}')


    return None

def generate_big_tex_tree(tree, replays, q_history, need_history, file_path, tree_height=None, between_height=0.1, between_width=0.7, state_height=0.2, state_width=0.45, gap=0.22):
    
    if tree_height is None:
        y_max = 6
    else:
        y_max = tree_height

    x_max = 10

    with open(file_path, 'w') as f:

        f.write(r'\documentclass[tikz, border=1mm]{standalone}' + '\n')
        f.write(r'\begin{document}' + '\n')

        # f.write(r'\begin{minipage}{\textwidth}' + '\n')
        # f.write(r'\centering' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{between} = [rectangle, minimum height=%.2fcm, minimum width=%.2fcm, draw opacity=0]'%(between_height, between_width) + '\n')
        f.write(r'\tikzstyle{qval}    = [rectangle, text centered, text width=2cm]' + '\n')
        
        # generate nodes towards which arrows will be drawn
        # (these are in between the resulting belief states)
        between_nodes = {hi:[] for hi in range(tree.horizon)}
        for hi in range(1, tree.horizon):

            num_nodes = len(q_history[hi-1])*2
            # num_nodes = 2*4**(hi-1)
            x_node    = -x_max + 0.8*(hi*2)

            for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/tree.horizon, y_max*(hi+1)/tree.horizon, num_nodes))):
                
                node_name = str(hi) + '_b_' + str(idx)
                f.write(r'\node[between] at (%.2f, %.2f) (%s){};'%(x_node, y_node, node_name) + '\n')
                between_nodes[hi].append(node_name)

        # now generate the individual belief state nodes
        state_nodes = {hi:[] for hi in range(tree.horizon)}
        for hi in range(tree.horizon):
            
            # num_nodes = 2*4**(hi-1)
            x_node    = -x_max + 0.8*(hi*2)

            if hi == 0:
                node_name = str(hi) + '_s_' + str(0)
                y_node    = 0
                alpha     = need_history[hi][0]
                f.write(r'\node[rectangle, text centered, draw=black, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1, text opacity=1] at (%.2f, %.2f) (%s) {\tiny %.2f};'%(state_height, state_width, alpha, x_node, y_node, node_name, alpha) + '\n')
                state_nodes[hi].append(node_name)
            else:
                num_nodes = len(q_history[hi-1])*2
                for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/tree.horizon, y_max*(hi+1)/tree.horizon, num_nodes))):
                    
                    if (idx*2) in need_history[hi].keys():
                        node_name = str(hi) + '_s_' + str(idx*2)
                        alpha     = need_history[hi][idx*2]
                        f.write(r'\node[rectangle, text centered, draw=black, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1, text opacity=1] at (%.2f, %.2f) (%s) {\tiny %.2f};'%(state_height, state_width, alpha, x_node, y_node+gap, node_name, alpha) + '\n')
                        if hi == tree.horizon-1:
                            qvals  = q_history[hi][idx*2]
                            # val    = tree._value(qvals)
                            val    = np.max(qvals)
                            f.write(r'\node[qval] at (%.2f, %.2f) () {\tiny \textcolor{blue}{%.2f}};'%(x_node+0.60, y_node+gap, val) + '\n')
                        state_nodes[hi].append(node_name)

                    if (idx*2+1) in need_history[hi].keys():
                        node_name = str(hi) + '_s_' + str(idx*2+1)
                        alpha = need_history[hi][idx*2+1]
                        f.write(r'\node[rectangle, text centered, draw=black, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1, text opacity=1] at (%.2f, %.2f) (%s) {\tiny %.2f};'%(state_height, state_width, alpha, x_node, y_node-gap, node_name, alpha) + '\n')
                        if hi == tree.horizon-1:
                            qvals  = q_history[hi][idx*2+1]
                            # val    = tree._value(qvals)
                            val    = np.max(qvals) 
                            f.write(r'\node[qval] at (%.2f, %.2f) () {\tiny \textcolor{blue}{%.2f}};'%(x_node+0.60, y_node-gap, val) + '\n')
                        state_nodes[hi].append(node_name)

        for hi in range(tree.horizon-1):
            for k in state_nodes[hi]:
                idx1 = int(k.split('_')[-1])

                for vals in tree.belief_tree[hi][idx1][1]:
                    
                    a    = vals[0]
                    idx2 = int(vals[1][0]/2)
                    k1   = str(hi+1) + '_b_' + str(idx2)

                    colour  = 'black'
                    text    = None
                    for rep_idx, rep in enumerate(replays[::-1]):
                        if rep is None:
                            break
                        if hi in rep[0]:
                            hidx = np.argwhere(rep[0] == hi).flatten()[0]
                            if (idx1 == rep[1][hidx]):
                                if (rep[2][hidx]) == a:
                                # if (rep[1][hidx]*2 + rep[-1][hidx]) == idx2:
                                    colour  = 'red'
                                    if rep_idx == 0:
                                        text = hidx
                                    break

                    q_val = q_history[hi][idx1][a]

                    if text is None:
                        f.write(r'\draw[->, thick, %s] (%s) -- (%s) node [pos=0.70, above=-0.2em, sloped, font=\tiny] () {\textcolor{blue}{%.2f}};'%(colour, k+'.east', k1+'.west', q_val) + '\n')
                    else:
                        f.write(r'\draw[->, thick, green] (%s) -- (%s) node [pos=0.35, above=-0.2em, sloped, font=\tiny] () {\textcolor{black}{%u}} node [pos=0.80, above=-0.2em, sloped, font=\tiny] () {\textcolor{blue}{%.2f}};'%(k+'.east', k1+'.west', text, q_val) + '\n')

        f.write(r'\end{tikzpicture}' + '\n')
        # f.write(r'\end{minipage}' + '\n')
        f.write(r'\end{document}')

    return None