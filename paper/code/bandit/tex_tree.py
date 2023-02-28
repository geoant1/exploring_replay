import numpy as np
from copy import deepcopy

def generate_tex_tree(M, replays, save_path):

    # root prior values
    alpha0 = M[0, 0]
    beta0  = M[0, 1]
    alpha1 = M[1, 0]
    beta1  = M[1, 1]

    x0, y0 = -8, 0

    x1, y1 = -2, 5.6

    x2, y2 =  4, 9.2

    x1_0, y1_0 = x1,  y1
    # -------- #
    x1_2, y1_2 = x1, -y1

    x2_0, y2_0   = x2, y2
    x2_2, y2_2   = x2, y2-2.4
    x2_4, y2_4   = x2, y2-2.4*2
    x2_6, y2_6   = x2, y2-2.4*3
    # ---------- #
    x2_8, y2_8   = x2, y2-2.4*3 - 4
    x2_10, y2_10 = x2, y2-2.4*4 - 4
    x2_12, y2_12 = x2, y2-2.4*5 - 4
    x2_14, y2_14 = x2, y2-2.4*6 - 4

    with open(save_path, 'w') as f:

        f.write(r'\begin{minipage}{\textwidth}' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{state}   = [rectangle, text centered, draw=black, minimum width=2.2cm, fill=orange!30]' + '\n')
        f.write(r'\tikzstyle{between} = [rectangle, draw=none, minimum width=3.5cm]' + '\n')

        # root
        f.write(r'\node[state] at (%.2f, %.2f)   '%(x0, y0)    + r'(h0){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1) + '\n')
        
        # action 0 rew 1
        f.write(r'\node[state] at (%.2f, %.2f)  '%(x1_0, y1_0) + r'(h1_0){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h1_01){};'%(x1_0, y1_0-0.45) + '\n')
        # action 0 rew 0
        f.write(r'\node[state] at (%.2f, %.2f)'%(x1_0, y1_0-0.9)             + r'(h1_1){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1, beta1) + '\n')

        # action 1 rew 1
        f.write(r'\node[state] at (%.2f, %.2f)  '%(x1_2, y1_2) + r'(h1_2){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h1_23){};'%(x1_2, y1_2-0.45) + '\n')
        # action 1 rew 0 
        f.write(r'\node[state] at (%.2f, %.2f)'%(x1_2, y1_2-0.9)             + r'(h1_3){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_0, y2_0) + r'(h2_0){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+2, beta0, alpha1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_01){};'%(x2_0, y2_0-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_0, y2_0-0.9)             + r'(h2_1){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0+1, alpha1, beta1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_2, y2_2) + r'(h2_2){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_23){};'%(x2_2, y2_2-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_2, y2_2-0.9)             + r'(h2_3){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_4, y2_4) + r'(h2_4){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0+1, alpha1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_45){};'%(x2_4, y2_4-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_4, y2_4-0.9)             + r'(h2_5){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+2, alpha1, beta1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_6, y2_6) + r'(h2_6){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_67){};'%(x2_6, y2_6-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_6, y2_6-0.9)             + r'(h2_7){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)  '%(x2_8, y2_8) + r'(h2_8){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1+1, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_89){};'%(x2_8, y2_8-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_8, y2_8-0.9)             + r'(h2_9){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1+1, beta1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_10, y2_10) + r'(h2_10){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+2, beta1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_1011){};'%(x2_10, y2_10-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_10, y2_10-0.9)             + r'(h2_11){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_12, y2_12) + r'(h2_12){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0+1, beta0, alpha1, beta1+1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_1213){};'%(x2_12, y2_12-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_12, y2_12-0.9)             + r'(h2_13){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0+1, alpha1, beta1+1) + '\n')

        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_14, y2_14) + r'(h2_14){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1+1, beta1+1) + '\n')
        f.write(r'\node[between] at (%.2f, %.2f) (h2_1415){};'%(x2_14, y2_14-0.45) + '\n')
        f.write(r'\node[state] at (%.2f, %.2f)'%(x2_14, y2_14-0.9)             + r'(h2_15){\scriptsize \begin{tabular}{c c} $\alpha_0=%u$ & $\beta_0=%u$ \\ $\alpha_1=%u$ & $\beta_1=%u$ \end{tabular}};'%(alpha0, beta0, alpha1, beta1+2) + '\n')

        # horizon 0 action 0
        for rep in replays:
            if rep[1] == 0:
                if rep[-1] == (0, 0, 0, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=5.5cm]h0)   |- (h1_01)  node[above, pos=0.8] {arm $0$};'%(colour) + '\n')
        
        # horizon 0 action 1
        for rep in replays:
            if rep[1] == 0:
                if rep[-1] == (0, 0, 0, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=-5.5cm]h0)  |- (h1_23)  node[above, pos=0.8] {arm $1$};'%(colour) + '\n')
        
        # (horizon 0 action 0 rew 1) -> action 0
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (0, 0, 0, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=4.0cm]h1_0) |- (h2_01)  node[above, pos=0.8] {arm $0$};'%(colour) + '\n')
        
        # (horizon 0 action 0 rew 1) -> action 1
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (0, 0, 0, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_0) |- (h2_23)  node[above, pos=0.8] {arm $1$};'%(colour) + '\n')
        
        # (horizon 0 action 0 rew 0) -> action 0
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (0, 0, 1, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_1) |- (h2_45)  node[above, pos=0.8] {arm $0$};'%(colour) + '\n')
        
        # (horizon 0 action 0 rew 0) -> action 1
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (0, 0, 1, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_1) |- (h2_67)  node[above, pos=0.8] {arm $1$};'%(colour) + '\n')
        
        # (horizon 0 action 1 rew 1) -> action 0
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (1, 0, 2, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_2) |- (h2_89)  node[above, pos=0.8] {arm $0$};'%(colour) + '\n')
        
        # (horizon 0 action 1 rew 1) -> action 1
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (1, 0, 2, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_2) |- (h2_1011) node[above, pos=0.8] {arm $1$};'%(colour) + '\n')
        
        # (horizon 0 action 1 rew 0) -> action 0
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (1, 0, 3, 0):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_3) |- (h2_1213) node[above, pos=0.8] {arm $0$};'%(colour) + '\n')
        
        # (horizon 0 action 1 rew 0) -> action 1
        for rep in replays:
            if rep[1] == 1:
                if rep[-1] == (1, 0, 3, 1):
                    colour = 'red'
                    break
                else:
                    colour = 'black'
            else:
                colour = 'black'
        f.write(r'\draw[->, thick, %s] ([yshift=1.5cm]h1_3) |- (h2_1415) node[above, pos=0.8] {arm $1$};'%(colour) + '\n')

        f.write(r'\end{tikzpicture}' + '\n')
        f.write(r'\end{minipage}' + '\n')

    return None

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

def generate_big_tex_tree(replays, q_history, need_history, file_path, display_need=True, tree_height=None, between_height=0.1, between_width=0.7, state_height=0.2, state_width=0.45, gap=0.22):
    
    h = len(q_history)

    if tree_height is None:
        y_max = 9
    else:
        y_max = tree_height

    x_max = 10

    with open(file_path, 'w') as f:

        f.write(r'\documentclass[tikz, border=1mm]{standalone}' + '\n')
        f.write(r'\begin{document}' + '\n')

        f.write(r'\begin{minipage}{\textwidth}' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\begin{tikzpicture}' + '\n') 
        f.write(r'\tikzstyle{between} = [rectangle, minimum height=%.2fcm, minimum width=%.2fcm, draw opacity=0]'%(between_height, between_width) + '\n')
        f.write(r'\tikzstyle{qval}    = [rectangle, text centered, text width=2cm]' + '\n')
        
        # generate nodes towards which arrows will be drawn
        # (these are in between the resulting belief states)
        between_nodes = {hi:[] for hi in range(h)}
        for hi in range(1, h):

            if hi == 1:
                num_nodes = 2
            else:
                num_nodes = 2*4**(hi-1)
            x_node    = -x_max + 0.8*(hi*2)

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

            x_node    = -x_max + 0.8*(hi*2)

            if hi == 0:
                node_name = str(hi) + '_s_' + str(0)
                y_node    = 0
                alpha     = need_history[hi][0]
                if display_need:
                    f.write(r'\node[rectangle, text centered, draw=black, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1, text opacity=1] at (%.2f, %.2f) (%s) {\tiny %.2f};'%(state_height, state_width, alpha, x_node, y_node, node_name, alpha) + '\n')
                else:
                    f.write(r'\node[rectangle, draw=black, inner sep=0pt, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1] at (%.2f, %.2f) (%s) {};'%(state_height, state_width, alpha, x_node, y_node, node_name) + '\n')
                state_nodes[hi].append(node_name)
            else:
                for idx, y_node in enumerate(reversed(np.linspace(-y_max*(hi+1)/h, y_max*(hi+1)/h, num_nodes))):
                    node_name = str(hi) + '_s_' + str(idx*2)
                    alpha     = need_history[hi][idx*2]
                    if display_need:
                        f.write(r'\node[rectangle, text centered, draw=black, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1, text opacity=1] at (%.2f, %.2f) (%s) {\tiny %.2f};'%(state_height, state_width, alpha, x_node, y_node+gap, node_name, alpha) + '\n')
                    else:
                        f.write(r'\node[rectangle, draw=black, inner sep=0pt, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1] at (%.2f, %.2f) (%s) {};'%(state_height, state_width, alpha, x_node, y_node+gap, node_name) + '\n')
                    if hi == h-1:
                        qvals  = q_history[hi][idx*2]
                        probas = np.exp(qvals)/np.sum(np.exp(qvals))
                        val    = np.dot(probas, qvals) 
                        f.write(r'\node[qval] at (%.2f, %.2f) () {\tiny \textcolor{blue}{%.2f}};'%(x_node+0.60, y_node+gap, val) + '\n')
                    state_nodes[hi].append(node_name)
                    node_name = str(hi) + '_s_' + str(idx*2+1)
                    
                    alpha = need_history[hi][idx*2+1]
                    if display_need:
                        f.write(r'\node[rectangle, text centered, draw=black, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1, text opacity=1] at (%.2f, %.2f) (%s) {\tiny %.2f};'%(state_height, state_width, alpha, x_node, y_node-gap, node_name, alpha) + '\n')
                    else:
                        f.write(r'\node[rectangle, draw=black, inner sep=0pt, minimum height=%.2fcm, minimum width=%.2fcm, fill=orange, fill opacity=%.2f, draw opacity=1] at (%.2f, %.2f) (%s) {};'%(state_height, state_width, alpha, x_node, y_node-gap, node_name) + '\n')
                    if hi == h-1:
                        qvals = q_history[hi][idx*2+1]
                        probas = np.exp(qvals)/np.sum(np.exp(qvals))
                        val    = np.dot(probas, qvals) 
                        f.write(r'\node[qval] at (%.2f, %.2f) () {\tiny \textcolor{blue}{%.2f}};'%(x_node+0.60, y_node-gap, val) + '\n')
                    state_nodes[hi].append(node_name)

        for hi in range(h-1):
            for k in state_nodes[hi]:
                idx1 = int(k.split('_')[-1])
                for k1 in between_nodes[hi+1]:
                    idx2 = int(k1.split('_')[-1])

                    cond = (idx1*2 == idx2) or (idx1*2+1 == idx2)
                    
                    if cond:
                        colour  = 'black'
                        text    = None
                        for rep_idx, rep in enumerate(replays[::-1]):
                            if rep is None:
                                break
                            if hi in rep[0]:
                                hidx = np.argwhere(rep[0] == hi).flatten()[0]
                                if (idx1 == rep[1][hidx]):
                                    if (rep[1][hidx]*2 + rep[-1][hidx]) == idx2:
                                        colour  = 'red'
                                        if rep_idx == 0:
                                            text = hidx
                                        break

                        for kq, vq in q_history[hi].items():
                            if kq == idx1:
                                if (kq*2 == idx2): 
                                    q_val = vq[idx2%2]
                                    break
                                if (kq*2+1 == idx2):
                                    q_val = vq[idx2%2]
                                    break

                        if text is None:
                            f.write(r'\draw[->, thick, %s] (%s) -- (%s) node [pos=0.70, above=-0.2em, sloped, font=\tiny] () {\textcolor{blue}{%.2f}};'%(colour, k+'.east', k1+'.west', q_val) + '\n')
                        else:
                            f.write(r'\draw[->, thick, green] (%s) -- (%s) node [pos=0.35, above=-0.2em, sloped, font=\tiny] () {\textcolor{black}{%u}} node [pos=0.80, above=-0.2em, sloped, font=\tiny] () {\textcolor{blue}{%.2f}};'%(k+'.east', k1+'.west', text, q_val) + '\n')

        f.write(r'\end{tikzpicture}' + '\n')
        f.write(r'\end{minipage}' + '\n')
        f.write(r'\end{document}')

    return None