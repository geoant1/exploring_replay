import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon, Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns
import os, glob, shutil, ast
sns.set_style('white')

def load_env(env_file_path):

    '''
    ---
    Load the environment configuration file

    env_file_path -- path to the configuration file 
    ---
    '''

    with open(env_file_path, 'r') as f:
        env_config = {}
        for line in f:
            k, v = line.strip().split('=')
            env_config[k.strip()] = ast.literal_eval(v.strip())
    
    return env_config 

def plot_env(ax, env):

    '''
    ---
    Visualise the environment 

    ax  -- axis to plot on
    env -- instance of the Environment class
    ---
    '''

    # state grid
    for st_x in range(env.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(env.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)
    
    for st in range(env.num_states):
        for ac in range(env.num_actions):
            for bidx, l in enumerate(env.blocked_state_actions):
                if [st, ac] in l:
                    if env.barriers[bidx]:
                        i, j = np.argwhere(np.arange(env.num_states).reshape(env.num_y_states, env.num_x_states) == st).flatten()
                        if ac == 0:
                            ax.hlines((env.num_y_states-i), j, j+1, linewidth=6, color='b')
                        elif ac == 2:
                            ax.vlines(j, (env.num_y_states-i)-1, (env.num_y_states-i), linewidth=6, color='b')
                    break

    if len(env.nan_states) > 0:
        patches = []
        for s in env.nan_states:
            sy, sx   = env._convert_state_to_coords(s)
            patches += [Rectangle((sx, env.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

    # goal symbol
    for (goal_y, goal_x) in env.goal_coords:
        ax.scatter(goal_x+0.5, env.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    ax.set_xlim(0, env.num_x_states)
    ax.set_ylim(0, env.num_y_states)

    return None

def add_patches(s, a, q, num_y_states, num_x_states):

    '''
    ---
    Add coloured triangles for actions and their Q-values 
    when visualising the maze

    s            -- state
    a            -- action
    q            -- Q-value for this state and action
    num_y_states -- number of states in the y dimension
    num_x_states -- number of states in the x dimension 
    ---
    '''
    
    num_states = num_y_states * num_x_states

    patches = []
    
    if q >= 0:
        # colour red
        col_tuple = (1, 0, 0, q)
    else:
        # colour blue
        col_tuple = (0, 0, 1, -q)
    
    # find coordinates of this state
    i, j = np.argwhere(np.arange(num_states).reshape(num_y_states, num_x_states) == s).flatten()
    
    # move up
    if a == 0:
        patches.append(RegularPolygon((0.5+j, num_y_states-0.18-i), 3, radius=0.13, lw=0.5, orientation=0, edgecolor='k', fill=True, facecolor=col_tuple))
    # move down
    elif a == 1:
        patches.append(RegularPolygon((0.5+j, num_y_states-0.82-i), 3, radius=0.13, lw=0.5, orientation=np.pi, edgecolor='k', fill=True, facecolor=col_tuple))
    # move left
    elif a == 2:
        patches.append(RegularPolygon((0.20+j, num_y_states-0.49-i), 3, radius=0.13, lw=0.5, orientation=np.pi/2, edgecolor='k', fill=True, facecolor=col_tuple))
    # move right
    else:
        patches.append(RegularPolygon((0.80+j, num_y_states-0.49-i), 3, radius=0.13, lw=0.5, orientation=-np.pi/2, edgecolor='k', fill=True, facecolor=col_tuple))
                    
    return patches

def plot_maze(ax, Q, agent, move=None, colorbar=True, colormap='Blues'):
    
    # state grid
    for st_x in range(agent.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(agent.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)

    # nan_idcs = np.argwhere(np.all(np.isnan(Q), axis=1)).flatten()
    # Q[nan_idcs, :] = 0

    Q_plot = np.zeros(agent.num_states)
    for s in range(agent.num_states):
        max_val   = 0
        if np.all(np.isnan(Q[s, :])):
            Q_plot[s] = 0
        else:
            Q_plot[s] = np.nanmax(Q[s, :])

    # Q_plot   = np.nanmax(Q, axis=1).reshape(agent.num_y_states, agent.num_x_states)[::-1, :]
    Q_plot = Q_plot.reshape(agent.num_y_states, agent.num_x_states)[::-1, :]

    if np.all(Q_plot == 0):
        sns.heatmap(np.absolute(Q_plot), cmap=['white'], annot=False, fmt='.2f', cbar=colorbar, vmin=0, vmax=1, ax=ax)
    else:
        sns.heatmap(np.absolute(Q_plot), cmap=colormap, annot=False, fmt='.2f', cbar=colorbar, vmin=0, vmax=1, ax=ax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(agent.num_states), agent.goal_states + agent.nan_states):
        for ac in range(4):
            if ~np.isnan(Q[st, ac]):
                if Q[st, ac] == 0:
                    patches += add_patches(st, ac, 0, agent.num_y_states, agent.num_x_states)
                else:
                    patches += add_patches(st, ac, Q[st, ac], agent.num_y_states, agent.num_x_states)
            for bidx, l in enumerate(agent.blocked_state_actions):
                if [st, ac] in l:
                    if agent.barriers[bidx]:
                        i, j = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == st).flatten()
                        if ac == 0:
                            ax.hlines((agent.num_y_states-i), j, j+1, linewidth=6, color='b')
                        elif ac == 2:
                            ax.vlines(j, (agent.num_y_states-i)-1, (agent.num_y_states-i), linewidth=6, color='b')
                    break

    if len(agent.nan_states) > 0:
        for s in agent.nan_states:
            sy, sx   = agent._convert_state_to_coords(s)
            patches += [Rectangle((sx, agent.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # goal symbol
    # goal_y, goal_x   = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent.goal_state).flatten()
    # ax.scatter(goal_x+0.5, agent.num_y_states - goal_y -0.5, s=300, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    for (goal_y, goal_x) in agent.goal_coords:
        ax.scatter(goal_x+0.5, agent.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    # agent location
    if move is not None:
        agent_state      = move[-1]
        agent_y, agent_x = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent_state).flatten()
        ax.scatter(agent_x+0.5, agent.num_y_states - agent_y -0.5, s=80, c='green', alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, agent.num_x_states)
    ax.set_ylim(0, agent.num_y_states)

    # if move is not None:
        # ax.set_title('[' + ' '.join(map(str, [int(i) for i in move])) + ']', fontsize=20)

    return None

def plot_need(ax, need, agent, colorbar=True, colormap='Blues', normalise=True, vmin=0, vmax=1):
    
    need_plot = need.reshape(agent.num_y_states, agent.num_x_states)[::-1, :]
    if normalise:
        need_plot = need_plot/np.nanmax(need_plot)

    if np.all(need_plot == 0):
        sns.heatmap(need_plot, cmap=['white'], annot=False, fmt='.2f', cbar=colorbar, ax=ax, vmin=vmin, vmax=vmax)
    else:
        sns.heatmap(need_plot, cmap=colormap, annot=True, fmt='.2f', cbar=colorbar, ax=ax, vmin=vmin, vmax=vmax)
    
    # arrows for actions
    patches = []
    for st in np.delete(range(agent.num_states), agent.goal_states + agent.nan_states):
        for ac in range(4):
            for bidx, l in enumerate(agent.uncertain_states_actions):
                if [st, ac] in l:
                    if agent.barriers[bidx]:
                        i, j = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == st).flatten()
                        if ac == 0:
                            ax.hlines((agent.num_y_states-i), j, j+1, linewidth=6, color='b')
                        elif ac == 2:
                            ax.vlines(j, (agent.num_y_states-i)-1, (agent.num_y_states-i), linewidth=6, color='b')
                    break

    if len(agent.nan_states) > 0:
        for s in agent.nan_states:
            sy, sx   = agent._convert_state_to_coords(s)
            patches += [Rectangle((sx, agent.num_y_states-sy-1), 1, 1, edgecolor='k', facecolor='k', linewidth=1)]

    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)
                
    # state grid
    for st_x in range(agent.num_x_states):
        ax.axvline(st_x, c='k', linewidth=0.6)
    for st_y in range(agent.num_y_states):
        ax.axhline(st_y, c='k', linewidth=0.6)

    # goal symbol
    # goal_y, goal_x   = np.argwhere(np.arange(agent.num_states).reshape(agent.num_y_states, agent.num_x_states) == agent.goal_state).flatten()
    for (goal_y, goal_x) in agent.goal_coords:
        ax.scatter(goal_x+0.5, agent.num_y_states - goal_y -0.5, s=600, c='orange', marker=r'$\clubsuit$', alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(0, agent.num_x_states)
    ax.set_ylim(0, agent.num_y_states)

    return None