import plotly.graph_objects as go
import numpy as np
import imageio.v2 as imageio
import os

def _convert_to_cartesian(coords, coord_type):
    if coord_type == 'cartesian':
        x, y, z = zip(*coords)
    elif coord_type == 'ra_dec':
        ra, dec = zip(*coords)
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
    elif coord_type == 'spherical':
        r, theta, phi = zip(*coords)
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        x = np.multiply(r, np.sin(phi) * np.cos(theta))
        y = np.multiply(r, np.sin(phi) * np.sin(theta))
        z = np.multiply(r, np.cos(phi))
    else:
        raise ValueError("Invalid coordinate type. Choose from 'cartesian', 'ra_dec', 'spherical'.")
    
    return list(x), list(y), list(z)

def create_gif(coords, coord_type='cartesian', labels=None, num_frames=36, output_file='scatter_plot.gif', duration=None):
    """
    Creates an animated GIF of a rotating 3D scatter plot for the given coordinates.
    
    Parameters:
    - coords: List of tuples/lists containing the coordinates.
    - coord_type: Type of coordinates ('cartesian', 'ra_dec', 'spherical').
    - labels: Optional list of labels for the points.
    - num_frames: Number of frames for the rotation animation.
    - output_file: Name of the output GIF file.
    """
    x, y, z = _convert_to_cartesian(coords, coord_type)
    
    frames = []
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    for i in range(num_frames):
        angle = 360 * i / num_frames
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,                # Color by z coordinate
                colorscale='Viridis',   # Choose a colorscale
                opacity=0.8
            ),
            text=labels,  # Use text for labels if provided
            hoverinfo='text'
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=False, showline=False, showticklabels=False, zeroline=False, title=''),
                yaxis=dict(showbackground=False, showgrid=False, showline=False, showticklabels=False, zeroline=False, title=''),
                zaxis=dict(showbackground=False, showgrid=False, showline=False, showticklabels=False, zeroline=False, title=''),
                camera_eye=dict(x=np.sin(np.deg2rad(angle)), y=np.cos(np.deg2rad(angle)), z=0.5)
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            scene_bgcolor='black',
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        frame_file = f"{temp_dir}/frame_{i:03d}.png"
        fig.write_image(frame_file)
        frames.append(imageio.imread(frame_file))
    
    imageio.mimsave(output_file, frames, duration=0.1)
    for frame_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)

# Example usage

ra_dec_coords = [(0, 0), (45, 45), (90, 0), (135, -45), (180, 0)]
#cartesian_coords = [(1, 10, 20), (2, 11, 21), (3, 12, 22), (4, 13, 23), (5, 14, 24)]
#spherical_coords = [(1, 0, 0), (1, 45, 45), (1, 90, 90), (1, 135, 135), (1, 180, 180)]

point_labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5']

create_gif(ra_dec_coords, 'ra_dec', point_labels, output_file='plot.gif', num_frames=50)
#create_rotating_3d_scatter_gif(cartesian_coords, 'cartesian', point_labels)
#create_rotating_3d_scatter_gif(spherical_coords, 'spherical', point_labels, output_file='spherical_scatter_plot.gif')