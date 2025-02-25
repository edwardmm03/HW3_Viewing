import numpy as np
import matplotlib.pyplot as plt

# Create a 3D cube
def create_cube():
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    return vertices, edges

# Camera Transformation
def camera_transform(vertices, eye, gaze, up):
    # TODO: complete this function
    gnorm = gaze/np.linalg.norm(gaze)
    w = (-1*gaze)/gnorm

    u = (up*w)/(np.linalg.norm(up*w))

    v = w *u

    m1 = np.array([
        [u[0],u[1],u[2],0],
        [v[0],v[1],v[2],0],
        [w[0],w[1],w[2],0],
        [0,0,0,1]
    ])
    m2 = np.array([
        [1,0,0,-1*eye[0]],
        [0,1,0,-1*eye[1]],
        [0,0,1,-1*eye[2]],
        [0,0,0,1]
    ])

    transform = np.multiply(m1,m2)
    newvertices = np.array([])
    for point in vertices:
        #i think this is making my matrices the correct size
        np.append(point,1)
        np.append(newvertices,np.multiply(transform,np.transpose(point)))

    return newvertices

# Projection Transformation
def project_vertices(vertices, projection_type, near=1, far=10, fov=np.pi/4, aspect=1.0):
    """
    TODO: complete this function

    Apply a projection transformation to 3D vertices.
    - perspective: applies a perspective projection.
    - orthographic: applies an orthographic projection.
    """
    #if projection_type == "projection":
     #   pass
    #else:
        #performs an orthographic projection transformation
     #   Morth = np.array([])

    return vertices

# Viewport Transformation
def viewport_transform(vertices, width, height):
    Mvp = np.array([width/2,0,0,(width-1)/2],
                   [0, height/2,0,(height-1)/2],
                   [0,0,1,0],
                   [0,0,0,1])

    newvertices = np.array([])
    for point in vertices:
        np.append(point,1)
        np.append(newvertices,np.multiply(Mvp,np.transpose(point)))
    return newvertices

# Render the scene
def render_scene(vertices, edges, ax, **kwargs):
    for edge in edges:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], **kwargs)

# Main function
def main():
    # Scene setup
    vertices, edges = create_cube()
    eye = np.array([0.5, .5, -3])  # Camera at the origin
    gaze = np.array([-1, -1, -5])  # Looking towards the cube
    up = np.array([0, 1, 0])  # Up is along +Y-axis

    # Camera transformation
    transformed_vertices = camera_transform(vertices, eye, gaze, up)

    # Projection transformations
    perspective_vertices = project_vertices(transformed_vertices, "perspective", near=1, far=10, fov=np.pi/4, aspect=800/600)

    orthographic_vertices = project_vertices(transformed_vertices, "orthographic", near=1, far=10)

    # Viewport transformation
    viewport_width, viewport_height = 1920, 1080
    persp_2d = viewport_transform(perspective_vertices, viewport_width, viewport_height)
    ortho_2d = viewport_transform(orthographic_vertices, viewport_width, viewport_height)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].set_title("Perspective Projection")
    axes[1].set_title("Orthographic Projection")

    render_scene(persp_2d, edges, axes[0], color="blue", marker="o")
    render_scene(ortho_2d, edges, axes[1], color="red", marker="o")

    for ax in axes:
        ax.set_xlim(0, viewport_width)
        ax.set_ylim(0, viewport_height)
        ax.set_aspect('equal')
        ax.invert_yaxis()

    plt.show()


if __name__ == "__main__":
    main()
