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
    gnorm = gaze/np.linalg.norm(gaze)
    w = (-1*gaze)/gnorm

    u = (np.cross(up,w))/(np.linalg.norm(up*w))

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
    
    vp_vertices = np.hstack((vertices,np.ones([8,1]))) #appending a 1 to the end of each point for proper sizing
    res_pts = transform @ vp_vertices.T
    print(res_pts)
    res_pts = res_pts[:3,:]

    # MJ: The shape of res_pts is now 4 by 8, it's now homogeneous coordinates.
    # MJ: But what you need to return is the x, y, and z coordinates of the points.
    # MJ: So, you need to get rid of the 4th dimension and then return it.

    return res_pts

# Projection Transformation
def project_vertices(vertices, projection_type, near=1, far=10, fov=np.pi/4, aspect=1.0):
    
    #assume r is 1 and l is -1
    r =1
    t =1
    l =-1
    b=-1
    M = np.array([
            [2/(r-l), 0,0, -(r+l)/(r-l)],
            [0,2/(t-b),0,-(t+b)/(t-b)],
            [0,0,2/(near-far),-(near+far)/(near-far)],
            [0,0,0,1]
        ])

    if projection_type == "perspective":
        #performs a perspective projection type
        M = np.array([
            [near,0,0,0],
            [0,near,0,0],
            [0,0,near + far, -near*far],
            [0,0,1,0]
        ])
  
    #vp_vertices = np.hstack((vertices,np.ones([8,1]))) #appending a 1 to the end of each point for proper sizing    
    res_pts = vertices.T @ M

    if projection_type == "perspective":
        res_pts[:,0] = res_pts[:,0]/res_pts[:,3]
        res_pts[:,1] = res_pts[:,1]/res_pts[:,3]
        res_pts[:,2] = res_pts[:,2]/res_pts[:,3]

    return res_pts[:,0:3]

# Viewport Transformation
def viewport_transform(vertices, width, height):
    print(vertices)
    Mvp = np.array([
        [width/2,0,0,(width-1)/2],
        [0, height/2,0,(height-1)/2],
        [0,0,1,0],
        [0,0,0,1]
    ])

    vp_vertices = np.hstack((vertices,np.ones([8,1]))) #appending a 1 to the end of each point for proper sizing
    res_pts = Mvp @ vp_vertices.T #transposing the vertices so the matrix multiplication is correct

    res = res_pts[:2,:] #only getting points from the matrix
    print(res)
    return res

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
        # ax.set_xlim(0, viewport_width)    # MJ: delete this line
        # ax.set_ylim(0, viewport_height)   # MJ: delete this line
        ax.set_aspect('equal')
        ax.invert_yaxis()

    plt.show()


if __name__ == "__main__":
    main()
