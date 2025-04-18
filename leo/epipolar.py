import numpy as np
import cv2
import json

def compute_fundamental_matrix(K, R1, t1, R2, t2):
    # Relative rotation and translation
    R = R2 @ R1.T
    t = t2 - R @ t1

    # Skew-symmetric matrix of t
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

    # Essential matrix
    E = tx @ R

    # Fundamental matrix
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F

def draw_epipolar_lines(img, lines, color=(0, 255, 0)):
    h, w = img.shape[:2]
    img_drawn = img.copy()
    for line in lines:
        a, b, c = line
        x0, y0 = 0, int(-c / b) if b != 0 else 0
        x1, y1 = w, int(-(a * w + c) / b) if b != 0 else h
        img_drawn = cv2.line(img_drawn, (x0, y0), (x1, y1), color, 1)
    return img_drawn

def draw_epipolar_from_poses(img1, img2, K, R1, t1, R2, t2, pts1_2d):
    """
    img1, img2: np.array (images)
    K: camera intrinsics (3x3)
    R1, t1, R2, t2: extrinsics of the two cameras
    pts1_2d: Nx2 points in img1 (e.g. keypoints)
    """
    # Step 1: Compute Fundamental Matrix
    F = compute_fundamental_matrix(K, R1, t1, R2, t2)

    # Step 2: Convert points to homogeneous
    pts1_h = cv2.convertPointsToHomogeneous(pts1_2d).reshape(-1, 3)

    # Step 3: Compute epipolar lines in img2
    lines2 = pts1_h @ F.T  # Each row is a line: [a, b, c]

    # Step 4: Draw lines on img2
    img2_with_lines = draw_epipolar_lines(img2, lines2)

    return img2_with_lines, lines2

def draw_dotted_error_lines(img, keypoints, lines, color=(0, 255, 255), step=5):
    """
    Draw dotted lines from keypoints to the closest point on epipolar lines.

    Parameters:
    - img: The input image (numpy array).
    - keypoints: Nx2 array of keypoints.
    - lines: Nx3 array of lines [a, b, c] representing ax + by + c = 0.
    - color: BGR tuple for line color.
    - step: Distance between dashes.
    """
    img_copy = img.copy()
    for pt in keypoints:
        distance = np.iinfo(np.int32).max
        x_closest, y_closest = -1, -1
        for line in lines:
            a, b, c = line
            x0, y0 = pt
            denom = a**2 + b**2
            if denom == 0:
                continue
            # Closest point on the epipolar line
            x_closest_to_line = (b * (b * x0 - a * y0) - a * c) / denom
            y_closest_to_line = (a * (-b * x0 + a * y0) - b * c) / denom

            # Calculate distance
            dist = np.sqrt((x_closest_to_line - x0)**2 + (y_closest_to_line - y0)**2)
            if dist < distance:
                distance = dist
                x_closest, y_closest = int(x_closest_to_line), int(y_closest_to_line)

        # Draw dotted line from keypoint to epipolar line
        p1 = np.array([x0, y0])
        p2 = np.array([x_closest, y_closest])
        total_dist = np.linalg.norm(p1 - p2)
        num_steps = int(total_dist // step)

        cv2.circle(img_copy, (int(p1[0]), int(p1[1])), 2, (255, 255, 0), -1)
        cv2.circle(img_copy, (int(p2[0]), int(p2[1])), 2, (0, 0, 255), -1)

        for i in range(0, num_steps, 2):
            alpha = i / max(1, num_steps) 
            beta = min(1, (i + 1) / max(1, num_steps))
            dot_start = (p1 * (1 - alpha) + p2 * alpha).astype(int)
            dot_end = (p1 * (1 - beta) + p2 * beta).astype(int)
            cv2.line(img_copy, tuple(dot_start), tuple(dot_end), (180, 255, 255), 1, cv2.LINE_AA)

    return img_copy

def compute_epipolar_loss(F_mat, pts1, pts2):
    """ Compute average symmetric epipolar distance """
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    # l2 = F * x1
    l2 = np.dot(F_mat, pts1_h.T).T
    l1 = np.dot(F_mat.T, pts2_h.T).T
    # Distance from point to epipolar line
    d2 = np.abs(np.sum(l2 * pts2_h, axis=1)) / np.sqrt(l2[:, 0]**2 + l2[:, 1]**2)
    d1 = np.abs(np.sum(l1 * pts1_h, axis=1)) / np.sqrt(l1[:, 0]**2 + l1[:, 1]**2)
    return np.mean(d1 + d2)

def epi_with_pose():
    pose_file_1 = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/000.npy'
    pose_file_2 = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/010.npy'
    img1_path = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/000.png'
    img2_path = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/010.png'
    json_file = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/matches.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    H, W= img1.shape[:2]

    pose1 = np.load(pose_file_1, allow_pickle=True)
    pose2 = np.load(pose_file_2, allow_pickle=True)
    R1 = pose1[:3, :3]
    t1 = pose1[:3, 3]
    R2 = pose2[:3, :3]
    t2 = pose2[:3, 3]

    fov_horizontal = 49.1
    f = W / (2 * np.tan(np.deg2rad(fov_horizontal) / 2))
    cx = W / 2
    cy = H / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    
    pts1_2d = data['kpts1']
    pts1_2d = np.array(pts1_2d)
    pts2_2d = data['kpts2']
    pts2_2d = np.array(pts2_2d)

    img2_lines, lines2 = draw_epipolar_from_poses(img1, img2, K, R1, t1, R2, t2, pts1_2d)
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/epi_010.png', img2_lines)

    # Compute Fundamental Matrix
    F, mask = cv2.findFundamentalMat(
        pts1_2d, pts2_2d, method=cv2.USAC_MAGSAC, ransacReprojThreshold=0.2, confidence=0.999999, maxIters=10000
    )
    loss = compute_epipolar_loss(F, pts1_2d, pts2_2d)
    print('Epipolar loss: ', loss)

    # Draw epipolar lines on img1
    img1_lines, lines1 = draw_epipolar_from_poses(img2, img1, K, R2, t2, R1, t1, pts2_2d)
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/epi_000.png', img1_lines)

    # Draw dotted lines from keypoints to the closest point on epipolar lines
    dotted_img1 = draw_dotted_error_lines(img1_lines, pts1_2d, lines1)
    dotted_img2 = draw_dotted_error_lines(img2_lines, pts2_2d, lines2)
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/dotted_img1.png', dotted_img1)
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1_0/dotted_img2.png', dotted_img2)

def epi_without_pose():
    img1_path = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/patch_0_0.jpg'
    img2_path = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/patch_0_1.jpg'
    json_file = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/matches.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    H, W= img1.shape[:2]

    fov_horizontal = 49.1
    f = W / (2 * np.tan(np.deg2rad(fov_horizontal) / 2))
    cx = W / 2
    cy = H / 2
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    
    pts1_2d = data['kpts1']
    pts1_2d = np.array(pts1_2d)
    pts2_2d = data['kpts2']
    pts2_2d = np.array(pts2_2d)

    # Compute Fundamental Matrix    
    F, mask = cv2.findFundamentalMat(
        pts1_2d, pts2_2d, method=cv2.USAC_MAGSAC, ransacReprojThreshold=0.2, confidence=0.999999, maxIters=10000
    )
    # Check if F is valid
    if F is None:
        print("Fundamental matrix could not be computed.")
        return
    # Normalize the fundamental matrix
    F /= F[2, 2]

    # Compute epipolar lines in img2
    lines2 = cv2.computeCorrespondEpilines(pts1_2d.reshape(-1, 1, 2), 1, F)
    
    # Draw epipolar lines on img2
    img2_lines = draw_epipolar_lines(img2, lines2.reshape(-1, 3))
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/epi_0_1.png', img2_lines)

    # Draw epipolar lines on img1
    pts2_2d = data['kpts2']
    pts2_2d = np.array(pts2_2d)

    lines1 = cv2.computeCorrespondEpilines(pts2_2d.reshape(-1, 1, 2), 2, F.T)
    img1_lines = draw_epipolar_lines(img1, lines1.reshape(-1, 3))
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/epi_0_0.png', img1_lines)

    # Compute epipolar loss
    loss = compute_epipolar_loss(F, pts1_2d, pts2_2d)
    print('Epipolar loss: ', loss)

    # Draw dotted lines from keypoints to the closest point on epipolar lines
    dotted_img1 = draw_dotted_error_lines(img1_lines, pts1_2d, lines1.reshape(-1, 3))
    dotted_img2 = draw_dotted_error_lines(img2_lines, pts2_2d, lines2.reshape(-1, 3))
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/dotted_img1.png', dotted_img1)   
    cv2.imwrite('/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/matches/test_1/dotted_img2.png', dotted_img2)

def main():
    epi_with_pose()
    epi_without_pose()

if __name__== "__main__":
    main()
    # Example usage