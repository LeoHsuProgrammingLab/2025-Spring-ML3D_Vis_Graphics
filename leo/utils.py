import cv2 

def threshold_img(img, threshold=20):
    """Threshold the image to create a binary mask."""
    img[img < threshold] = 0

    return img

def segment_image(image_path, output_dir):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]  # should be 960 x 320

    patches = []
    for i in range(0, height, 320):
        for j in range(0, width, 320):
            patch = image[i:i+320, j:j+320]
            patches.append(patch)
            cv2.imwrite(f"{output_dir}/patch_{i//320}_{j//320}.jpg", patch)

def main():
    img_path = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/output_mv/test_0_0/010.png'

    img = cv2.imread(img_path)
    new_img = threshold_img(img, threshold=30)
    cv2.imwrite(img_path, new_img)

    # img_path = '/home/agenuinedream/repo/zero123plus/output.png'
    # output_dir = '/home/agenuinedream/repo/2025-Spring-ML3D_Vis_Graphics/leo/output/test_0'
    # segment_image(img_path, output_dir)

if __name__ == "__main__":
    main()