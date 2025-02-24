import cv2 
import time
import numpy as np
import matplotlib.pyplot as plt

from lightglue import viz2d
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import numpy_image_to_torch, rbd


data_path = '../data/images'


class cfg:
    img0 = data_path + "/car.png"
    img1 = data_path + "/car_park.png"
    
    size = (512, 512)
    interpolation = cv2.INTER_AREA
    
    opencv = {
        "extractor": cv2.KAZE_create(),
        "normType": cv2.NORM_L2,
        "crossCheck": True, 
        "homography": {
            "method": cv2.RANSAC,
            "ransacReprojThreshold": 3.0
        }
    }
    
    lightglue = {
        "extractor": "SuperPoint", # SuperPoint, DISK
        "device": "cpu", # cpu, cuda
        "max_kpts": 2048,
        "homography": {
            "method": cv2.RANSAC,
            "ransacReprojThreshold": 3.0
        }
    }

def load_img(file, size, interpolation):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=interpolation)
    return img


def get_homography(src_pts, dst_pts, method, ransacReprojThreshold):
    homography, mask = cv2.findHomography(
        src_pts, 
        dst_pts, 
        method=method, 
        ransacReprojThreshold=ransacReprojThreshold
    )
    return homography, mask


def plot_two_imgs(img0, img1, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img0)
    axes[0].axis("off")
    axes[0].set_title(title)
    axes[1].imshow(img1)
    axes[1].axis("off")
    axes[1].set_title(title)
    plt.show()


def preprocess_opencv(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def match_opencv(img0, img1, cfg):    
    # preprocess rgb images
    img0 = preprocess_opencv(img0)
    img1 = preprocess_opencv(img1)

    s_time = time.time()

    # extract local features
    extractor = cfg["extractor"]
    kp0, des0 = extractor.detectAndCompute(img0, None)
    kp1, des1 = extractor.detectAndCompute(img1, None)
    
    # match the features
    matcher = cv2.BFMatcher_create(normType=cfg["normType"], crossCheck=cfg["crossCheck"])
    matches = matcher.match(des0, des1)
    
    # extract point coordinates from keypoint objects
    src_pts = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    print(time.time() - s_time)

    return {
        "src_pts": src_pts,
        "dst_pts": dst_pts,
        "matches": matches,
        "kp0": kp0,
        "kp1": kp1,
        "img0": img0,
        "img1": img1
    }
    

def visualize_opencv(src_pts, dst_pts, kp0, kp1, matches, img0, img1, cfg, title="OpenCV", **kwargs):   
    homography, mask = get_homography(src_pts, dst_pts, cfg["method"], cfg["ransacReprojThreshold"])
    matches_mask = mask.ravel().tolist()
    
    # visualize mapping
    h, w = img0.shape[:2]
    pts = np.float32([[0, 0],[0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, homography)
    vis_mapping = cv2.polylines(img1.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    # visualize matches
    draw_params = dict(
        matchColor = (0, 255, 0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matches_mask, # draw only inliers
        flags = 2
    )
    vis_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches, None, **draw_params)
    
    # plot visualizations
    plot_two_imgs(vis_mapping, vis_matches, title=title)
    
    return homography


img0 = load_img(cfg.img0, size=cfg.size, interpolation=cfg.interpolation)
img1 = load_img(cfg.img1, size=cfg.size, interpolation=cfg.interpolation)

results_opencv = match_opencv(img0, img1, cfg.opencv)
homography_opencv = visualize_opencv(**results_opencv, cfg=cfg.opencv["homography"], title="OpenCV")
