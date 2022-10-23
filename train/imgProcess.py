import numpy as np
from train import config


def Face_detect(imgObj):
    # 转换成图片矩阵
    img = np.array(imgObj)[:, :, :3]
    dets = config.face_detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    # for i, d in enumerate(dets):
    #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
    #         i, d.left(), d.top(), d.right(), d.bottom()))
    # 人脸检测
    if len(dets) == 1:
        return img, dets[0]
    else:
        return None, None


def Face_vec(img, img_full_object_detections):
    # 人脸向量化
    shape = config.shape_predictor(img, img_full_object_detections)
    face_descriptor = config.face_net_model.compute_face_descriptor(img, shape)
    return np.array(face_descriptor)


def Face_close_check(face_vec1, face_vec2):
    # 人脸相似度
    # return np.linalg.norm(face_vec1 - face_vec2)
    return
