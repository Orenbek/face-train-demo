from PIL import Image
from train.imgProcess import Face_detect, Face_vec


def main():
    # 1 读取图片
    originImg = Image.open("imgs/portrait-1.png")
    # 2 人脸检测
    img, img_full_object_detections = Face_detect(originImg)
    if img is None or img_full_object_detections is None:
        print("没有人脸")
        return
    # 3 人脸向量化
    face_vec = Face_vec(img, img_full_object_detections)
    print(face_vec)


if __name__ == '__main__':
    main()
