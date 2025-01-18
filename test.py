from utils import extract_faces, get_list_embedding, compare_embeddings, embedd,get_face_embedding
import time
import cv2
import numpy as np

def recog_image(face):
    distan=0
    embedding_list = embedd()
    new_image_embedd = get_face_embedding(face=face[0])

    distan = compare_embeddings(0.5 ,embedding_list, new_image_embedd)
    if distan == 1:  # Ngưỡng tùy chỉnh để nhận diện
        print("đây đúng là tôi ")
    else:
        print("Không phải tôi.")
    return distan

def draw_bounding_box(image,face):
    # Lấy tọa độ bounding box
    x1,y1,x2,y2=map(int,(face[0]['bbox'][0],face[0]['bbox'][1],face[0]['bbox'][2],face[0]['bbox'][3]))

    # Vẽ bounding box
    # Tạo hiệu ứng nhấp nháy
    for i in range(3):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0) if i % 2 == 0 else (0, 0, 0), 2)
        cv2.imshow('Webcam', image)
        cv2.waitKey(300)

    return image,(x1,y1,x2,y2)
# def draw_rays(image, bbox):
#     x1, y1, x2, y2 = bbox
#     length = 30  # Độ dài của tia

#     # Vẽ các tia từ góc bounding box
#     points = [
#         (x1, y1), (x2, y1), (x2, y2), (x1, y2)
#     ]
    
#     for point in points:
#         end_point = (
#             int(point[0] + length * np.sign(np.random.rand() - 0.5)),
#             int(point[1] + length * np.sign(np.random.rand() - 0.5))
#         )
#         cv2.line(image, point, end_point, (0, 255, 0), 2)

#     return image
def zoom_on_face(image, box):
    x1, y1, x2, y2 = box
    face_image = image[y1:y2, x1:x2]
    return face_image
def draw_backgroud():
    bg_width, bg_height = 1080,720
    background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 255
    cv2.imshow('background',background)
    cv2.moveWindow('background',440,0)
    cv2.waitKey(0)
if __name__=="__main__":
    # start_time=time.time()
    # test()
    # end_time=time.time()
    # print("thời gian thực thi: ",end_time-start_time)
    cap=cv2.VideoCapture(0)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam',640,480)
    cv2.moveWindow('Webcam', 440,0) 
    # draw_backgroud()
    while True:
        success,image=cap.read()
        #print("loại image:",image.dtype)
        if success:
            # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # face = extract_faces(image)  #trích xuất khuôn mặt
            # if len(face)==0:
            #     continue
            # distan=recog_image(face)    # tính khoảng cách nhúng khuôn mặt
            # if distan == 1: 
            #     #nếu phát hiện được khuôn mặt thì video đứng yên tại khung hình phát hiện và vẽ box vào khung hình
            #     image,box=draw_bounding_box(image=image,face=face)  #vẽ box vào image
            #     # image=draw_rays(image,box)  #vẽ line
            #     face_image=zoom_on_face(image,box)  #vẽ khuôn mặt

            #     cv2.imshow('face',face_image)
            #     cv2.waitKey(0)
            #     break
            cv2.imshow('Webcam',image)
            cv2.waitKey(1)
