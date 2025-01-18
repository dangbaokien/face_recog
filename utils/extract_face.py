import insightface
import cv2


# 1. Khởi tạo mô hình ArcFace
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 sử dụng GPU, 0 nếu bạn dùng CPU

# 2. Hàm trích xuất khuôn mặt từ ảnh
def extract_faces(image_path):
    img = cv2.imread(image_path)
    # Chuyển ảnh từ BGR sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(img_rgb)  # Trích xuất khuôn mặt từ ảnh
    return faces, img
if __name__ == "__main__":
    path= 'data/1.jpg'
    face,img = extract_faces(image_path=path)
    x1,y1,x2,y2= face[0].bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    face_img = img[y1:y2,x1:x2]
    print(face_img.shape)
    # face_img = cv2.resize(face_img,(320,320))
    cv2.imshow('face',face_img)
    cv2.waitKey(0)