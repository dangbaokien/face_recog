import insightface
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from utils import path_utils

# 1. Khởi tạo mô hình ArcFace
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 sử dụng GPU, 0 nếu bạn dùng CPU

# 2. Hàm trích xuất khuôn mặt từ ảnh
def extract_faces(image):
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = model.get(image)  # Trích xuất khuôn mặt từ ảnh
    return faces

# 3. Hàm trích xuất vector nhúng (embedding) từ khuôn mặt
def get_face_embedding(face):
    # Mỗi đối tượng face chứa thông tin khuôn mặt, từ đó trích xuất embedding
    return face.embedding

# 4. Tính vector nhúng trung bình từ nhiều ảnh
def get_list_embedding(image_paths):
    embeddings = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = model.get(img)
        for face in faces:
            embedding = get_face_embedding(face)
            embeddings.append(embedding)
    # Tính trung bình của các vector nhúng
    emb = np.mean(embeddings, axis=0)
    return emb

# 5. Hàm so sánh vector nhúng
def compare_embeddings(nguong = 0.5, embedding_1=None, embedding2=None):

    distant = cosine(embedding_1, embedding2)
    #print("distant",distant)
    if distant<nguong:
        return 1
    else: 
        return 0

def embedd():
    direct_path_image = r'C:\Users\Dell M4800\Desktop\face\data'
    image_paths = path_utils.get_image_paths(directory=direct_path_image)
    # 6. Sử dụng một tập ảnh để tính vector nhúng trung bình cho một người
    embedding = get_list_embedding(image_paths)
    return embedding
# 7. Đưa ảnh mới vào và so sánh

if __name__ == "__main__":
    new_image_path = 'data_test/persion3.jpg'  # Ảnh mới cần nhận dạng
    face, img = extract_faces(new_image_path)

    # list_embedding = get_list_embedding(face)
    # So sánh vector nhúng mới với vector nhúng trung bình
    embedding_list = embedd()
    new_image_embedd = get_face_embedding(face=face[0])

    distan = compare_embeddings(embedding_1=embedding_list, embedding2=new_image_embedd)
    
    if distan == 1:  # Ngưỡng tùy chỉnh để nhận diện
        print("Người này là tôi!")
    else:
        print("Không phải tôi.")

    x1,y1,x2,y2= face[0].bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    face_img = img[y1:y2,x1:x2]
    # face_img = cv2.resize(face_img,(320,320))
    cv2.imshow('face',face_img)
    cv2.waitKey(0)
