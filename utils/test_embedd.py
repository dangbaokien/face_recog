import insightface
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import path_utils
import time
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

# 3. Hàm trích xuất vector nhúng (embedding) từ khuôn mặt
def get_face_embedding(face):
    # Mỗi đối tượng face chứa thông tin khuôn mặt, từ đó trích xuất embedding
    return face.embedding

# 4. Tính vector nhúng trung bình từ nhiều ảnh
def get_average_embedding(image_paths):
    embeddings = []
    for image_path in image_paths:
        faces, img = extract_faces(image_path)
        for face in faces:
            embedding = get_face_embedding(face)
            embeddings.append(embedding)
    # Tính trung bình của các vector nhúng
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

# 5. Hàm so sánh vector nhúng
def compare_embeddings(embedding1, embedding2):
    return cosine(embedding1, embedding2)

# 6. Sử dụng một tập ảnh để tính vector nhúng trung bình cho một người
direct_path_image = r'C:\Users\Dell M4800\Desktop\face\data'
image_paths = path_utils.get_image_paths(directory=direct_path_image)
avg_embedding = get_average_embedding(image_paths)
#print("Vector nhúng trung bình của người này:", avg_embedding)

# 7. Đưa ảnh mới vào và so sánh
if __name__ == "__main__":
    start=time.time()
    new_image_path = './data_test/kien.jpg'  # Ảnh mới cần nhận dạng
    faces, img = extract_faces(new_image_path)
    for face in faces:
        new_embedding = get_face_embedding(face)
        # So sánh vector nhúng mới với vector nhúng trung bình
        distance = compare_embeddings(avg_embedding, new_embedding)
        #print(f"Khoảng cách cosine: {distance}")
        if distance < 0.5:  # Ngưỡng tùy chỉnh để nhận diện
            print("Người này là người đã đăng ký!")
        else:
            print("Không nhận diện được người này.")
    end=time.time()
    print(f"thời gian thực thi: {end-start}")