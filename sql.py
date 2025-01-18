import psycopg2
from utils import path_utils
import cv2
import insightface
import numpy as np
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640)) 
def main(name, vector_embbed):
    # Kết nối tới cơ sở dữ liệu
    conn = psycopg2.connect(
        dbname="face_recognition",
        user="kien",
        password="kien121299",
        host="localhost",
        port="5432"
    )
    
    cur = conn.cursor()
    
    # Tạo bảng
    cur.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        embedding FLOAT8[]
    )
    """)
    conn.commit()
    
    # Chèn dữ liệu
    cur.execute("""
    INSERT INTO faces (name, embedding) VALUES (%s, %s)
    """, (name, vector_embbed))
    conn.commit()
    
    # Truy vấn dữ liệu
    cur.execute("SELECT name FROM faces")
    names = cur.fetchall()
    
    for name in names:
        print(name[0])
    
    # Đóng kết nối
    cur.close()
    conn.close()

# 3. Hàm trích xuất vector nhúng (embedding) từ khuôn mặt
def get_face_embedding(face):
    # Mỗi đối tượng face chứa thông tin khuôn mặt, từ đó trích xuất embedding
    return face.embedding
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
def embedd(path):
    image_paths = path_utils.get_image_paths(directory=path)
    # 6. Sử dụng một tập ảnh để tính vector nhúng trung bình cho một người
    embedding = get_list_embedding(image_paths)
    return embedding
if __name__ == "__main__":
    path="./database/elon musk"
    vector_numpy=embedd(path)
    embedding_list = vector_numpy.tolist()
    name="Elon Musk"
    main(name, embedding_list)
