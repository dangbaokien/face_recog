import psycopg2
import insightface
import numpy as np
from scipy.spatial.distance import cosine
import cv2
import time

def connect_to_db():
# Kết nối tới PostgreSQL
    conn = psycopg2.connect(
        dbname="face_recognition",
        user="kien",
        password="kien121299",
        host="localhost",
        port="5432"
    )
    return conn
# Hàm lấy danh sách embeddings từ cơ sở dữ liệu
def get_embeddings_from_db():
    conn = connect_to_db()
    cursor = conn.cursor()
    # Lấy vector nhúng từ bảng
    cursor.execute("SELECT id, name, embedding FROM faces")
    all_faces = cursor.fetchall()
    embeddings = {}
    for row in all_faces:
        embeddings[row[0]] = {
            'name': row[1],
            'embedding': np.array(row[2])
        }
    cursor.close()
    conn.close()
    return embeddings
# Hàm nhận diện nhiều khuôn mặt trong video và vẽ hộp
def recognize_faces(video_source,iswrite):
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(640, 640))# ctx_id=0 sử dụng GPU, 0 nếu bạn dùng CPU
    # Kết nối đến cơ sở dữ liệu và lấy các vector nhúng
    embeddings_db = get_embeddings_from_db()
    #Mở video
    cap = cv2.VideoCapture(video_source)
    if iswrite:
        out=write_video(cap)

    cnt=0
    while cap.isOpened():
        start =time.time()
        ret, frame = cap.read()
        frame= cv2.flip(frame, 1)
        if cnt%3!=0: 
            cnt+=1
            continue
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = model.get(rgb_frame)
        if len(faces)==0:
            cv2.imshow("video",frame)
            cv2.waitKey(1) 
            continue
        for face in faces:
            # Vẽ hộp quanh khuôn mặt
            x1,y1,x2,y2=map(int,(face['bbox'][0],face['bbox'][1],face['bbox'][2],face['bbox'][3]))
            # Lấy tọa độ hộp (x1, y1, x2, y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),1)
            # Lấy vector nhúng khuôn mặt
            face_embedding = face.embedding
            # Biến lưu tên của người nhận diện
            matched_name = "Unknown"
            # So sánh với tất cả các vector nhúng trong cơ sở dữ liệu
            min_distance = 0.5
            for id, data in embeddings_db.items():
                db_embedding = data['embedding']
                distance = cosine(db_embedding,face_embedding)
                # Nếu khoảng cách nhỏ hơn khoảng cách tối thiểu hiện tại, cập nhật tên
                if distance < min_distance:
                    min_distance = distance
                    matched_name = data['name']
            cv2.putText(frame, matched_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        cnt+=1
        cv2.imshow("image",frame)
        end=time.time()
        print(f"thời gian thực thi: {end-start}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    out.release()
    cv2.destroyAllWindows()
def write_video(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Chiều rộng khung hình
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_filename = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # Codec XVID
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return out
if __name__=="__main__":
    video_path="./data_test/trump.mp4"
    iswrite=True
    recognize_faces(video_path,iswrite)

