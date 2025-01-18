import psycopg2
import insightface
import numpy as np
from scipy.spatial.distance import cosine
import cv2
import time
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_size=(640, 640))# ctx_id=0 sử dụng GPU, 0 nếu bạn dùng CPU
def find_vector(new_embbed, nguong):
# Kết nối tới PostgreSQL
    conn = psycopg2.connect(
        dbname="face_recognition",
        user="kien",
        password="kien121299",
        host="localhost",
        port="5432"
    )

    cursor = conn.cursor()
    # Lấy vector nhúng từ bảng
    cursor.execute("SELECT id, name, embedding FROM faces")
    all_faces = cursor.fetchall()
    # tính các vector nhúng trong bảng và vector nhúng mới
    distant_min=[]
    for face in all_faces:
        db_id, db_name, db_embedding = face
        db_embedding = np.array(db_embedding)
        distance = cosine(db_embedding, new_embbed)
        if distance<nguong:
            if len(distant_min)==0:
                distant_min.append((distance,db_name))  # thêm vào list distant_min một cặp dist và name
            else:
                if distance < distant_min[0][0]:
                    distant_min[0]=(distance,db_name)
    return distant_min  # trả ra một list distant, nếu len(distant)==0 thì chứng tỏ khuôn mặt chưa có trong database và ngược lại
def extract_face(img):
    return model.get(img)
def extract_vector_embbedd(face):
    # trích xuất khuôn mặt
    embbed=face[0].embedding # nếu khung hình chỉ có một khuôn mặt
    return embbed  
    
if __name__=="__main__":
    
    nguong=0.5
    # đọc ảnh 
    start=time.time()
    path_img="./data_test/elon musk2.jpg"
    img=cv2.imread(path_img)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face=extract_face(img)
    if len(face)!=0:
         # tính vecto nhúng mới từ khuôn mặt mới
        new_embbed = extract_vector_embbedd(face)
        # gọi vecto nhúng từ cơ sở dữ liệu và tính cosine và trả ra list vector gần nhất
        distant=find_vector(new_embbed,nguong)
        if len(distant)!=0:
            _,name=distant[0]
            print(name)
            x1,y1,x2,y2=map(int,(face[0]['bbox'][0],face[0]['bbox'][1],face[0]['bbox'][2],face[0]['bbox'][3]))
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),1)
            cv2.putText(img,f"{name}",(x1-10,y1-10),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),1)
            
        else:
            print("chưa có khuôn mặt này trong database nên không nhận diện được")
    if len(face)==0: 
        print("không phát hiện đươc khuôn mặt")
    end=time.time()
    cv2.imshow("img",img)
    cv2.waitKey(0)
    print(f"thời gian thực thi: {end-start}")