import os

def get_image_paths(directory):
    image_paths = []
    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra xem tệp có phải là ảnh hay không (tùy thuộc vào phần mở rộng)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # Tạo đường dẫn đầy đủ của tệp ảnh
            full_path = os.path.join(directory, filename)
            image_paths.append(full_path)
    return image_paths

# Sử dụng hàm trên


# In danh sách các đường dẫn
if __name__=="__main__":
    directory = r'C:\Users\Dell M4800\Desktop\face\data'  # Đường dẫn tới thư mục chứa ảnh
    image_paths = get_image_paths(directory)
    print(image_paths)
