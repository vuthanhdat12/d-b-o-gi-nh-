# Dự báo giá nhà bằng Python

## 1. Giới thiệu

Chương trình dự báo giá nhà dựa vào diện tích căn nhà.

Chương trình sử dụng Python thuần và không cần cài thêm thư viện.

Ví dụ:

- 50 m2 → khoảng 1500 triệu
- 100 m2 → khoảng 3000 triệu
- 150 m2 → khoảng 4500 triệu

## 2. Các file

### du_lieu_gia_nha.txt

Chứa dữ liệu diện tích và giá nhà.

Ví dụ:

30 900
40 1200
50 1500

Trong đó:

- Số thứ nhất: diện tích (m2)
- Số thứ hai: giá nhà (triệu đồng)

### du_bao_gia_nha.py

Là chương trình chính.

Chương trình sẽ:

1. Đọc dữ liệu từ file.
2. Tính giá trị trung bình.
3. Tính hệ số a.
4. Tính hệ số b.
5. Nhập diện tích cần dự báo.
6. Tính giá nhà dự báo.

## 3. Cách chạy

Đặt 3 file trong cùng một thư mục.

Mở CMD tại thư mục đó.

Chạy:

python du_bao_gia_nha.py

## 4. Ví dụ

Nhập:

100

Kết quả:

Diện tích: 100 m2
Giá dự báo: 3000 triệu đồng
Tương đương: 3.0 tỷ đồng

## 5. Công thức

Chương trình sử dụng công thức hồi quy tuyến tính:

Gia = a * Dien tich + b

Trong đó:

- a: hệ số
- b: hệ số chặn
- Diện tích: dữ liệu đầu vào
- Giá: giá nhà dự báo

## 6. Lưu ý

Đây là chương trình đơn giản phục vụ mục đích học tập.

Giá nhà thực tế còn phụ thuộc vào vị trí, số tầng, số phòng, đường giao thông và nhiều yếu tố khác.