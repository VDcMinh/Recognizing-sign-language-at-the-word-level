# Phân tích thư mục `data/datasets/WLASL/raw`

Ngày đọc thư mục: 2026-04-20  
Phạm vi: chỉ phân tích bản dữ liệu đang có trong workspace local.

## 1. Tổng quan

Thư mục `raw` là bản dữ liệu gốc/local của WLASL, gồm metadata mô tả nhãn, file chia tập train/val/test và các video `.mp4` đã tải về.

```text
data/datasets/WLASL/raw/
+-- docs/
|   +-- README.md
+-- metadata/
|   +-- WLASL_v0.3.json
|   +-- missing.txt
|   +-- nslt_100.json
|   +-- nslt_300.json
|   +-- nslt_1000.json
|   +-- nslt_2000.json
|   +-- wlasl_class_list.txt
+-- videos/
    +-- *.mp4
```

Có thể hiểu folder này theo 2 lớp:

| Lớp dữ liệu | File/thư mục | Vai trò |
| --- | --- | --- |
| Metadata gốc | `metadata/WLASL_v0.3.json` | Nguồn annotation đầy đủ nhất: gloss, video_id, split, bbox, frame, signer, source, URL. |
| File video local | `videos/*.mp4` | Video thực tế dùng để đọc frame, trích đặc trưng hoặc train model. |
| Trạng thái thiếu video | `metadata/missing.txt` | Danh sách `video_id` có trong metadata nhưng chưa có file `.mp4` local. |
| Mapping nhãn | `metadata/wlasl_class_list.txt` | Ánh xạ `class_id -> gloss`, dùng để giải mã nhãn số. |
| Manifest train classification | `metadata/nslt_*.json` | Các split đã chuẩn bị sẵn cho bài toán nhận dạng 100/300/1000/2000 lớp. |
| Tài liệu local | `docs/README.md` | Tóm tắt nhanh cấu trúc và thống kê dataset. |

## 2. Thống kê nhanh bản local

| Hạng mục | Số lượng / giá trị |
| --- | ---: |
| Số gloss trong master metadata | 2,000 |
| Số instance trong `WLASL_v0.3.json` | 21,083 |
| Số `video_id` duy nhất trong master metadata | 21,083 |
| Số signer duy nhất | 119 |
| Số nguồn dữ liệu | 19 |
| Số video `.mp4` có trong `videos/` | 11,980 |
| Số ID thiếu local, nằm trong `missing.txt` | 9,103 |
| Dung lượng thư mục `videos/` | khoảng 5.016 GB |
| File video nhỏ nhất | khoảng 0.013 MB |
| File video lớn nhất | khoảng 7.344 MB |
| Kích thước video trung bình | khoảng 0.429 MB |

Phân bố split trong master metadata:

| Split | Số instance |
| --- | ---: |
| `train` | 14,289 |
| `val` | 3,916 |
| `test` | 2,878 |

Độ phủ video local theo split:

| Split | Tổng instance | Có video local | Thiếu video local | Độ phủ |
| --- | ---: | ---: | ---: | ---: |
| `train` | 14,289 | 8,313 | 5,976 | 58.2% |
| `val` | 3,916 | 2,253 | 1,663 | 57.5% |
| `test` | 2,878 | 1,414 | 1,464 | 49.1% |

Điểm quan trọng: metadata gốc mô tả 21,083 instance, nhưng local chỉ có 11,980 file video. Vì vậy khi train/evaluate phải lọc theo file thật sự tồn tại hoặc xử lý danh sách thiếu trong `missing.txt`.

## 3. `docs/README.md`

File này là tài liệu tổng quan đã có sẵn trong folder. Nội dung chính:

- Mô tả đây là bản copy local của WLASL metadata kèm một phần video đã tải.
- Nêu `WLASL_v0.3.json` là nguồn annotation chính.
- Nêu các file `nslt_*.json` là manifest phục vụ bài toán classification.
- Ghi thống kê nhanh: số gloss, số instance, số video local, số ID missing, số signer, số source, dung lượng video.
- Ghi một số ghi chú nhất quán dữ liệu, ví dụ `missing.txt` khớp với phần video còn thiếu trong local.

Công dụng: dùng như trang giới thiệu nhanh cho người mới mở folder. Tuy nhiên README đang viết bằng tiếng Anh, còn file này diễn giải chi tiết hơn bằng tiếng Việt.

## 4. `metadata/WLASL_v0.3.json`

Đây là file metadata quan trọng nhất. Nếu cần một "source of truth" cho dữ liệu WLASL raw, nên bắt đầu từ file này.

### 4.1. Cấu trúc

File là một list JSON. Mỗi phần tử ứng với một gloss, tức một từ/cụm nhãn trong ASL.

Ví dụ rút gọn:

```json
[
  {
    "gloss": "book",
    "instances": [
      {
        "bbox": [385, 37, 885, 720],
        "fps": 25,
        "frame_end": -1,
        "frame_start": 1,
        "instance_id": 0,
        "signer_id": 118,
        "source": "aslbrick",
        "split": "train",
        "url": "http://aslbricks.org/New/ASL-Videos/book.mp4",
        "variation_id": 0,
        "video_id": "69241"
      }
    ]
  }
]
```

### 4.2. Ý nghĩa các trường

| Trường | Kiểu | Ý nghĩa / công dụng |
| --- | --- | --- |
| `gloss` | string | Nhãn dạng chữ, ví dụ `book`, `drink`, `computer`. Đây là label semantic của sign. |
| `instances` | list | Danh sách các video instance thuộc cùng gloss. Một gloss có nhiều video do nhiều signer/nguồn khác nhau. |
| `video_id` | string | ID định danh video. File local tương ứng nằm ở `videos/<video_id>.mp4`. |
| `instance_id` | number | ID instance trong manifest. Dùng để trace annotation. |
| `split` | string | Tập dữ liệu: `train`, `val`, hoặc `test`. Cần giữ nguyên khi train/evaluate để tránh rò rỉ dữ liệu. |
| `bbox` | list number | Bounding box vùng chứa signer/người ký trong frame. Thường dùng để crop hoặc tập trung model vào vùng tay/thân trên. |
| `fps` | number | Frame rate annotation. Trong bản này tất cả instance ghi `fps: 25`. |
| `frame_start` | number | Frame bắt đầu của đoạn sign trong video. |
| `frame_end` | number | Frame kết thúc. Giá trị `-1` thường được hiểu là không có frame kết thúc rõ ràng hoặc dùng đến hết clip. |
| `signer_id` | number | ID người ký. Hữu ích để kiểm tra phân bố người ký hoặc tránh bias. |
| `source` | string | Nguồn gốc video, ví dụ `signingsavvy`, `handspeak`, `aslpro`. |
| `url` | string | URL nguồn ban đầu. Không phải URL nào cũng là file mp4 trực tiếp; có cả YouTube hoặc SWF cũ. |
| `variation_id` | number | Biến thể của cùng một gloss. Đa số là `0`, một phần nhỏ là `1` hoặc `2`. |

### 4.3. Công dụng

`WLASL_v0.3.json` dùng cho:

- Liệt kê toàn bộ vocabulary 2,000 gloss.
- Biết một video thuộc nhãn nào.
- Biết video nằm trong split `train`, `val`, hay `test`.
- Map `video_id` sang file `.mp4` local.
- Crop video theo `bbox`.
- Cắt đoạn theo `frame_start` và `frame_end`.
- Phân tích nguồn dữ liệu, signer, số instance mỗi lớp.
- Tạo lại manifest training nếu không muốn dùng trực tiếp `nslt_*.json`.

### 4.4. Một vài đặc điểm dữ liệu

- Có 2,000 gloss.
- Có 21,083 instance.
- Mỗi gloss có ít nhất 6 instance và nhiều nhất 40 instance.
- Trung bình khoảng 10.54 instance cho mỗi gloss.
- Có 119 signer duy nhất.
- Có 19 source khác nhau.
- 18,958 instance có `frame_end = -1`.
- 2,125 instance có `frame_end` không âm.

Các source lớn nhất:

| Source | Số instance |
| --- | ---: |
| `signingsavvy` | 2,668 |
| `handspeak` | 2,211 |
| `signschool` | 1,968 |
| `aslsearch` | 1,875 |
| `asldeafined` | 1,833 |
| `aslu` | 1,827 |
| `aslpro` | 1,736 |
| `spreadthesign` | 1,584 |
| `asl5200` | 1,561 |
| `aslsignbank` | 1,071 |
| `asllex` | 814 |
| `startasl` | 623 |

## 5. `metadata/wlasl_class_list.txt`

Đây là file mapping từ class index sang gloss. File có 2,000 dòng, mỗi dòng gồm:

```text
<class_id>    <gloss>
```

Ví dụ đầu file:

```text
0    book
1    drink
2    computer
3    before
4    chair
```

Ví dụ cuối file:

```text
1995    washington
1996    waterfall
1997    weigh
1998    wheelchair
1999    whistle
```

Công dụng:

- Giải mã output model từ class ID sang gloss.
- Kiểm tra thứ tự nhãn khi train classification.
- Dùng chung với `nslt_*.json`, vì các file NSLT lưu nhãn bằng số class ID.

Lưu ý: thứ tự class trong file này khớp với thứ tự gloss trong `WLASL_v0.3.json`. Ví dụ class `0` là `book`, class `99` là `thursday`, class `299` là `money`, class `999` là `suggest`, class `1999` là `whistle`.

## 6. `metadata/nslt_*.json`

Các file này là manifest đã chuẩn bị sẵn cho bài toán nhận dạng sign theo số lớp khác nhau:

| File | Số entry | Số class | Class ID | Split counts |
| --- | ---: | ---: | --- | --- |
| `nslt_100.json` | 2,038 | 100 | `0..99` | test 258, train 1,442, val 338 |
| `nslt_300.json` | 5,118 | 300 | `0..299` | test 668, train 3,549, val 901 |
| `nslt_1000.json` | 13,174 | 1,000 | `0..999` | test 1,876, train 8,978, val 2,320 |
| `nslt_2000.json` | 21,095 | 2,000 | `0..1999` | test 2,879, train 14,296, val 3,920 |

### 6.1. Cấu trúc

Các file `nslt_*.json` không cùng schema với `WLASL_v0.3.json`. Chúng là dictionary/object với key là `video_id`.

Ví dụ:

```json
{
  "05237": {
    "subset": "train",
    "action": [77, 1, 55]
  }
}
```

Ý nghĩa:

| Trường | Ý nghĩa |
| --- | --- |
| Key `"05237"` | `video_id`, tương ứng file `videos/05237.mp4` nếu video tồn tại local. |
| `subset` | Split: `train`, `val`, hoặc `test`. |
| `action[0]` | Class ID, tra trong `wlasl_class_list.txt` để lấy gloss. |
| `action[1]` | Frame bắt đầu. |
| `action[2]` | Frame kết thúc. |

### 6.2. Công dụng

Các file này phù hợp khi muốn train model classification nhanh:

- Muốn thử bài toán nhỏ trước thì dùng `nslt_100.json`.
- Muốn tăng độ khó vừa phải thì dùng `nslt_300.json`.
- Muốn gần hơn với full dataset thì dùng `nslt_1000.json`.
- Muốn dùng đủ 2,000 gloss thì dùng `nslt_2000.json`.

Pipeline thường sẽ:

1. Chọn một file `nslt_*.json`.
2. Đọc từng key `video_id`.
3. Kiểm tra file `videos/<video_id>.mp4` có tồn tại không.
4. Dùng `action[0]` làm label số.
5. Dùng `wlasl_class_list.txt` để đổi label số sang gloss khi cần hiển thị/kết luận.
6. Dùng `subset` để chia train/val/test.

### 6.3. Quan hệ giữa các file NSLT

Các file này là các mức mở rộng lớp:

- `nslt_100.json` dùng class `0..99`.
- `nslt_300.json` dùng class `0..299`.
- `nslt_1000.json` dùng class `0..999`.
- `nslt_2000.json` dùng class `0..1999`.

Nói cách khác, đây là các cấu hình classification theo prefix của danh sách class. Điều này tiện cho thí nghiệm tăng dần độ khó: 100 lớp trước, sau đó 300, 1000 và 2000.

### 6.4. Điểm lệch cần biết

`nslt_2000.json` có 21,095 entry, nhiều hơn `WLASL_v0.3.json` 12 ID. Các ID có trong `nslt_2000.json` nhưng không xuất hiện trong master manifest:

```text
09500, 12209, 13422, 16096, 20065, 20138,
39347, 47639, 48251, 51153, 57839, 60721
```

Hàm ý:

- Nếu cần metadata giàu thông tin như `bbox`, `source`, `signer_id`, nên ưu tiên `WLASL_v0.3.json`.
- Nếu dùng `nslt_*.json` để train, cần kiểm tra file video tồn tại và chấp nhận rằng một số ID có thể không tra được ngược về master manifest.
- Khi cần dữ liệu sạch nhất cho phân tích, nên join theo intersection giữa `nslt_*.json`, `WLASL_v0.3.json` và `videos/`.

## 7. `metadata/missing.txt`

File này là danh sách `video_id` có trong master metadata nhưng không có file `.mp4` trong `videos/`.

Ví dụ đầu file:

```text
65225
68011
68208
68012
70212
```

Ví dụ cuối file:

```text
63049
63185
67065
63187
63189
```

Thống kê:

- Có 9,103 dòng.
- Mỗi dòng là một `video_id`.
- `video_id` trong file này tương ứng với file bị thiếu theo dạng `videos/<video_id>.mp4`.

Công dụng:

- Kiểm tra độ đầy đủ của bản download local.
- Lọc bỏ sample thiếu trước khi train.
- Tạo script download bổ sung nếu còn nguồn video hợp lệ.
- Báo cáo rõ ràng rằng local dataset không đủ 21,083 video như metadata gốc.

## 8. `videos/`

Thư mục này chứa video `.mp4` đã tải về. Cấu trúc là flat directory, không chia theo class hay split.

Ví dụ tên file:

```text
00335.mp4
00336.mp4
...
69546.mp4
69547.mp4
```

Quy ước:

```text
videos/<video_id>.mp4
```

Ví dụ:

- Metadata có `video_id = "69241"` thì file local tương ứng là `videos/69241.mp4`.
- Manifest NSLT có key `"05237"` thì file local tương ứng là `videos/05237.mp4`.

Công dụng:

- Nguồn dữ liệu hình ảnh thực tế cho model.
- Dùng để decode frame, lấy pose/keypoint, optical flow, crop signer, hoặc trích đặc trưng video.
- Dùng trong pipeline training/inference sau khi map với metadata.

Lưu ý:

- Không nên suy ra label từ tên file, vì tên file chỉ là ID.
- Muốn biết label phải tra `video_id` trong `WLASL_v0.3.json` hoặc `nslt_*.json`.
- Do local chỉ có 11,980 file, pipeline phải kiểm tra tồn tại file trước khi đọc.

Các file lớn nhất hiện có:

| File | Kích thước |
| --- | ---: |
| `69206.mp4` | 7.344 MB |
| `69412.mp4` | 7.334 MB |
| `69255.mp4` | 7.280 MB |
| `69225.mp4` | 7.146 MB |
| `69212.mp4` | 7.092 MB |
| `69311.mp4` | 7.036 MB |
| `69358.mp4` | 7.035 MB |
| `69269.mp4` | 6.983 MB |

## 9. Nên dùng file nào cho việc gì?

| Nhu cầu | Nên dùng | Lý do |
| --- | --- | --- |
| Hiểu toàn bộ dataset | `WLASL_v0.3.json` | Metadata đầy đủ nhất, có gloss, bbox, signer, source, URL. |
| Train classification nhanh | `nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, hoặc `nslt_2000.json` | Manifest đã có sẵn label số và split. |
| Giải mã class ID ra chữ | `wlasl_class_list.txt` | Mapping class index sang gloss. |
| Kiểm tra file video thiếu | `missing.txt` | Danh sách ID thiếu trong local. |
| Đọc frame/video thật | `videos/*.mp4` | Dữ liệu hình ảnh đầu vào cho model. |
| Viết báo cáo/tổng quan | `docs/README.md` và file phân tích này | Có số liệu và diễn giải cấu trúc. |

## 10. Gợi ý pipeline xử lý an toàn

### 10.1. Nếu train từ `nslt_*.json`

Quy trình nên làm:

1. Đọc manifest NSLT mong muốn.
2. Với từng `video_id`, kiểm tra `videos/<video_id>.mp4` tồn tại.
3. Nếu file thiếu, bỏ sample hoặc ghi log.
4. Dùng `subset` để chia train/val/test.
5. Dùng `action[0]` làm class label.
6. Dùng `action[1]`, `action[2]` để cắt frame nếu pipeline có hỗ trợ trim.
7. Dùng `wlasl_class_list.txt` để giải mã class ID khi evaluate.

Ưu điểm: đơn giản, phù hợp cho training classification.  
Nhược điểm: ít metadata hơn master manifest, và `nslt_2000.json` có một số ID không có trong `WLASL_v0.3.json`.

### 10.2. Nếu train từ `WLASL_v0.3.json`

Quy trình nên làm:

1. Flatten list gloss thành list instance.
2. Với mỗi instance, lấy `video_id`, `gloss`, `split`, `bbox`, `frame_start`, `frame_end`.
3. Kiểm tra file `videos/<video_id>.mp4` tồn tại.
4. Bỏ các instance thiếu video hoặc xử lý download bổ sung.
5. Tạo class mapping từ gloss sang class ID.
6. Train/evaluate theo split có sẵn.

Ưu điểm: metadata đầy đủ, dễ phân tích và kiểm soát dữ liệu.  
Nhược điểm: phải tự tạo manifest training dạng phẳng.

## 11. Các lưu ý cho luận văn/thí nghiệm

- Dataset local hiện không đầy đủ so với metadata gốc, chỉ có khoảng 56.8% video của master manifest.
- Tập test có độ phủ local thấp hơn train/val, chỉ khoảng 49.1%, nên kết quả đánh giá có thể lệch nếu chỉ dùng video local.
- Cần ghi rõ trong báo cáo rằng thí nghiệm dùng bản WLASL local gồm 11,980 video, không phải toàn bộ 21,083 instance trong metadata.
- Không nên trộn lại split nếu muốn so sánh với các kết quả theo chuẩn WLASL/NSLT.
- Nếu dùng `frame_end = -1`, nên quy ước rõ trong code: đọc đến hết video hoặc bỏ trimming end.
- `bbox` có thể dùng cho crop signer, nhưng nếu pipeline dùng pose/keypoint toàn frame thì vẫn nên giữ metadata này để có thể so sánh phương án crop và không crop.
- Vì nguồn video rất đa dạng, chất lượng, độ phân giải, background và signer có thể không đồng nhất. Đây là yếu tố quan trọng khi phân tích lỗi model.

## 12. Kết luận ngắn

Thư mục `data/datasets/WLASL/raw` là tầng dữ liệu gốc cho bài toán isolated sign language recognition với WLASL. Trong đó:

- `WLASL_v0.3.json` là metadata đầy đủ nhất và nên xem là nguồn chính.
- `videos/` là dữ liệu video local nhưng chưa đầy đủ.
- `missing.txt` mô tả phần còn thiếu giữa metadata và video local.
- `wlasl_class_list.txt` là mapping class ID sang gloss.
- `nslt_*.json` là các manifest classification tiện dùng cho thí nghiệm 100/300/1000/2000 lớp.

Nếu mục tiêu là xây dựng model, cách thực tế nhất là chọn một manifest `nslt_*.json`, lọc các video thật sự tồn tại trong `videos/`, dùng `wlasl_class_list.txt` để giải mã nhãn, và giữ nguyên split train/val/test để đảm bảo đánh giá nhất quán.
