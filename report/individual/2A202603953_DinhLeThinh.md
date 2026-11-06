# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603953
- **Họ và tên:** ĐINH LÊ THỊNH
- **SEED:** 8057

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook có output | Có |
| [`lab1_report.json`](lab1_report.json) — báo cáo sinh từ notebook | Có |
| `hand_calc.jpg` — ảnh tính tay có các bước |  |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB (40 tin) | p95 (ms/tin) | USD / 1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.877 | 0.974 | 0.132 | 0.000 |
| 2. BoW + MLP | 0.945 | 0.947 | 0.347 | 0.000 |
| 3a. MiniLM + kNN | 0.875 | 1.000 | 72.775 | 0.000 |
| 3b. MiniLM + LogReg | 0.932 | 0.974 | 41.008 | 0.000 |
| 3c. MiniLM + MLP head | 0.932 | 0.947 | 41.926 | 0.000 |
| 4a. DeepSeek Flash zero-shot | 0.976 | 0.976 | 1,077.309 | 0.023 |
| 4b. DeepSeek V4 Pro zero-shot | 0.923 | 0.923 | 1,361.132 | 0.100 |
| 4c. DeepSeek Flash few-shot | 0.976 | 0.976 | 1,020.917 | 0.055 |

## 3. Dự đoán và thực tế

- **4.A — model lớn/nhỏ:** Dự đoán V4 Pro sẽ nhỉnh hơn Flash. Thực tế Flash đạt F1 **0.976**, cao hơn V4 Pro (**0.923**); Flash cũng rẻ hơn, khoảng **$0.023/1.000 tin** so với **$0.100/1.000 tin**. Dự đoán model lớn tốt hơn không đúng; kích thước model không đảm bảo kết quả tốt hơn trên tập này.
- **4.B — few-shot:** Dự đoán thêm ví dụ sẽ làm F1 tăng nhẹ và chi phí khoảng gấp đôi. F1 thực tế vẫn **0.976**, còn chi phí tăng từ khoảng **$0.023** lên **$0.055/1.000 tin** (xấp xỉ **2.34 lần**). Ví dụ trong prompt làm tăng chi phí nhưng không cải thiện F1 ở lần chạy này.
- **Stress test:** Đoán prompt injection là nhóm khó nhất với LLM. Thực tế LLM nhỏ đạt 100% ở nhóm prompt injection; nhóm khó nhất của nó là spam giọng tự nhiên (60% đúng với zero-shot, 80% với prompt phòng thủ). BoW cũng yếu ở né từ khóa, nhưng mức thấp nhất khác theo đầu phân loại: MLP đúng 0% nhóm né từ khóa, còn LogReg chỉ đúng 12% nhóm thao túng/lừa đảo.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1_SUB giảm từ **0.974** xuống **0.378** (giảm **0.596**). BoW + MLP giảm từ **0.947** xuống **0.439**. Trong khi đó, LLM nhỏ zero-shot giảm từ **0.976** xuống **0.926**, và prompt phòng thủ đạt **0.964**.

BoW dựa vào token riêng lẻ nên khó xử lý cách viết né từ khóa; MLP chỉ đúng 0% nhóm này, còn LogReg chỉ đúng 17%. LogReg đặc biệt yếu ở nhóm thao túng/lừa đảo (12%). Embedding giảm bớt vấn đề từ vựng, nhưng đầu phân loại vẫn ảnh hưởng: kNN chỉ đúng 25% nhóm “nói về spam”, còn LogReg và MLP chỉ đúng 38% nhóm thao túng/lừa đảo. LLM tốt hơn trên phần lớn nhóm nhưng dễ nhầm spam giọng tự nhiên.

Hai case mà tất cả model đều sai là **#13**, lời nhắc thân thiện về ảnh sự kiện kèm lời mời xem/đặt ảnh, và **#16**, lời nhắn giả làm bạn cũ để giới thiệu việc kinh doanh skincare. Cả hai dùng ngữ cảnh quen thuộc, không giống spam quảng cáo điển hình, nên các model đều bỏ sót ý định bán hàng. Prompt phòng thủ đạt 100% ở nhóm injection trong bộ này, nhưng không sửa được hai case spam giọng tự nhiên trên.

Test thường có điểm quá lạc quan vì các case được lấy theo phân phối quen thuộc, còn 50 case này cố tình tập trung vào bẫy và cách diễn đạt khác thường. Bộ stress test hữu ích để tìm failure mode và làm kiểm tra hồi quy; nó không đại diện cho tỷ lệ spam thật và không nên dùng một mình để ước lượng chất lượng triển khai.

## 5. Thử thách cá nhân


## 6. Phương án cho sản phẩm thật

Với SLA **p95 < 50 ms/tin**, các phương án đạt ngưỡng trong lần đo này là BoW + LogReg (**0.132 ms**), BoW + MLP (**0.347 ms**), MiniLM + LogReg (**41.008 ms**) và MiniLM + MLP (**41.926 ms**). Trong số đó, BoW + MLP có F1 test cao nhất (**0.945**) và throughput khoảng **29,685 tin/s**, cao hơn mức trung bình cần thiết khoảng **116 tin/s**.

Tuy nhiên, BoW + MLP chỉ đạt accuracy **0.54** và F1 **0.439** trên stress test, nên không nên triển khai đơn lẻ. Sẽ dùng nó làm bộ lọc nhanh ban đầu, kết hợp chuẩn hóa đầu vào và kiểm tra URL/người gửi; tin có độ tin cậy thấp sẽ được chuyển sang hàng đợi rà soát bất đồng bộ. MiniLM + LogReg bền hơn trên stress test (F1 **0.612**) và vẫn đạt p95 50 ms trong lần đo, nhưng throughput đo được chỉ khoảng **38 tin/s**, nên cần tối ưu hoặc chạy song song để đáp ứng tải.

LLM không phù hợp làm lớp đồng bộ nếu SLA là 50 ms: p95 đo được khoảng **1,021–1,361 ms**. Theo chi phí ước tính trong lần chạy, xử lý 10 triệu tin/ngày bằng Flash zero-shot tốn khoảng **$233/ngày** (xấp xỉ **$7,004/tháng** nếu tính 30 ngày); Flash few-shot khoảng **$545/ngày** (xấp xỉ **$16,364/tháng**). Đây là ngoại suy từ 40 tin SUB và giá/token cấu hình trong notebook, nên chi phí thực tế sẽ thay đổi theo độ dài tin và mức sử dụng cache.

## 7. AI log và điều đã học

- **Phần 4:** Hỏi cách cấu hình DeepSeek cho 4.A/4.B. Tự chạy API và đối chiếu F1, độ trễ, chi phí với output; không ghi API key vào báo cáo.
- **Phần 5:** Nnhờ AI tổng hợp bảng so sánh, Pareto và stress test thành phần phân tích. Đối chiếu các con số với output notebook và chỉnh lời giải thích theo kết quả của mình.

Hiểu rõ hơn rằng accuracy cao trên dữ liệu lệch lớp có thể che việc bỏ sót spam, nên cần xem precision, recall và F1. Cũng thấy model lớn hoặc few-shot không mặc nhiên tốt hơn: trên SUB 40, Flash zero-shot tốt hơn V4 Pro và few-shot không tăng F1. Cuối cùng, điểm test thường không phản ánh hết khả năng chịu các biến thể spam có chủ đích; cần kiểm tra thêm trên dữ liệu đại diện và các case khó.