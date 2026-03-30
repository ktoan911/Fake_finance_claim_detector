# Báo cáo đồ án tốt nghiệp

## Tóm tắt (Abstract)
Hệ thống này có chức năng đầu vào là 1 nhận định (claim) hệ thống sẽ tự tìm các nguồn tin liên quan và đưa vào mô hình để dự đoán độ xác thực của claim này, từ đó có thể nhận diện sớm các chiến dịch lừa đảo có quy mô rộng. Hệ thống được định hình bằng luồng quy trình (pipeline) bao gồm 3 khối kiến trúc chính: (1) Cơ chế Cơ sở Tri thức tĩnh và động , (2) Tinh chỉnh Mô hình Ngôn ngữ Lớn (LLM) thông qua phương pháp Low-Rank Adaptation (LoRA) chuyên biệt cho tác vụ tự hồi quy, và (3) Hệ thống Mạng nơ-ron Dung hợp dự đoán có trọng số (Confidence-Aware Fusion) đánh giá chéo giữa Lập luận của LLM và Đặc trưng Truy xuất. 

---

## 1. Thu thập Dữ liệu & Cấu trúc Cơ sở Tri thức (Knowledge Base Construction)

Cơ sở tri thức (KB) cung cấp nguồn "chân lý" để mô hình tham chiếu thông qua Kỹ thuật Truy xuất tăng cường sinh ngôn (RAG).

### 1.1. Phân tách Văn bản và Định dạng Vector (Chunking & Embedding)
Hệ thống xử lý định kỳ theo các _batch size_ để tương tác hiệu quả với OpenSearch.
1. **Tiền xử lý & Độ chia độ phân giải văn bản (Granularity Resolution)**: Ký tự rác (`\n, \t`) được loại bỏ dứt điểm. Nếu chiều dài của nội dung vượt mức `200 ký tự`, hệ thống thực thi thuật toán chia câu thông minh bằng biểu thức chính quy Regex (`(?<=[.!?])\s+`). Các câu sau đó được ráp tiếp nối sao cho kích thước của một phân đoạn (chunk) xấp xỉ nhưng không vượt quá 200 ký tự. Điều này giới hạn nhiễu ngữ nghĩa (semantic noise).
2. **Khởi tạo ID Phân khúc**: Mỗi chunk được tạo khóa chính định danh (ID) bằng thuật toán băm `hashlib.md5(article_url_chunk_idx).hexdigest()`, nhằm đảm bảo không ghi đè chồng chéo trong CSDL và xử lý song song không xung đột.
3. **Mã hóa Vector (Vector Encoding)**: Mọi chunk được đi qua Transformer `AITeamVN/Vietnamese_Embedding` giới hạn thực thi trên CPU. Kết quả trả về một ma trận nhúng `768` chiều, và đẩy trực tiếp thành vector trường (field_vector) trong hệ quản trị OpenSearch.

---

## 2. Huấn luyện Mô hình Ngôn ngữ Lớn với LoRA (LLM Tuning Process)

Mã nguồn định ra một thiết kế chuyên biệt để đóng gói LLM thành một phân loại viên (classifier) sử dụng nhân Causal LLM.

### 2.1. Kiến trúc Causal LM & Token Space
Mô hình nền tảng được ưu tiên là **Qwen3-4B-Instruct-2507**. Mô hình không được khởi tạo dưới hình dáng của sequence-classification mà là mô hình sinh từ tiếp theo (Causal LM).
Đầu vào văn bản được ánh xạ cẩn mật theo template `{claim}` và `{evidence}` (`"Dựa vào thông tin sau:\n{evidence}\n\nTuyên bố:\n{claim}\n=> Kết luận:"`). Hệ thống tự động giới hạn chiều dài bằng cách ép các câu Evidence bị cắt đi (Evidence Truncation) nếu không gian tổng kích thước input vượt qua `max_length = 256` hoặc cao hơn.

Hệ thống rút gọn không gian sinh của LLM thông qua ánh xạ cấu trúc nhãn đa lớp kích thước $C=3$: 
- Token ID đối chiếu: `0=Đúng`, `1=Sai`, `2=Thiếu (Not Enough Info)`.
- Hàm `_get_label_token_ids` kiểm tra bắt buộc mỗi chuỗi nhãn kể trên khi Tokenize phải luôn tạo ra đúng `1 Token` duy nhất nhằm khống chế logic phân loại chính xác, tránh hiện tượng LLM sinh ra multi-token.

### 2.2. Chi tiết Lớp thích ứng Hạng thấp (Low Rank Adaptation Matrix)
LoRA được thêm vào LLM để đóng băng trọng số trước huấn luyện của LLM ($W_0$) nhằm bảo tồn tri thức tổng quát và tiết kiệm dung lượng VRAM GPU. Thuật toán chèn thêm hạng ma trận $A$, $B$ ($W = W_0 + BA$).
Thông số hyperparameters tại mã nguồn cho tác vụ:
- **Thứ hạng phân rã (Rank - $r$)**: `16`.
- **Hệ số tăng cường (Scaling Alpha - $\alpha$)**: `32` (Scale factor $\frac{\alpha}{r} = 2.0$).
- **Dropout**: `0.05` chống quá khớp (Overfitting).
- **Target Modules**: Nhắm mục tiêu bão hòa vào toàn bộ khối Attention và khối MLP của kiến trúc Causal LM: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

### 2.3. Trích xuất Metrics và Hàm mục tiêu (Cross-Entropy & Đầu ra Tự hồi quy)

Khác với các kiến trúc phân loại chuỗi thông thường (Sequence Classification) như BERT, khi ứng dụng một mô hình sinh tự hồi quy (Causal LM - nhóm GPT) vào bài toán phân loại, việc trích xuất logits và tính toán hàm mục tiêu sẽ phức tạp hơn đáng kể.

**Quá trình Dịch chuyển Vị trí Logits (Logit Shifting Logic)**:
Trong các mô hình Causal LM, vector `Logit` tại vị trí $t$ mang xác suất dự đoán cho token tiếp theo ở vị trí $t+1$. 
Hàm `preprocess_logits_for_metrics` nhận một Tensor Logits ba chiều `[batch_size, sequence_length, vocab_size]` (ví dụ: `vocab_size` có thể lên tới 150,000+ token) cùng danh sách mảng `labels`. Hàm này thực hiện việc tìm kiếm vị trí của token nhãn (ví dụ "Đúng") thông qua mã chỉ mục (tìm phần tử khác `-100`, vì padding của PyTorch đánh dấu là `-100`):
$$ label\_pos = \min(\{t \mid labels[t] \neq -100\}) $$
Do vị trí token thực tế của nhãn là $label\_pos$, dự đoán của mô hình sinh ra từ token liền kề ngay trước đó phải nằm ở vị trí $pred\_pos = label\_pos - 1$. Vì thế, thay vì lưu toàn bộ vector Logits khổng lồ làm tiêu tốn bộ nhớ (nguồn gốc lỗi OOM trong quy trình Validation của transformers), hàm chỉ trích xuất Tensor logit cụ thể: `logits[:, pred_pos, :]`.

Sau đó, hàm tiếp tục lọc vector của toàn bộ từ vựng (vocabulary size) này chỉ lấy đúng 3 giá trị logit tương ứng với Token ID của 3 nhãn quy định `[Token(Đúng), Token(Sai), Token(Thiếu)]`. Nhờ thế, vector Logit khổng lồ được chuẩn hoá và nén cực tiểu về một Tensor vô cùng nhẹ có shape `[batch_size, 3]`.

**Hàm mục tiêu Huấn luyện (Objective Loss)**:
Quá trình tối ưu Loss vẫn được bảo toàn nguyên trạng bằng việc gọi tới Gradient của đối tượng `Trainer`: mô hình sử dụng **Cross-Entropy Loss** của Auto-Regressive LM, tính trung bình (mean) trên vùng các Token bị che (masking padding label `-100`). Với cấu trúc thiết kế Prompt, Loss của mạng sẽ dồn trị số trừng phạt mạnh nhất (Loss penalty) vào độ lệch xác suất dự đoán tại Token Kết luận cuối cùng.

**Các Hệ số Đánh giá (Evaluation Metrics)**:
Hàm `compute_metrics` tiếp nhận mảng Logits thu gọn `[batch_size, 3]` và tính Softmax dọc theo từng class. Thao tác biến đổi `np.argmax(probs)` được dùng để xác định nhãn dự đoán cuối cùng $y_{pred}$.
- Lõi hệ thống áp dụng thước đo F1-Score vĩ mô (**Macro F1**) làm mỏ neo kiểm định `metric_for_best_model`, chặn thay cho độ chính xác (Accuracy). Cụ thể, $F1_{macro} = \frac{F1_{Đúng} + F1_{Sai} + F1_{Thiếu}}{3}$. 
- Việc lựa chọn Macro-F1 là một thủ thuật thiết yếu chống lại hiện tượng Mất cân bằng Lớp phân phối (Imbalanced Classes) phổ biến trong dữ liệu dò Crypto, khi thực tiễn phần lớn dữ kiện cào về mang tính chất là `Sai` (Scam/Lừa đảo) và `Thiếu` (Chưa được kiểm chứng) thay vì các thông tin `Đúng` từ nội bộ quỹ.

---

## 3. Kiến trúc Huấn luyện Mạng Nơ-ron Dung hợp (Retrieval-LLM Fusion Architecture)

Dung hợp Nhận thức Độ tin cậy (Confidence-aware Fusion) là xương sống kỹ thuật được triển khai. Nó phá bỏ cách RAG cũ nơi LLM là người ra quyết định cuối cùng; ở đây, Lập luận của LLM và Trọng số Truy xuất sẽ "tranh luận" với nhau thông qua mạng trọng số $\beta$.

### 3.1. Kỹ nghệ Mạng Trích xuất Đặc trưng Truy xuất (Retrieval Feature Encoder)
Input của truy xuất là số lượng văn bản tìm thấy $K_{retrieved} = 10$. Mỗi tài liệu có $D_f = 4$ đặc trưng vật lý độc lập (Score BM25/Semantic, RRF, Recency, Cyclicity). Vector chứa là `[Batch_Size, 10, 4]`.
Hệ thống xây dựng một **Attention Gate** để lấy có chọn lọc trên 4 đặc trưng của từng tài liệu:
$$ a = \text{Softmax}(\text{Linear}_{[16 \to 1]}(\tanh(\text{Linear}_{[4 \to 16]}(F_{scores})))) $$
Điểm đặc trưng được trọng số hóa: $F_{weighted} = a \odot F_{scores}$.
Cuối cùng, tất cả được trải dài (Flatten) xuống $10 \times 4$ và cho đi qua **Encoder MLP**:
$$ E_{encoded} = \text{ReLU}(\text{Linear}_{[(10 \cdot 4) \to 64]}(F_{flat})) \rightarrow \text{Linear}_{[64 \to 64]} $$

### 3.2. Không gian Không Tương tác của Khối Dung hợp (Confidence-Aware Fusion Block)
Khối `ConfidenceAwareFusion` thiết lập một lớp chiếu $MLP_{ret}$ để map đặc trưng Encode truy xuất $E_{encoded}$ sang không gian logit nhãn $C$ chiều (số nhãn).
Logit hợp nhất cuối cùng tính theo công thức kỳ vọng (Equation 2 trong Code):
$$ \text{Logit}_{final} = \beta \cdot p_{LLM} + (1 - \beta) \cdot \text{MLP}_{ret}(E_{encoded}) $$
- $p_{LLM}$: Logit Vector đầu ra của Mô hình Ngôn ngữ ($C$ chiều).
- Khởi tạo giá trị $\beta$: Để đảm bảo $\beta \in [0, 1]$, code cài đặt hàm đảo nghịch Sigmoid: $\beta_{logit} = \ln(\frac{\beta_{init}}{1 - \beta_{init}})$. Mặc định $\beta_{init} = 0.8$, ưu tiên niềm tin của hệ thống vào LLM trong chặng đầu rèn luyện thuật toán.

### 3.3. Tối ưu Bộ nhớ và Cân bằng Sự mất mát (Memory Optimized Micro-Batch Training)

1. **Pre-computation (Xử lý trước)**: Đóng băng lớp tham số LLM (`model.eval(), requires_grad=False`), và sử dụng **Micro-Batching** kết hợp Cuda BF16 để trích xuất sạch toàn bộ hệ số dự đoán logits của LLM $p_{LLM}$ đối với từng dữ liệu Training trước khi thực thụ khởi động quá trình tối ưu Optimizer lên bộ mạng Dung Môi (`fusion layer` & `retrieval encoder`). Sau khi có tensor offline, `del llm` và giải phóng GPU.
2. **Hàm mục tiêu Regularization CE Loss**: Do dữ liệu huấn luyện thường mất cân bằng (Ví dụ `False=62%`, `True=38%`), Cross-Entropy được tính gia trọng bằng Tần số nghịch đảo (Inverse-frequency weighting).
Hàm mục tiêu bao hàm phương trình $\beta$-Regularization nhằm tạo ra lực đẩy ngược chiều, trừng phạt mạng lưới nếu nó chây ì giữ nguyên vẹn $\beta \to 1$:
$$ \mathcal{L} = \mathcal{L}_{CE}(y, \text{Logit}_{final}, w_{class}) + \lambda_{reg} \|\beta\|^2 $$
(Với $\lambda_{reg} = 0.01$).

---

## 4. Quá trình Suy luận Trọn vẹn (End-to-end Inference Pipeline)

Hệ thống phối hợp toàn bộ linh kiện của kiến trúc tạo nên quy trình dò tìm theo thời gian thực (Real-time fact checking). Toàn bộ Logic gồm các bước:

**Bước 1: Nhận diện Độ lai ghép của Truy vấn (Query Verification)**
Hàm `_is_verbatim_query` kiểm tra xem claim người dùng nhập vào là văn bản dài (đoạn quote dài hơn `120 ký tự` hoặc có breakline, `>20 token`). Nếu chỉ là cụm từ ngắn cần phân tích mở rộng (`Expand Query`) sử dụng một đối tượng Generative Query Expander để diễn giải đồng nghĩa câu trước khi tra thông tin qua OpenSearchBM25.

**Bước 2: Truy xuất OpenSearch và Trộn điểm RRF**
Thực thi cùng lúc BM25 (`title^3, description^2, content`) và Vector Semantic. Nếu Query là nguyên văn mảng dài (Verbatim), tăng trọng số phụ BM25 Factor: $W_{bm25}=1.35$.
Sử dụng thuật toán **Khôi phục hạng đối ứng RRF**:
$$ \text{RRF}(d) = \frac{W_{bm25}}{60 + \text{Rank}_{bm25}(d)} + \frac{W_{vec}}{60 + \text{Rank}_{vec}(d)} $$
RRF sau đó được Normalize chuẩn hoá chia cho Max RRF sinh ra phân phối từ 0 tới 1.

**Bước 3: Hàm Suy giảm Dựa trên Độ Cũ mới (Temporal Scorer)**
Các hệ số ngày xuất bản bằng chứng `published_at` trả về điểm chênh lệch thời gian (Age_hours). `TemporalScorer` áp dụng Decaying function lấy $\lambda$ là động học (Adaptive). Biến thiên chu kỳ của tin tức (Cyclicity_Score) và độ mới (Recency_Score) kết hợp với hệ số phân rã trả ra điểm thời gian $Temporal$.
Lúc này, Mảng Đặc trưng của $K=10$ tài liệu đã đầy đủ.

**Bước 4: Đồng Trích dẫn và Dung hợp (Predict)**
1. Chọn top $M=3$ đoạn trích Evidence chữ text dài (Tránh Context Limit 2048 Tokens) và cho cào vào LLM đã tuning LoRA $\rightarrow$ Kết xuất được một Logit Dự đoán Causal Tensor theo đúng vị trí Label Token.
2. Các Vector điểm đặc trưng K=10 được Mạng `RetrievalFeatureEncoder` đưa sang không gian Hidden 64 chiều.
3. Chạy quá trình Mạng Tích Hợp Cuối Cùng (`ConfidenceAwareFusion` Forward Pass): Thu nhận điểm phân loại cuối cùng thông qua `Softmax(FusedLogits)` và định dạng lại chuỗi thành một đánh giá có chỉ số độ tin cậy (Confidence metric). 

## 6. Ứng dụng Thực tiễn (Practical Application)

Giải pháp phân loại tuyên bố ảo của dự án Fake Crypto Claim Detector không chỉ dừng lại tại mức kiểm chứng thuật toán, mà đã được thiết kế mở rộng thành một đường ống ứng dụng hoàn chỉnh, tích hợp qua giao diện dòng lệnh (CLI).

Luồng ứng dụng thiết lập tự động hóa quy trình theo các bước:

1. **Thu thập Dữ liệu Đa hình trên Nền tảng (Data Crawling)**: Hệ thống tự động quét và bóc tách các bài viết từ nền tảng Facebook, tự động điều hướng cấu trúc theo dạng Trang cá nhân (Page/Profile) hoặc Nhóm (Group). Hệ thống hỗ trợ khởi chạy hàng loạt thông qua danh sách URL nguồn lập trình sẵn trong tệp văn bản tĩnh.
2. **Kiểm soát Khối lượng và Ngưỡng Tương tác (Batch Processing & Thresholding)**: Dữ liệu thô từ mạng xã hội được tải về theo cơ chế lô nhỏ (Batch size), kết hợp với cấu hình độ trễ nhằm đảm bảo tính ổn định tối đa cho nguồn thu.
3. **Phân loại Sự thật Tự động (Automated Fact-Checking Pipeline)**: Tập lệnh thông điệp (claims) trích xuất từ văn bản bài viết sẽ được bơm trực tiếp vào hệ thống Mạng Dung hợp Suy luận Khép kín. Các cơ chế thành phần bao gồm Khối Nhúng Semantic (Vietnamese_Embedding), Cơ sở Truy xuất (OpenSearch Knowledge Base) và Mô hình Sinh Ngôn ngữ Lớn (LLM qua tinh chỉnh LoRA) được phối hợp đồng thời để đưa ra phán quyết cuối cùng về nội dung tiền mã hoá.
4. **Kết quả Hệ thống Giám sát (Supervisory Output Export)**: 
Kết quả của quá trình phân loại, những bài đăng nào được gán nhãn sai hoặc thiếu thông tin được phân cụm để dự báo trước các chiến dịch lừa đảo theo quy mô
