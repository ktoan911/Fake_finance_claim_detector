import time

import psutil
from sentence_transformers import SentenceTransformer

# load model và ép chạy CPU
model = SentenceTransformer("/media/kateee/New Volume/Python/Python/social_media_crypto/Fake_Crypto_Claim_Detector/artifacts/retriever_model", device="cpu")


def generate_sentences(n):
    """
    Tạo n câu giả để test embedding
    """
    return [
        f"This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n This is test sentence number {i}\n "
        for i in range(n)
    ]


def test_max_batch(start=0, step=100, max_test=10000):
    """
    start: batch size bắt đầu
    step: mỗi lần tăng bao nhiêu
    max_test: giới hạn trên
    """

    for batch_size in range(start, max_test + step, step):
        sentences = generate_sentences(batch_size)

        try:
            start_time = time.time()

            embeddings = model.encode(
                sentences,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            end_time = time.time()

            ram = psutil.virtual_memory().percent

            print(
                f"Batch size: {batch_size} | "
                f"time: {end_time - start_time:.2f}s | "
                f"RAM usage: {ram}%"
            )

        except Exception as e:
            print(f"❌ Failed at batch size = {batch_size}")
            print(e)
            break


test_max_batch()
