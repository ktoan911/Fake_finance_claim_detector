import time
import argparse
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
def test_max_batch(model, start, step, max_test):

    for batch_size in range(start, max_test + step, step):

        sentences = generate_sentences(batch_size)

        try:
            start_time = time.time()

            embeddings = model.encode(
                sentences,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            end_time = time.time()

            ram = psutil.virtual_memory().percent

            print(
                f"Batch size: {batch_size} | "
                f"time: {end_time-start_time:.2f}s | "
                f"RAM usage: {ram}%"
            )

        except Exception as e:
            print(f"\n❌ Failed at batch size = {batch_size}")
            print(e)
            break


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True,
                        help="sentence transformer model name or path")

    parser.add_argument("--start", type=int, default=100,
                        help="starting batch size")

    parser.add_argument("--step", type=int, default=100,
                        help="batch increase step")

    parser.add_argument("--max_test", type=int, default=10000,
                        help="maximum batch size to test")

    args = parser.parse_args()

    print(f"\nLoading model: {args.model}")
    model = SentenceTransformer(args.model, device="cpu")

    print("\nStart benchmarking...\n")

    test_max_batch(
        model=model,
        start=args.start,
        step=args.step,
        max_test=args.max_test
    )


if __name__ == "__main__":
    main()