from datasets import load_dataset
import random


def main():
    print("📥 Loading dataset from HuggingFace: 'QizhiPei/e3fp-chebi-molgen' ...")
    try:
        ds = load_dataset("QizhiPei/e3fp-chebi-molgen", split="train")
    except Exception as e:
        print(f"❌ Failed to load dataset. Please ensure you have run 'huggingface-cli login'.\nError: {e}")
        return

    print(f"✅ Dataset loaded! Total train samples: {len(ds)}")
    print(f"📊 Features (Columns): {ds.column_names}")

    print("\n" + "=" * 60)
    print("🔍 Inspecting 3 Random Examples:")
    print("=" * 60)

    indices = random.sample(range(len(ds)), 3)
    for idx in indices:
        item = ds[idx]
        print(f"\n--- Example {idx} ---")
        for key in ds.column_names:
            print(f"[{key}]: {item[key]}")


if __name__ == "__main__":
    main()