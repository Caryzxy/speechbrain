import os
import random
from pathlib import Path
from collections import defaultdict

import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm




# =========================
# 1. SETTINGS
# =========================
DATA_ROOT = r"/home/caryzxy/datasets/Audio_Speech_Actors_01-24"   # change this
SEED = 42
ENROLL_PER_SPEAKER = 6   # number of enrollment files per actor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. REPRODUCIBILITY
# =========================
random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# 3. LOAD PRETRAINED MODEL
# =========================
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)


# =========================
# 4. HELPER FUNCTIONS
# =========================
def parse_actor_id_from_folder(folder_name: str) -> str:
    """
    Folder example: Actor_01 -> returns '01'
    """
    return folder_name.split("_")[-1]


def collect_files(data_root: str):
    """
    Walk through Actor_XX folders and collect wav file paths.
    Returns a dict:
        {
            '01': [path1, path2, ...],
            '02': [path1, path2, ...],
            ...
        }
    """
    speaker_to_files = defaultdict(list)
    root = Path(data_root)

    for actor_dir in sorted(root.glob("Actor_*")):
        if actor_dir.is_dir():
            actor_id = parse_actor_id_from_folder(actor_dir.name)
            wavs = sorted(actor_dir.glob("*.wav"))
            for wav_path in wavs:
                speaker_to_files[actor_id].append(str(wav_path))

    return speaker_to_files



def load_audio(audio_path: str):
    """
    Load audio with soundfile instead of torchaudio.load(),
    so we avoid TorchCodec on Windows.
    Returns:
        signal: torch.Tensor of shape [1, time]
        sr: sample rate
    """
    signal, sr = sf.read(audio_path, dtype="float32")

    # If stereo, average to mono
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, time]

    # ECAPA models are commonly run at 16 kHz; resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
        sr = 16000

    return signal, sr


def extract_embedding(audio_path: str):
    """
    Extract a normalized speaker embedding for one audio file.
    """
    signal, sr = load_audio(audio_path)

    # Encode batch expects waveform batch
    signal = signal.to(DEVICE)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    # Output shape often [1, 1, emb_dim] or [1, emb_dim]
    embedding = embedding.squeeze()

    # Normalize for cosine similarity
    embedding = F.normalize(embedding, p=2, dim=0)

    return embedding.cpu()


def split_enroll_test(speaker_to_files, enroll_per_speaker=6):
    """
    For each speaker:
      - shuffle files
      - take first N as enrollment
      - rest as test
    """
    enroll = {}
    test = {}

    for speaker, files in speaker_to_files.items():
        files = files.copy()
        random.shuffle(files)

        if len(files) <= enroll_per_speaker:
            raise ValueError(
                f"Speaker {speaker} has only {len(files)} files, "
                f"not enough for enroll_per_speaker={enroll_per_speaker}"
            )

        enroll[speaker] = files[:enroll_per_speaker]
        test[speaker] = files[enroll_per_speaker:]

    return enroll, test


def build_speaker_centroids(enroll_dict):
    """
    Average enrollment embeddings to get one centroid per speaker.
    """
    centroids = {}

    for speaker, file_list in tqdm(enroll_dict.items(), desc="Building centroids"):
        embeddings = []
        for path in file_list:
            emb = extract_embedding(path)
            embeddings.append(emb)

        stacked = torch.stack(embeddings, dim=0)   # [N, emb_dim]
        centroid = stacked.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        centroids[speaker] = centroid

    return centroids


def predict_speaker(audio_path, centroids):
    """
    Predict speaker by cosine similarity to speaker centroids.
    """
    emb = extract_embedding(audio_path)

    best_speaker = None
    best_score = -1e9

    for speaker, centroid in centroids.items():
        score = F.cosine_similarity(emb.unsqueeze(0), centroid.unsqueeze(0)).item()
        if score > best_score:
            best_score = score
            best_speaker = speaker

    return best_speaker, best_score


def evaluate(test_dict, centroids):
    """
    Evaluate identification accuracy on all test files.
    """
    total = 0
    correct = 0
    results = []

    for true_speaker, file_list in tqdm(test_dict.items(), desc="Evaluating"):
        for path in file_list:
            pred_speaker, score = predict_speaker(path, centroids)
            is_correct = int(pred_speaker == true_speaker)

            total += 1
            correct += is_correct

            results.append({
                "file": path,
                "true_speaker": true_speaker,
                "pred_speaker": pred_speaker,
                "score": score,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, results


# =========================
# 5. MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print("Collecting files...")

    speaker_to_files = collect_files(DATA_ROOT)

    print(f"Found {len(speaker_to_files)} speakers")
    for spk, files in speaker_to_files.items():
        print(f"Speaker {spk}: {len(files)} files")

    print("\nSplitting enrollment and test sets...")
    enroll_dict, test_dict = split_enroll_test(
        speaker_to_files,
        enroll_per_speaker=ENROLL_PER_SPEAKER
    )

    print("Building speaker centroids...")
    centroids = build_speaker_centroids(enroll_dict)

    print("Running evaluation...")
    accuracy, results = evaluate(test_dict, centroids)

    print(f"\nSpeaker identification accuracy: {accuracy * 100:.2f}%")
    print("\nSome sample predictions:")
    for row in results[:10]:
        print(row)


if __name__ == "__main__":
    main()