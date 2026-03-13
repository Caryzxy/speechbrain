import random
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition
from tqdm import tqdm
import soundfile as sf

# =========================
# 1. SETTINGS
# =========================
DATA_ROOT = "/home/caryzxy/datasets/Audio_Speech_Actors_01-24"
FINETUNED_CKPT = "/home/caryzxy/projects/ravdess_speaker_id/exp_ecapa_finetune/finetuned_embedding_model.pt"
PRETRAINED_SAVEDIR = "/home/caryzxy/projects/ravdess_speaker_id/pretrained_ecapa"
SEED = 42
ENROLL_PER_SPEAKER = 6
DEVICE = "cpu"   # change to "cuda:0" if torch.cuda.is_available()

random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# 2. LOAD PRETRAINED + FINETUNED ENCODER
# =========================
verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=PRETRAINED_SAVEDIR,
    run_opts={"device": DEVICE},
)

# replace pretrained encoder weights with your fine-tuned ones
state = torch.load(FINETUNED_CKPT, map_location=DEVICE)
verifier.mods.embedding_model.load_state_dict(state)
verifier.mods.embedding_model.eval()
verifier.mods.embedding_model.to(DEVICE)


# =========================
# 3. HELPERS
# =========================
def parse_actor_id_from_folder(folder_name: str) -> str:
    return folder_name.split("_")[-1]


def collect_files(data_root: str):
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
    signal, sr = sf.read(audio_path, dtype="float32")

    # stereo -> mono
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, time]

    # resample to 16k if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
        sr = 16000

    return signal, sr


def extract_embedding(audio_path: str):
    signal, sr = load_audio(audio_path)
    signal = signal.to(DEVICE)

    with torch.no_grad():
        emb = verifier.encode_batch(signal)

    emb = emb.squeeze()
    emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu()


def split_enroll_test(speaker_to_files, enroll_per_speaker=6):
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
    centroids = {}

    for speaker, file_list in tqdm(enroll_dict.items(), desc="Building centroids"):
        embeddings = []
        for path in file_list:
            embeddings.append(extract_embedding(path))

        stacked = torch.stack(embeddings, dim=0)
        centroid = stacked.mean(dim=0)
        centroid = F.normalize(centroid, p=2, dim=0)
        centroids[speaker] = centroid

    return centroids


def predict_speaker(audio_path, centroids):
    emb = extract_embedding(audio_path)

    best_speaker = None
    best_score = -1e9

    for speaker, centroid in centroids.items():
        score = F.cosine_similarity(
            emb.unsqueeze(0), centroid.unsqueeze(0)
        ).item()
        if score > best_score:
            best_score = score
            best_speaker = speaker

    return best_speaker, best_score


def evaluate(test_dict, centroids):
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
# 4. MAIN
# =========================
def main():
    print(f"Using device: {DEVICE}")
    print("Collecting files...")
    speaker_to_files = collect_files(DATA_ROOT)

    print(f"Found {len(speaker_to_files)} speakers")
    for spk, files in speaker_to_files.items():
        print(f"Speaker {spk}: {len(files)} files")

    enroll_dict, test_dict = split_enroll_test(
        speaker_to_files, enroll_per_speaker=ENROLL_PER_SPEAKER
    )

    centroids = build_speaker_centroids(enroll_dict)
    accuracy, results = evaluate(test_dict, centroids)

    print(f"\nFine-tuned ECAPA cosine accuracy: {accuracy * 100:.2f}%")
    print("\nSample predictions:")
    for row in results[:10]:
        print(row)


if __name__ == "__main__":
    main()