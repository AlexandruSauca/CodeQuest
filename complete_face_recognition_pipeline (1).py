import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

# --- CONFIGURARE ---
ONNX_MODEL_PATH = "best.onnx"
DB_PKL_PATH     = "face_recognition_model.pkl"
TEST_DATA_PATH  = "test"  # Folderul care conține Outdoor/Indoor

# Pragul de decizie (Threshold).
# Pentru FaceNet, de obicei 10.0 este un punct bun de pornire.
# Sub 10 = Aceeași persoană. Peste 10 = Unknown/Nesigur.
RECOGNITION_THRESHOLD = 11.0

class InferenceSystem:
    def __init__(self, onnx_path, db_path):
        print(f"Loading YOLO: {onnx_path}...")
        self.detector = YOLO(onnx_path, task='detect')

        print(f"Loading Face DB: {db_path}...")
        with open(db_path, 'rb') as f:
            self.face_db = pickle.load(f)

    def get_embedding(self, face_img):
        """Transformă fața decupată în vector (128 numere)"""
        try:
            # DeepFace așteaptă path, dar poate primi și numpy array direct
            results = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False, # Fața e deja decupată de YOLO
                detector_backend="skip"
            )
            return results[0]["embedding"]
        except:
            return None

    def find_match(self, target_embedding):
        """Compară vectorul curent cu TOȚI vectorii din baza de date"""
        best_name = "Unknown"
        min_distance = float('inf') # Infinit la început

        # Iterăm prin fiecare persoană din DB
        for name, db_embeddings_list in self.face_db.items():
            # O persoană poate avea mai multe vectori (indoor, outdoor)
            for db_emb in db_embeddings_list:
                # --- AICI E MATEMATICA ---
                # Calculăm Distanța Euclidiană (L2 Norm)
                dist = np.linalg.norm(np.array(target_embedding) - np.array(db_emb))

                # Reținem cea mai mică distanță găsită vreodată
                if dist < min_distance:
                    min_distance = dist
                    best_person_temp = name

        # Decizia finală bazată pe Threshold
        if min_distance < RECOGNITION_THRESHOLD:
            return best_person_temp, min_distance
        else:
            return "Unknown", min_distance

    def run(self):
        # Structura folderelor tale din dataset/test
        categories = [
            "Outdoor/Non-masked",
            "Indoor/Non-masked",
            "Outdoor/Masked",
            "Indoor/masked"
        ]

        for category in categories:
            folder_full_path = os.path.join(TEST_DATA_PATH, category)
            if not os.path.exists(folder_full_path):
                print(f"⚠️ Skip: Nu găsesc folderul {folder_full_path}")
                continue

            print(f"\n--- 📂 Testare Categorie: {category} ---")

            images = os.listdir(folder_full_path)
            for img_name in images:
                img_path = os.path.join(folder_full_path, img_name)

                # 1. Citire imagine
                frame = cv2.imread(img_path)
                if frame is None: continue

                # 2. Detecție fețe (YOLO ONNX)
                results = self.detector(frame, verbose=False)

                # Verificăm dacă YOLO a găsit ceva
                found_face = False
                for r in results:
                    for box in r.boxes:
                        found_face = True
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Decupăm fața
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0: continue

                        # 3. Generare Embedding (DeepFace)
                        emb = self.get_embedding(face_crop)

                        if emb is not None:
                            # 4. Comparație (Distanța Euclidiană)
                            name, dist = self.find_match(emb)

                            # Afișare rezultate
                            color = "🟢" if name != "Unknown" else "🔴"
                            print(f"{color} {img_name} -> {name} (Distanță: {dist:.2f})")
                        else:
                            print(f"⚠️ {img_name}: YOLO a văzut fața, dar DeepFace a eșuat.")

                if not found_face:
                    print(f"👻 {img_name}: Nicio față detectată de YOLO.")

if __name__ == "__main__":
    if os.path.exists(ONNX_MODEL_PATH) and os.path.exists(DB_PKL_PATH):
        system = InferenceSystem(ONNX_MODEL_PATH, DB_PKL_PATH)
        system.run()
    else:
        print("Lipsesc fișierele best.onnx sau face_verification_database.pkl")