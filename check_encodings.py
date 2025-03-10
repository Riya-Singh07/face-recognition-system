import pickle

try:
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"✅ Loaded {len(data['encodings'])} face encodings.")
    print(f"✅ Found names: {set(data['names'])}")
except FileNotFoundError:
    print("❌ Error: encodings.pkl not found! Run encode_faces.py first.")
except Exception as e:
    print(f"❌ Error loading encodings: {e}")
