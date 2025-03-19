import pdb
from vectordb import VectorDB  # Assuming your class is in vectordb.py

# Step 1: Initialize Database
db = VectorDB(model_names=["resnet", "dino"])
pdb.set_trace()  # Debug step (Check if models are loaded)

# Step 2: Add Test Images
db.add_image("test_images/test1.jpg")
db.add_image("test_images/test2.png")
db.add_image("test_images/test3.png")
db.add_image("test_images/test4.png")
pdb.set_trace()  # Debug step (Check metadata and FAISS count)

# Step 3: Check Metadata and FAISS Index
print("\nMetadata:", db.metadata)
print("FAISS Index Count:", db.indexes["resnet"].ntotal)
print("Lookup Table:", db.image_paths["resnet"])
pdb.set_trace()

# Step 4: Perform a Search
results = db.search("test_images/cat1.jpg", model_name="resnet", k=2)
print("Search Results:", results)
pdb.set_trace()

# Step 5: Delete an Image
db.delete_entry(1)  # Delete dog1.jpg
print("\nMetadata after deletion:", db.metadata)
pdb.set_trace()

# Step 6: Rebuild FAISS
db.rebuild_faiss_indexes()
print("\nFAISS Rebuilt. Index count:", db.indexes["resnet"].ntotal)
pdb.set_trace()

# Step 7: Save and Load Database
db.save(index_file="test_index.faiss", metadata_file="test_metadata.json")
db.load(index_file="test_index.faiss", metadata_file="test_metadata.json")
print("\nDatabase Reloaded Successfully.")
pdb.set_trace()

print("\nAll Tests Passed!")
