from datasets import load_from_disk

# Step 1: Load the datasets
train_dataset = load_from_disk('_data/therapy_train')
test_dataset = load_from_disk('_data/therapy_test')
val_dataset = load_from_disk('_data/therapy_val')

# Step 2: Display the entire datasets
print("Training Dataset:")
print(train_dataset)
print("\nTest Dataset:")
print(test_dataset)
print("\nValidation Dataset:")
print(val_dataset)

# Optional: Show a few examples from each dataset
print("\nFirst 5 examples from Training Dataset:")
for i in range(min(10, len(train_dataset))):
    print(train_dataset[i])

# print("\nFirst 5 examples from Test Dataset:")
# for i in range(min(5, len(test_dataset))):
#     print(test_dataset[i])

# print("\nFirst 5 examples from Validation Dataset:")
# for i in range(min(5, len(val_dataset))):
#     print(val_dataset[i])