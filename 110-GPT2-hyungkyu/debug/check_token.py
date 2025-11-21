import json

# Load vocabulary
with open("../assets/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Reverse mapping: id -> token
id_to_token = {v: k for k, v in vocab.items()}

# Check suspicious tokens
token_ids = [284, 287, 257, 262, 379, 307]

print("Token IDs and their corresponding tokens:")
for token_id in token_ids:
    token = id_to_token.get(token_id, "UNKNOWN")
    # Decode special characters
    token_decoded = token.replace("Ġ", " ")  # GPT-2 uses Ġ for spaces
    print(f"  Token {token_id}: '{token}' -> '{token_decoded}'")
