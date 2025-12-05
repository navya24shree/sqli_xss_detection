import pandas as pd

def load_payloads(path, label):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({"payload": lines, "label": label})

# Load each category
df_sqli = load_payloads("sqli_time_based_payload.txt", "sqli")
df_xss = load_payloads("xss_payload.txt", "xss")
df_benign = load_payloads("benign_payload.txt", "benign")

# Combine them
df = pd.concat([df_sqli, df_xss, df_benign], ignore_index=True)

# Save to CSV
df.to_csv("payload_dataset1.csv", index=False)

print("CSV file created successfully!")
print("Total samples:", len(df))
