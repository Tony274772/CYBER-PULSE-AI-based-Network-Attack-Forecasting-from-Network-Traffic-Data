from huggingface_hub import hf_hub_download

repo_id = "bencorn/CICIDS2017"
filename = "pcaps/Wednesday-workingHours.pcap"   # note: lowercase "w" in "workingHours" — matches the official file name

local_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=filename,
)
print(local_path)